"""Daily signal email: compose and send the morning prediction summary.

Composition and transport are separate so the email body is fully testable
offline. Transport is pluggable, selected by which env vars are present:

- **Resend HTTP API** (primary): `TWOPERCENT_RESEND_API_KEY` — a scoped,
  revocable, send-only key; no personal account credential ever touches this
  box. POST via stdlib urllib, no new dependency.
- **Generic SMTP** (fallback): `TWOPERCENT_SMTP_HOST`/`_PORT`/`_USER`/
  `_PASSWORD` over STARTTLS — documented for a DEDICATED sending account
  only, never a personal one.
- Both configured → Resend wins and the routine step WARNs that the SMTP
  config is ignored. Neither → the routine step skips loudly.

Security posture (this module handles credentials):

- Secrets come ONLY from the environment / a gitignored .env file — never
  argv, never logged. SMTPAuthenticationError becomes a fixed message and
  Resend HTTP errors quote only the status code and the API's error message,
  so no credential or raw exception repr reaches a routine summary line.
- `scrub` exists for the generic-exception path: any detail string headed
  for a log or summary is passed through it for every configured secret.
- No shell subprocesses; smtplib STARTTLS or HTTPS only.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import math
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from html import escape
from pathlib import Path

from twopercent.predict import PredictResult
from twopercent.track import PickPerformance

logger = logging.getLogger(__name__)

ENV_TO = "TWOPERCENT_EMAIL_TO"
ENV_FROM = "TWOPERCENT_EMAIL_FROM"
ENV_RESEND_KEY = "TWOPERCENT_RESEND_API_KEY"
ENV_SMTP_HOST = "TWOPERCENT_SMTP_HOST"
ENV_SMTP_PORT = "TWOPERCENT_SMTP_PORT"
ENV_SMTP_USER = "TWOPERCENT_SMTP_USER"
ENV_SMTP_PASSWORD = "TWOPERCENT_SMTP_PASSWORD"  # noqa: S105 — env var NAME, not a credential
# Anchored to the repo root (src/twopercent/notify.py -> parents[2]), not the
# CWD, so a run launched from any directory still finds the same config.
# Real environment variables always win over the file.
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

RESEND_URL = "https://api.resend.com/emails"
DEFAULT_SMTP_PORT = 587

BASKET_SIZE = 5
TABLE_ROWS = 10
ATTACHMENT_NAME = "dashboard.html"

_DISCLAIMER = (
    "This message is automated model output from the twopercent research "
    "system. Model scores are statistical estimates, not calibrated "
    "probabilities — on any given day most named candidates will not make a "
    "2% move. Simulated and benchmark results assume perfect fills at the "
    "open and close with estimated round-trip trading costs, and are subject "
    "to survivorship bias in the historical candidate pool and to regime "
    "change. Nothing in this message is investment advice."
)


class SendError(RuntimeError):
    """Send failure with a message already safe for summaries/logs."""


def scrub(detail: str, secret: str) -> str:
    """Redact a secret from any string headed for a log or summary."""
    return detail.replace(secret, "[redacted]") if secret else detail


@dataclass
class EmailConfig:
    to: list[str]
    sender: str
    transport: str  # "resend" | "smtp"
    resend_api_key: str = ""
    smtp_host: str = ""
    smtp_port: int = DEFAULT_SMTP_PORT
    smtp_user: str = ""
    smtp_password: str = ""
    ignored_smtp: bool = field(default=False)  # both transports set; Resend won

    def secrets(self) -> list[str]:
        """Every configured secret — scrub ALL of them from any outbound text."""
        return [s for s in (self.resend_api_key, self.smtp_password) if s]


def _load_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE .env parser (stdlib only — no python-dotenv dep).

    Blank lines and full-line comments are skipped; values may be quoted.
    Real environment variables always win over the file (see email_config).
    """
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            parsed[key] = value
    return parsed


def email_config(env_path: Path | str | None = None) -> tuple[EmailConfig | None, str]:
    """Resolve email settings from the environment, falling back to .env.

    Returns (config, "") when fully configured, or (None, detail) naming what
    is missing — the routine's skip line quotes that detail, so an operator
    sees exactly which variable to set. Present-but-invalid values (bad
    addresses, partial SMTP config, junk port) raise ValueError:
    misconfiguration must WARN, never silently skip as "not configured".
    """
    file_vars = _load_env_file(DEFAULT_ENV_PATH if env_path is None else Path(env_path))

    def _get(name: str) -> str:
        return os.environ.get(name, "").strip() or file_vars.get(name, "")

    to_raw, sender = _get(ENV_TO), _get(ENV_FROM)
    resend_key = _get(ENV_RESEND_KEY)
    smtp_host, smtp_user = _get(ENV_SMTP_HOST), _get(ENV_SMTP_USER)
    smtp_password, smtp_port_raw = _get(ENV_SMTP_PASSWORD), _get(ENV_SMTP_PORT)
    smtp_vars = [
        (ENV_SMTP_HOST, smtp_host),
        (ENV_SMTP_USER, smtp_user),
        (ENV_SMTP_PASSWORD, smtp_password),
    ]
    smtp_any = any(v for _, v in smtp_vars) or bool(smtp_port_raw)
    smtp_complete = all(v for _, v in smtp_vars)

    missing = [name for name, value in ((ENV_TO, to_raw), (ENV_FROM, sender)) if not value]
    if not resend_key and not smtp_any:
        missing.append(
            f"{ENV_RESEND_KEY} (or the {ENV_SMTP_HOST}/{ENV_SMTP_USER}/{ENV_SMTP_PASSWORD} trio)"
        )
    if missing:
        return None, f"{', '.join(missing)} unset"

    if not resend_key and not smtp_complete:
        absent = ", ".join(name for name, value in smtp_vars if not value)
        raise ValueError(f"partial SMTP configuration — {absent} unset (and no {ENV_RESEND_KEY})")
    try:
        smtp_port = int(smtp_port_raw) if smtp_port_raw else DEFAULT_SMTP_PORT
    except ValueError:
        raise ValueError(f"invalid {ENV_SMTP_PORT}: {smtp_port_raw!r} is not a port") from None

    recipients = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
    bad = [addr for addr in recipients if "@" not in addr]
    if not recipients or bad:
        raise ValueError(
            f"invalid recipient address(es) in {ENV_TO}: {', '.join(bad) or 'none provided'}"
        )
    if "@" not in sender:
        raise ValueError(f"invalid sender address in {ENV_FROM}")

    transport = "resend" if resend_key else "smtp"
    ignored_smtp = bool(resend_key) and smtp_any
    if ignored_smtp:
        logger.warning(
            "both Resend and SMTP transports are configured — Resend takes "
            "precedence, the SMTP settings are IGNORED"
        )
    return (
        EmailConfig(
            to=recipients,
            sender=sender,
            transport=transport,
            resend_api_key=resend_key,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            ignored_smtp=ignored_smtp,
        ),
        "",
    )


def _next_weekday(day: dt.date) -> dt.date:
    day += dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day


def target_trading_day(signal_date: dt.date, generated_at: dt.datetime) -> dt.date:
    """The day being predicted: the trading day after `signal_date`.

    The predict routine runs pre-open on the target day itself, so a weekday
    run strictly after the signal date IS the target day. A weekend/holiday
    run (or a post-close run on the signal date) targets the next weekday.
    Exchange holidays are not modelled — same simplification as the clock
    gate — so a holiday-morning run names the holiday; the store's freshness
    gate bounds how far this can drift.
    """
    today = generated_at.date()
    if today.weekday() < 5 and today > signal_date:
        return today
    return _next_weekday(max(today, signal_date))


def _long_date(day: dt.date) -> str:
    return f"{day:%A}, {day:%B} {day.day}, {day:%Y}"


def _time_et(generated_at: dt.datetime) -> str:
    hour = generated_at.hour % 12 or 12
    return f"{hour}:{generated_at:%M} {'AM' if generated_at.hour < 12 else 'PM'}"


_RATIONALE = (
    "in walk-forward simulation the top-5 equal-weight basket was the only "
    "configuration with a positive compounded return net of assumed trading costs."
)


def _basket_note(n_basket: int) -> str:
    if n_basket == 0:
        return (
            "No candidates cleared today's ranking — no trade is suggested. "
            "This is stated outright rather than papered over."
        )
    if n_basket < BASKET_SIZE:
        return (
            f"Only {n_basket} candidate(s) cleared today's ranking, fewer than the "
            f"usual {BASKET_SIZE} — the basket below is smaller than usual, not padded."
        )
    return ""


def _benchmark_summary(benchmark: tuple[int, dict, dt.date | None, dt.date | None] | None) -> str:
    """One sentence of honest walk-forward stats — every number from the
    ledger row, never hard-coded; missing data says so instead of guessing."""
    if benchmark is None:
        return (
            "No standard walk-forward benchmark is recorded in the experiments "
            "ledger for this strategy yet, so no historical precision figures are quoted."
        )
    _exp_id, metrics, test_start, test_end = benchmark
    p5 = metrics.get("precision_at_5")
    base = metrics.get("base_rate")
    window = f"{test_start} to {test_end}"

    def _finite(value) -> bool:
        # NaN survives the JSON round-trip and passes a `<= 0` check
        # (NaN comparisons are False) — it must never reach the email
        # as "nan% — a nanx lift".
        return isinstance(value, int | float) and math.isfinite(value)

    if not _finite(p5) or not _finite(base) or base <= 0:
        return (
            f"The latest recorded benchmark (test window {window}) is missing "
            "precision_at_5/base_rate figures, so no precision claim is made."
        )
    return (
        f"Over the walk-forward test window {window}, the top-5 basket hit the "
        f"2% target on {p5:.1%} of picks versus an all-names base rate of "
        f"{base:.1%} — a {p5 / base:.1f}x lift."
    )


def _live_record_line(perf: PickPerformance) -> str:
    n_live = len(perf.live)
    if n_live == 0:
        return "The live track record begins today — no live trading days have been scored yet."
    return f"Live track record: {n_live} live trading day(s) scored to date."


def _system_summary_lines(strategy: str, benchmark, perf: PickPerformance) -> list[str]:
    return [
        f"Strategy {strategy}: a machine-learned ranking model retrained each "
        "morning on all labeled history, scoring liquid US names for the "
        "probability of a 2%+ open-to-close move on the target day.",
        _benchmark_summary(benchmark),
        _live_record_line(perf),
    ]


def compose_signal_email(
    prediction: PredictResult,
    perf: PickPerformance,
    benchmark: tuple[int, dict, dt.date | None, dt.date | None] | None,
    generated_at: dt.datetime,
) -> tuple[str, str, str]:
    """Build (subject, text_body, html_body) for the day's signal.

    `generated_at` must be an America/New_York timestamp; the subject names
    the TARGET day (the day being predicted), not the signal date.
    """
    target = target_trading_day(prediction.signal_date, generated_at)
    date_str = _long_date(target)
    subject = f"twopercent Daily Signal — {date_str}"
    header_line = f"{date_str} — generated pre-open at {_time_et(generated_at)} ET"
    signal_line = f"Signal computed from market data through {prediction.signal_date}."

    top10 = prediction.scored.head(TABLE_ROWS)
    basket = list(prediction.scored.head(BASKET_SIZE)["symbol"])
    basket_note = _basket_note(len(basket))
    suggestion = (
        f"Equal-weight basket of the model's top {len(basket)} candidate(s), "
        "bought at the open and exited at the close: " + (", ".join(basket) if basket else "(none)")
    )
    summary_lines = _system_summary_lines(prediction.strategy, benchmark, perf)

    text_rows = [
        f"{rank:>5}  {symbol:<8}{prob:.3f}"
        for rank, symbol, prob in zip(top10["rank"], top10["symbol"], top10["prob"], strict=True)
    ]
    text_body = "\n".join(
        [
            "twopercent Daily Signal",
            header_line,
            signal_line,
            "",
            "TRADE SUGGESTION",
            suggestion,
            *([basket_note] if basket_note else []),
            f"Why a top-{BASKET_SIZE} basket: {_RATIONALE}",
            "",
            f"TOP {TABLE_ROWS} CANDIDATES",
            f"{'rank':>5}  {'ticker':<8}{'score'}",
            *(text_rows or ["(no candidates today)"]),
            "",
            "SYSTEM SUMMARY",
            *summary_lines,
            "",
            "--",
            _DISCLAIMER,
        ]
    )

    cell = 'style="border:1px solid #ccc;padding:4px 10px;text-align:left;"'
    html_rows = (
        "".join(
            f"<tr><td {cell}>{rank}</td><td {cell}>{escape(str(symbol))}</td>"
            f"<td {cell}>{prob:.3f}</td></tr>"
            for rank, symbol, prob in zip(
                top10["rank"], top10["symbol"], top10["prob"], strict=True
            )
        )
        or f'<tr><td {cell} colspan="3">(no candidates today)</td></tr>'
    )
    note_html = f"<p><em>{escape(basket_note)}</em></p>" if basket_note else ""
    summary_html = "".join(f"<p>{escape(line)}</p>" for line in summary_lines)
    html_body = (
        # MIME/JSON transport already declares utf-8, but the in-body
        # declaration survives clients that ignore transport headers
        # (project standard: generated HTML always declares its charset).
        '<meta charset="utf-8">'
        "<div style=\"font-family:Georgia,'Times New Roman',serif;max-width:640px;"
        'margin:0 auto;color:#1a1a1a;line-height:1.5;">'
        '<h2 style="margin-bottom:2px;">twopercent Daily Signal</h2>'
        f'<p style="margin-top:0;color:#555;">{escape(header_line)}<br>'
        f"{escape(signal_line)}</p>"
        '<h3 style="border-bottom:1px solid #ccc;padding-bottom:2px;">Trade Suggestion</h3>'
        f"<p>{escape(suggestion)}</p>"
        f"{note_html}"
        f"<p>Why a top-{BASKET_SIZE} basket: {escape(_RATIONALE)}</p>"
        f'<h3 style="border-bottom:1px solid #ccc;padding-bottom:2px;">'
        f"Top {TABLE_ROWS} Candidates</h3>"
        '<table style="border-collapse:collapse;border:1px solid #ccc;">'
        f"<tr><th {cell}>Rank</th><th {cell}>Ticker</th><th {cell}>Model score</th></tr>"
        f"{html_rows}</table>"
        '<h3 style="border-bottom:1px solid #ccc;padding-bottom:2px;">System Summary</h3>'
        f"{summary_html}"
        f'<p style="color:#777;font-size:13px;border-top:1px solid #ccc;'
        f'padding-top:8px;">{escape(_DISCLAIMER)}</p>'
        "</div>"
    )
    return subject, text_body, html_body


@dataclass
class SendOutcome:
    recipients: list[str]
    attached: bool
    transport: str


def _resend_error_message(exc: urllib.error.HTTPError) -> str:
    """Resend's error message from the response body — never the API key."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        message = parsed.get("message") or parsed.get("error") or body
    except Exception:
        message = "(unreadable error body)"
    return str(message)[:200]


def _send_resend(
    config: EmailConfig,
    subject: str,
    text_body: str,
    html_body: str,
    attachment: bytes | None,
) -> None:
    payload: dict = {
        "from": config.sender,
        "to": config.to,
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }
    if attachment is not None:
        payload["attachments"] = [
            {
                "filename": ATTACHMENT_NAME,
                "content": base64.b64encode(attachment).decode("ascii"),
            }
        ]
    req = urllib.request.Request(
        RESEND_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.resend_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise SendError(
            f"Resend API rejected the send: HTTP {exc.code} — {_resend_error_message(exc)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SendError(f"Resend API unreachable: {exc.reason}") from exc
    if status // 100 != 2:  # defensive: urlopen raises on non-2xx, but never trust that silently
        raise SendError(f"Resend API returned HTTP {status}")


def _send_smtp(
    config: EmailConfig,
    subject: str,
    text_body: str,
    html_body: str,
    attachment: bytes | None,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.sender
    msg["To"] = ", ".join(config.to)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    if attachment is not None:
        msg.add_attachment(
            attachment.decode("utf-8", errors="replace"),
            subtype="html",
            filename=ATTACHMENT_NAME,
        )
    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=60) as smtp:
            # An explicit verifying context: bare starttls() uses an
            # UNVERIFIED context (CERT_NONE), letting an on-path MITM with a
            # self-signed cert harvest the password from the unattended run.
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(config.smtp_user, config.smtp_password)
            smtp.send_message(msg, from_addr=config.sender, to_addrs=config.to)
    except smtplib.SMTPAuthenticationError as exc:
        raise SendError("authentication failed (check the SMTP password)") from exc


def send_signal_email(
    config: EmailConfig,
    subject: str,
    text_body: str,
    html_body: str,
    dashboard_path: Path | str,
) -> SendOutcome:
    """Send the composed signal via the configured transport.

    The dashboard is attached when present; a missing file WARNS loudly and
    the email still goes out (the body is complete on its own) — the caller
    surfaces the missing attachment in its step summary, never silently.
    Transport failures raise SendError with messages already safe for
    summaries (status codes and API error text, never credentials).
    """
    path = Path(dashboard_path)
    attachment = path.read_bytes() if path.is_file() else None
    if attachment is None:
        logger.warning(
            "dashboard attachment missing at %s — sending the signal email WITHOUT it", path
        )
    if config.transport == "resend":
        _send_resend(config, subject, text_body, html_body, attachment)
    else:
        _send_smtp(config, subject, text_body, html_body, attachment)
    logger.info(
        "signal email sent via %s to %d recipient(s)%s",
        config.transport,
        len(config.to),
        "" if attachment is not None else " WITHOUT dashboard attachment",
    )
    return SendOutcome(
        recipients=list(config.to), attached=attachment is not None, transport=config.transport
    )
