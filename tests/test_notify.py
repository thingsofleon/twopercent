"""Offline tests for the daily signal email: composition, config, transports.

smtplib.SMTP and urllib.request.urlopen are monkeypatched everywhere —
nothing here may dial out.
"""

import base64
import datetime as dt
import io
import json
import re
import smtplib
import ssl
import sys
import types
import urllib.error

import pandas as pd
import pytest

from twopercent import backtest, notify, store
from twopercent.predict import PredictResult
from twopercent.track import PickPerformance

FRIDAY_0800_ET = dt.datetime(2026, 7, 17, 8, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))

METRICS = {
    "precision_at_5": 0.1234,
    "base_rate": 0.0567,
    "precision_at_n": 0.09,
    "lift": 1.59,
    "auc": 0.61,
}

ALL_EMAIL_VARS = (
    notify.ENV_TO,
    notify.ENV_FROM,
    notify.ENV_RESEND_KEY,
    notify.ENV_SMTP_HOST,
    notify.ENV_SMTP_PORT,
    notify.ENV_SMTP_USER,
    notify.ENV_SMTP_PASSWORD,
)


def _prediction(n: int = 12, signal_date: dt.date = dt.date(2026, 7, 16)) -> PredictResult:
    scored = pd.DataFrame(
        {
            "symbol": [f"TCK{i:02d}" for i in range(n)],
            "prob": [0.912 - 0.05 * i for i in range(n)],
            "rank": range(1, n + 1),
        }
    )
    return PredictResult("test_strat_v1", signal_date, scored, trained_rows=12345)


def _perf(n_live: int = 0, n_late: int = 0) -> PickPerformance:
    if n_live + n_late == 0:
        return PickPerformance(daily=pd.DataFrame())
    return PickPerformance(daily=pd.DataFrame({"late": [False] * n_live + [True] * n_late}))


def _seed_benchmark(con, strategy="test_strat_v1", metrics=METRICS, **params):
    return store.record_experiment(
        con,
        strategy=strategy,
        params={"months": 12, "top_n": 20, "strategy_params": {}, **params},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics=metrics,
    )


# --- shared ledger reader -----------------------------------------------------


def test_latest_standard_experiment_skips_variants_and_nonstandard(con):
    wanted = _seed_benchmark(con)
    _seed_benchmark(con, metrics={**METRICS, "precision_at_5": 0.999}, strategy_params={"d": 4})
    _seed_benchmark(con, metrics={**METRICS, "precision_at_5": 0.888}, months=2)
    latest = backtest.latest_standard_experiment(con, "test_strat_v1")
    assert latest is not None
    exp_id, metrics, test_start, test_end = latest
    assert exp_id == wanted
    assert metrics["precision_at_5"] == 0.1234  # never the variant's 0.999
    assert (test_start, test_end) == (dt.date(2025, 7, 1), dt.date(2026, 6, 30))


def test_latest_standard_experiment_none_when_unrecorded(con):
    assert backtest.latest_standard_experiment(con, "ghost_strat") is None


# --- composition --------------------------------------------------------------


def test_subject_names_the_target_day_not_the_signal_day():
    subject, _, _ = notify.compose_signal_email(
        _prediction(signal_date=dt.date(2026, 7, 16)), _perf(), None, FRIDAY_0800_ET
    )
    assert subject == "twopercent Daily Signal — Friday, July 17, 2026"


def test_weekend_run_targets_next_monday():
    saturday = dt.datetime(2026, 7, 18, 10, 0, tzinfo=FRIDAY_0800_ET.tzinfo)
    subject, _, _ = notify.compose_signal_email(
        _prediction(signal_date=dt.date(2026, 7, 17)), _perf(), None, saturday
    )
    assert "Monday, July 20, 2026" in subject


def test_bodies_carry_basket_table_and_generated_time():
    _, text, html = notify.compose_signal_email(_prediction(), _perf(), None, FRIDAY_0800_ET)
    for body in (text, html):
        for ticker in ("TCK00", "TCK01", "TCK02", "TCK03", "TCK04"):
            assert ticker in body  # top-5 basket present in both alternatives
        assert "TCK09" in body and "TCK10" not in body  # table is top-10, no more
        assert "0.912" in body  # model score to 3dp
        assert "generated pre-open at 8:00 AM ET" in body
        assert "Nothing in this message is investment advice" in body
        assert "not calibrated probabilities" in body


def test_stats_come_from_the_ledger_never_hardcoded(con):
    _seed_benchmark(con)
    bench = backtest.latest_standard_experiment(con, "test_strat_v1")
    _, text, html = notify.compose_signal_email(_prediction(), _perf(), bench, FRIDAY_0800_ET)
    for body in (text, html):
        assert "12.3%" in body  # precision_at_5 = 0.1234, seeded
        assert "5.7%" in body  # base_rate = 0.0567, seeded
        assert "2.2x" in body  # lift = 0.1234 / 0.0567
        assert "2025-07-01 to 2026-06-30" in body  # test window from the row


def test_no_recorded_benchmark_says_so_instead_of_inventing_numbers():
    _, text, html = notify.compose_signal_email(_prediction(), _perf(), None, FRIDAY_0800_ET)
    for body in (text, html):
        assert "No standard walk-forward benchmark is recorded" in body
        # no precision/base-rate style figure (e.g. "12.3%") appears anywhere
        assert not re.search(r"\d+\.\d+%", body)


def test_nan_metrics_fall_back_instead_of_emailing_nan(con):
    # NaN survives the JSON round-trip and passes a `<= 0` check (NaN
    # comparisons are False) — the email must never read "nan% — a nanx lift".
    _seed_benchmark(con, metrics={**METRICS, "precision_at_5": float("nan")})
    bench = backtest.latest_standard_experiment(con, "test_strat_v1")
    _, text, html = notify.compose_signal_email(_prediction(), _perf(), bench, FRIDAY_0800_ET)
    for body in (text, html):
        assert "nan" not in body.lower()
        assert "no precision claim is made" in body


def test_zero_live_days_says_track_record_begins_today():
    _, text, html = notify.compose_signal_email(_prediction(), _perf(0), None, FRIDAY_0800_ET)
    for body in (text, html):
        assert "live track record begins today" in body.lower()


def test_live_day_count_excludes_late_days():
    _, text, html = notify.compose_signal_email(
        _prediction(), _perf(n_live=7, n_late=3), None, FRIDAY_0800_ET
    )
    for body in (text, html):
        assert "7 live trading day(s)" in body


def test_thin_day_states_the_shortfall_and_never_pads():
    _, text, html = notify.compose_signal_email(_prediction(n=3), _perf(), None, FRIDAY_0800_ET)
    for body in (text, html):
        assert "Only 3 candidate(s)" in body
        assert "not padded" in body
        assert "TCK03" not in body  # nothing invented past the real candidates


# --- dashboard rendering (playwright mocked — CI has no browsers) --------------


def _install_fake_playwright(monkeypatch, screenshot_bytes=b"fake-png", launch_exc=None):
    """A fake playwright.sync_api in sys.modules recording the render calls."""
    calls: dict = {}

    class _Page:
        def emulate_media(self, color_scheme=None):
            calls["color_scheme"] = color_scheme

        def goto(self, url, wait_until=None):
            calls["url"] = url

        def screenshot(self, full_page=False, type=None):
            calls["full_page"] = full_page
            return screenshot_bytes

    class _Browser:
        def new_page(self, viewport=None, device_scale_factor=None, java_script_enabled=None):
            calls["viewport"] = viewport
            calls["device_scale_factor"] = device_scale_factor
            calls["java_script_enabled"] = java_script_enabled
            return _Page()

        def close(self):
            calls["closed"] = True

    class _Chromium:
        def launch(self, headless=True):
            if launch_exc is not None:
                raise launch_exc
            calls["headless"] = headless
            return _Browser()

    class _Playwright:
        chromium = _Chromium()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module = types.ModuleType("playwright.sync_api")
    module.sync_playwright = lambda: _Playwright()
    monkeypatch.setitem(sys.modules, "playwright.sync_api", module)
    return calls


def test_render_missing_dashboard_raises_render_unavailable(tmp_path):
    with pytest.raises(notify.RenderUnavailable, match="not found"):
        notify.render_dashboard_png(tmp_path / "nope.html")


def test_render_playwright_not_importable_raises_render_unavailable(monkeypatch, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash.write_text("<h1>x</h1>", encoding="utf-8")
    # None in sys.modules makes the lazy import raise ImportError even though
    # the package is installed on this box.
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(notify.RenderUnavailable, match="playwright not installed"):
        notify.render_dashboard_png(dash)


def test_render_browser_launch_failure_becomes_render_unavailable_first_line(monkeypatch, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash.write_text("<h1>x</h1>", encoding="utf-8")
    _install_fake_playwright(
        monkeypatch,
        launch_exc=RuntimeError(
            "Executable doesn't exist at /nowhere/chrome\nrun `playwright install`\nmore lines"
        ),
    )
    with pytest.raises(notify.RenderUnavailable) as excinfo:
        notify.render_dashboard_png(dash)
    message = str(excinfo.value)
    assert "browser render failed" in message
    assert "Executable doesn't exist" in message
    assert "more lines" not in message  # first line only — summary-safe


def test_render_empty_screenshot_raises_render_unavailable(monkeypatch, tmp_path):
    # b"" would sail past the size cap, then die in compose_dashboard_email
    # AFTER the fallback point — the one render failure that would skip the
    # fallback that exists for it.
    dash = tmp_path / "dashboard.html"
    dash.write_text("<h1>x</h1>", encoding="utf-8")
    _install_fake_playwright(monkeypatch, screenshot_bytes=b"")
    with pytest.raises(notify.RenderUnavailable, match="empty screenshot"):
        notify.render_dashboard_png(dash)


def test_render_oversize_png_raises_render_unavailable_with_size(monkeypatch, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash.write_text("<h1>x</h1>", encoding="utf-8")
    _install_fake_playwright(monkeypatch, screenshot_bytes=b"x" * (notify.MAX_PNG_BYTES + 1))
    with pytest.raises(notify.RenderUnavailable, match=r"20\.0MB"):
        notify.render_dashboard_png(dash)


def test_render_happy_path_dark_fullpage_at_declared_width(monkeypatch, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash.write_text("<h1>x</h1>", encoding="utf-8")
    calls = _install_fake_playwright(monkeypatch, screenshot_bytes=b"PNG-SENTINEL")
    assert notify.render_dashboard_png(dash) == b"PNG-SENTINEL"
    assert calls["color_scheme"] == "dark"
    assert calls["java_script_enabled"] is False  # no script surface in the unattended render
    assert calls["full_page"] is True
    assert calls["headless"] is True
    assert calls["viewport"]["width"] == notify.RENDER_VIEWPORT_WIDTH
    assert calls["device_scale_factor"] == 2
    assert calls["url"].startswith("file://")
    assert calls["closed"] is True  # browser closed even on success


# --- dashboard-image composition ------------------------------------------------


def test_compose_dashboard_email_subject_matches_text_email_and_body_is_the_image():
    subject, text, html = notify.compose_dashboard_email(
        _prediction(signal_date=dt.date(2026, 7, 16)), FRIDAY_0800_ET, b"png"
    )
    assert subject == "twopercent Daily Signal — Friday, July 17, 2026"
    assert '<meta charset="utf-8">' in html
    assert 'src="cid:dashboard"' in html
    assert "max-width:900px" in html
    assert 'alt="twopercent dashboard — Friday, July 17, 2026"' in html
    # minimal shell: no composed sections — the dashboard carries its own text
    assert "TCK00" not in html and "Trade Suggestion" not in html
    assert text == (
        "twopercent Daily Signal for Friday, July 17, 2026. This email is the "
        "rendered dashboard image; view with an HTML-capable client."
    )


def test_compose_dashboard_email_refuses_empty_png():
    with pytest.raises(ValueError, match="png_bytes is empty"):
        notify.compose_dashboard_email(_prediction(), FRIDAY_0800_ET, b"")


# --- config / transport selection ---------------------------------------------


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in ALL_EMAIL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(notify, "DEFAULT_ENV_PATH", tmp_path / "absent.env")
    return tmp_path


def _set_common(monkeypatch):
    monkeypatch.setenv(notify.ENV_TO, "leon@example.com")
    monkeypatch.setenv(notify.ENV_FROM, "signals@example.com")


def _set_smtp(monkeypatch):
    monkeypatch.setenv(notify.ENV_SMTP_HOST, "smtp.example.com")
    monkeypatch.setenv(notify.ENV_SMTP_USER, "sender@example.com")
    monkeypatch.setenv(notify.ENV_SMTP_PASSWORD, "smtp-pw")


def test_default_env_path_is_anchored_to_the_repo_root():
    # CWD-relative ".env" would silently lose the config for any run
    # launched from another directory (real env vars still win over it).
    assert notify.DEFAULT_ENV_PATH.is_absolute()
    assert (notify.DEFAULT_ENV_PATH.parent / "pyproject.toml").is_file()


def test_config_neither_transport_names_whats_missing(clean_env, monkeypatch):
    _set_common(monkeypatch)
    config, missing = notify.email_config()
    assert config is None
    assert notify.ENV_RESEND_KEY in missing and notify.ENV_SMTP_HOST in missing


def test_config_names_every_missing_common_variable(clean_env):
    config, missing = notify.email_config()
    assert config is None
    for var in (notify.ENV_TO, notify.ENV_FROM, notify.ENV_RESEND_KEY):
        assert var in missing


def test_config_resend_only_selects_resend(clean_env, monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_RESEND_KEY, "re_key_1")
    config, missing = notify.email_config()
    assert missing == ""
    assert config.transport == "resend"
    assert not config.ignored_smtp
    assert config.secrets() == ["re_key_1"]


def test_config_smtp_only_selects_smtp_with_default_port(clean_env, monkeypatch):
    _set_common(monkeypatch)
    _set_smtp(monkeypatch)
    config, _ = notify.email_config()
    assert config.transport == "smtp"
    assert config.smtp_port == 587  # default
    monkeypatch.setenv(notify.ENV_SMTP_PORT, "2525")
    config, _ = notify.email_config()
    assert config.smtp_port == 2525


def test_config_both_transports_resend_wins_and_flags_ignored_smtp(clean_env, monkeypatch):
    _set_common(monkeypatch)
    _set_smtp(monkeypatch)
    monkeypatch.setenv(notify.ENV_RESEND_KEY, "re_key_1")
    config, _ = notify.email_config()
    assert config.transport == "resend"
    assert config.ignored_smtp
    assert set(config.secrets()) == {"re_key_1", "smtp-pw"}


def test_config_partial_smtp_raises_instead_of_silently_skipping(clean_env, monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_SMTP_HOST, "smtp.example.com")  # no user/password
    with pytest.raises(ValueError, match="partial SMTP configuration"):
        notify.email_config()


def test_config_junk_port_raises(clean_env, monkeypatch):
    _set_common(monkeypatch)
    _set_smtp(monkeypatch)
    monkeypatch.setenv(notify.ENV_SMTP_PORT, "not-a-port")
    with pytest.raises(ValueError, match="not a port"):
        notify.email_config()


def test_config_reads_env_file_but_environment_wins(clean_env, monkeypatch):
    env_file = clean_env / ".env"
    env_file.write_text(
        "# comment line\n\n"
        f"{notify.ENV_TO}=file@example.com\n"
        f'{notify.ENV_FROM}="signals@example.com"\n'
        f"{notify.ENV_RESEND_KEY}=re_from_file\n"
    )
    config, missing = notify.email_config(env_path=env_file)
    assert missing == ""
    assert config.to == ["file@example.com"]
    assert config.sender == "signals@example.com"  # quotes stripped
    assert config.resend_api_key == "re_from_file"

    monkeypatch.setenv(notify.ENV_TO, "real@example.com")
    config, _ = notify.email_config(env_path=env_file)
    assert config.to == ["real@example.com"]  # environment beats the file


def test_config_splits_comma_separated_recipients(clean_env, monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_TO, "a@example.com, b@example.com")
    monkeypatch.setenv(notify.ENV_RESEND_KEY, "re_key")
    config, _ = notify.email_config()
    assert config.to == ["a@example.com", "b@example.com"]


def test_config_rejects_addresses_without_at(clean_env, monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_TO, "a@example.com,nonsense")
    monkeypatch.setenv(notify.ENV_RESEND_KEY, "re_key")
    with pytest.raises(ValueError, match="nonsense"):
        notify.email_config()


# --- sending: Resend ----------------------------------------------------------


def _resend_config(to=None, key="re_test_key_123"):
    return notify.EmailConfig(
        to=to or ["leon@example.com"],
        sender="signals@example.com",
        transport="resend",
        resend_api_key=key,
    )


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def resend_capture(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_resend_posts_json_with_bearer_auth_and_attachment(resend_capture, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash_bytes = b'<meta charset="utf-8"><h1>DASH-CONTENT</h1>'
    dash.write_bytes(dash_bytes)
    outcome = notify.send_signal_email(_resend_config(), "subj", "text-body", "<p>html</p>", dash)
    assert outcome.attached and outcome.transport == "resend"

    req = resend_capture["req"]
    assert req.full_url == "https://api.resend.com/emails"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer re_test_key_123"
    assert req.get_header("Content-type") == "application/json"

    payload = json.loads(req.data)
    assert payload["from"] == "signals@example.com"
    assert payload["to"] == ["leon@example.com"]
    assert payload["subject"] == "subj"
    assert payload["text"] == "text-body"
    assert payload["html"] == "<p>html</p>"
    assert len(payload["attachments"]) == 1
    assert payload["attachments"][0]["filename"] == "dashboard.html"
    # the base64 content round-trips to the exact dashboard bytes
    assert base64.b64decode(payload["attachments"][0]["content"]) == dash_bytes


def test_resend_missing_dashboard_warns_and_omits_attachments(resend_capture, tmp_path, caplog):
    with caplog.at_level("WARNING", logger="twopercent.notify"):
        outcome = notify.send_signal_email(
            _resend_config(), "subj", "t", "<p>h</p>", tmp_path / "nope.html"
        )
    assert not outcome.attached
    assert any("WITHOUT" in rec.message for rec in caplog.records)
    payload = json.loads(resend_capture["req"].data)
    assert "attachments" not in payload  # never an empty/padded attachment


def test_resend_http_error_raises_status_and_message_never_the_key(monkeypatch, tmp_path):
    key = "re_sentinel_key_XYZZY"
    body = io.BytesIO(json.dumps({"message": "domain is not verified"}).encode())

    def fail_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(notify.RESEND_URL, 403, "Forbidden", {}, body)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(notify.SendError) as excinfo:
        notify.send_signal_email(
            _resend_config(key=key), "subj", "t", "<p>h</p>", tmp_path / "d.html"
        )
    message = str(excinfo.value)
    assert "403" in message and "domain is not verified" in message
    assert key not in message


def test_resend_unreachable_raises_safe_message(monkeypatch, tmp_path):
    def fail_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(notify.SendError, match="unreachable"):
        notify.send_signal_email(_resend_config(), "subj", "t", "<p>h</p>", tmp_path / "d.html")


# --- sending: SMTP fallback ---------------------------------------------------


def _smtp_config(to=None):
    return notify.EmailConfig(
        to=to or ["leon@example.com"],
        sender="sender@example.com",
        transport="smtp",
        smtp_host="smtp.example.com",
        smtp_port=2525,
        smtp_user="sender@example.com",
        smtp_password="smtp-pw",
    )


class FakeSMTP:
    sent: list = []
    last = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.calls = []
        self.starttls_context = None
        type(self).last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")
        self.starttls_context = context

    def login(self, user, password):
        self.calls.append("login")

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.calls.append("send_message")
        type(self).sent.append((msg, from_addr, to_addrs))


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.sent = []
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def test_smtp_attaches_dashboard_with_name_mime_and_payload(fake_smtp, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash.write_text('<meta charset="utf-8"><h1>DASH-CONTENT</h1>', encoding="utf-8")
    outcome = notify.send_signal_email(_smtp_config(), "subj", "text", "<p>html</p>", dash)
    assert outcome.attached and outcome.transport == "smtp"
    assert fake_smtp.last.host == "smtp.example.com"
    assert fake_smtp.last.port == 2525
    (msg, from_addr, to_addrs) = fake_smtp.sent[-1]
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "dashboard.html"
    assert attachments[0].get_content_type() == "text/html"
    assert "DASH-CONTENT" in attachments[0].get_content()
    # both alternatives present alongside the attachment
    assert msg.get_body(("plain",)) is not None
    assert msg.get_body(("html",)) is not None


def test_smtp_reaches_all_comma_separated_recipients(fake_smtp, tmp_path):
    config = _smtp_config(to=["a@example.com", "b@example.com"])
    dash = tmp_path / "dashboard.html"
    dash.write_text("x", encoding="utf-8")
    outcome = notify.send_signal_email(config, "subj", "text", "<p>h</p>", dash)
    assert outcome.recipients == ["a@example.com", "b@example.com"]
    (msg, from_addr, to_addrs) = fake_smtp.sent[-1]
    assert to_addrs == ["a@example.com", "b@example.com"]
    assert msg["To"] == "a@example.com, b@example.com"
    assert from_addr == "sender@example.com"


def test_smtp_calls_starttls_before_login_with_verifying_context(fake_smtp, tmp_path):
    dash = tmp_path / "dashboard.html"
    dash.write_text("x", encoding="utf-8")
    notify.send_signal_email(_smtp_config(), "subj", "text", "<p>h</p>", dash)
    calls = fake_smtp.last.calls
    assert calls.index("starttls") < calls.index("login") < calls.index("send_message")
    # A bare starttls() would use Python's UNVERIFIED context (CERT_NONE) —
    # an on-path MITM could then harvest the password every unattended run.
    context = fake_smtp.last.starttls_context
    assert context is not None
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_smtp_auth_failure_raises_fixed_message_without_credentials(monkeypatch, tmp_path):
    class AuthFailSMTP(FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 rejected for smtp-pw")

    monkeypatch.setattr(notify.smtplib, "SMTP", AuthFailSMTP)
    with pytest.raises(notify.SendError) as excinfo:
        notify.send_signal_email(_smtp_config(), "subj", "text", "<p>h</p>", tmp_path / "d.html")
    message = str(excinfo.value)
    assert message == "authentication failed (check the SMTP password)"
    assert "535" not in message and "smtp-pw" not in message


# --- sending: dashboard-image email ------------------------------------------

DASH_HTML = '<meta charset="utf-8"><img src="cid:dashboard" alt="dash">'


def test_smtp_dashboard_email_is_related_wrapping_alternative_with_inline_cid(fake_smtp):
    png = b"\x89PNG-SENTINEL-BYTES"
    outcome = notify.send_dashboard_email(_smtp_config(), "subj", "text-alt", DASH_HTML, png)
    assert outcome.transport == "smtp" and outcome.attached
    (msg, _from_addr, _to_addrs) = fake_smtp.sent[-1]
    # the exact layout Gmail expects: related( alternative(plain, html), png )
    assert msg.get_content_type() == "multipart/related"
    alternative, image = msg.get_payload()
    assert alternative.get_content_type() == "multipart/alternative"
    assert [p.get_content_type() for p in alternative.get_payload()] == [
        "text/plain",
        "text/html",
    ]
    assert image.get_content_type() == "image/png"
    assert image["Content-ID"] == "<dashboard>"
    assert image.get_content_disposition() == "inline"  # never a paperclip duplicate
    assert image.get_content() == png
    html = msg.get_body(("html",)).get_content()
    assert 'src="cid:dashboard"' in html
    # no dashboard.html attachment in this shape
    assert not [p for p in msg.walk() if p.get_filename() == "dashboard.html"]


def test_smtp_dashboard_email_wire_form_declares_mime_version_at_top_level_only(fake_smtp):
    # The parsed-object tests above can't see this: make_related()+attach()
    # never call set_content(), which is what auto-adds MIME-Version — the
    # WIRE form is what strict clients and spam filters judge (RFC 2045).
    notify.send_dashboard_email(_smtp_config(), "subj", "text-alt", DASH_HTML, b"png")
    (msg, _from_addr, _to_addrs) = fake_smtp.sent[-1]
    wire = msg.as_string()
    top_headers, _, body = wire.partition("\n\n")
    assert "MIME-Version: 1.0" in top_headers
    assert "MIME-Version" not in body  # never on subparts
    assert wire.count("MIME-Version") == 1


def test_resend_dashboard_email_uses_content_id_inline_attachment(resend_capture):
    png = b"\x89PNG-SENTINEL-BYTES"
    outcome = notify.send_dashboard_email(_resend_config(), "subj", "text-alt", DASH_HTML, png)
    assert outcome.transport == "resend" and outcome.attached
    payload = json.loads(resend_capture["req"].data)
    assert 'src="cid:dashboard"' in payload["html"]
    assert payload["text"] == "text-alt"
    [attachment] = payload["attachments"]
    # Resend's documented inline field is `content_id`, referenced as cid: in html
    assert attachment["content_id"] == "dashboard"
    assert attachment["filename"] == "dashboard.png"
    assert base64.b64decode(attachment["content"]) == png


def test_dashboard_email_send_failure_still_raises_safe_send_error(monkeypatch):
    def fail_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(notify.SendError, match="unreachable"):
        notify.send_dashboard_email(_resend_config(), "subj", "t", DASH_HTML, b"png")


def test_scrub_redacts_the_secret():
    assert notify.scrub("boom with s3cr3t inside", "s3cr3t") == "boom with [redacted] inside"
    assert notify.scrub("boom", "") == "boom"
