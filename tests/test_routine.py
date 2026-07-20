import base64
import datetime as dt
import io
import json
import smtplib
import urllib.error

import pandas as pd
import pytest

from tests.conftest import seed_history, seed_planted
from twopercent import ingest, notify, routine, store

CLOSED = dt.datetime(2026, 7, 17, 7, 0, tzinfo=routine._EASTERN)  # Friday pre-open


def _db(con) -> str:
    return con.execute("PRAGMA database_list").fetchone()[2]


@pytest.fixture
def ready(con, monkeypatch, tmp_path):
    """A healthy seeded store, clock pre-open, network steps stubbed out.

    Email is UNCONFIGURED by default (env cleared, .env pointed at a missing
    file) and SMTP is rigged to fail the test if anything ever dials out —
    individual tests opt in with their own env + fake SMTP."""
    seed_planted(con, n_each=10)
    con.execute("UPDATE universe SET as_of = ?", [CLOSED.date()])
    # Newest bar lands 3 days before the pinned clock — deterministic forever.
    span = (CLOSED.date() - pd.Timestamp("2026-01-05").date()).days
    con.execute(f"UPDATE prices SET date = date + INTERVAL {span - 137} DAY")
    monkeypatch.setattr(routine, "_now_eastern", lambda: CLOSED)
    monkeypatch.setattr(
        routine.ingest,
        "ingest",
        lambda con_, symbols, **kw: ingest.IngestResult(symbols_skipped=list(symbols)),
    )
    monkeypatch.setattr(
        routine.universe, "refresh_universe", lambda **kw: pytest.fail("network touched")
    )
    monkeypatch.setattr(notify, "DEFAULT_ENV_PATH", tmp_path / "absent.env")
    for var in (
        notify.ENV_TO,
        notify.ENV_FROM,
        notify.ENV_RESEND_KEY,
        notify.ENV_SMTP_HOST,
        notify.ENV_SMTP_PORT,
        notify.ENV_SMTP_USER,
        notify.ENV_SMTP_PASSWORD,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        notify.smtplib, "SMTP", lambda *a, **kw: pytest.fail("SMTP touched without a fake")
    )
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **kw: pytest.fail("HTTP touched without a fake"),
    )
    return con


def test_routine_happy_path_completes_all_steps(ready):
    report = routine.run(db_path=_db(ready))
    names = [s.name for s in report.steps]
    assert names == [
        "clock",
        "doctor",
        "universe",
        "ingest",
        "freshness",
        "recheck",
        "predict",
        "dashboard",
        "scoring",
        "notify",
    ]
    assert report.status in ("ok", "warn")
    assert report.top_candidates
    assert all(sym.startswith("RUN") for sym, _ in report.top_candidates)


def test_routine_refuses_to_run_while_market_open(ready, monkeypatch):
    midday = dt.datetime(2026, 7, 17, 11, 0, tzinfo=routine._EASTERN)  # Friday 11:00 ET
    monkeypatch.setattr(routine, "_now_eastern", lambda: midday)
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert [s.name for s in report.steps] == ["clock"]
    assert "partial" in report.steps[0].detail
    n = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert n == 0  # nothing ran


def test_market_hours_boundaries():
    friday = dt.date(2026, 7, 17)
    saturday = dt.date(2026, 7, 18)

    def mk(d, h, m):
        return dt.datetime(d.year, d.month, d.day, h, m, tzinfo=routine._EASTERN)

    assert routine._market_is_open(mk(friday, 9, 25))
    assert routine._market_is_open(mk(friday, 16, 14))
    assert not routine._market_is_open(mk(friday, 9, 24))
    assert not routine._market_is_open(mk(friday, 16, 15))
    assert not routine._market_is_open(mk(saturday, 11, 0))


def test_routine_aborts_on_ingest_introduced_corruption(ready, monkeypatch):
    def corrupting_ingest(con_, symbols, **kw):
        # Ingest "succeeds" but writes a fresh extreme bar RECENT enough to be
        # today's data (within 3 days of the store's newest bar).
        newest = con_.execute("SELECT max(date) FROM prices").fetchone()[0]
        start = pd.bdate_range(end=newest, periods=1)[0].date()
        seed_history(con_, {"EVIL": [0.9]}, start=str(start))
        return ingest.IngestResult(symbols_skipped=list(symbols))

    monkeypatch.setattr(routine.ingest, "ingest", corrupting_ingest)
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert report.steps[-1].name == "recheck"
    assert "EVIL" in report.steps[-1].detail
    n = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert n == 0  # the model never trained on the corrupted store


def test_routine_backfilled_historical_extremes_warn_not_fail(ready, monkeypatch):
    # A weekly refresh backfilling a volatile symbol adds YEARS-OLD extreme
    # bars. Hard-failing on those would cry wolf weekly and teach the operator
    # to ignore exit 2 — they must WARN while recent corruption still FAILs.
    def backfilling_ingest(con_, symbols, **kw):
        seed_history(con_, {"BIOTC": [0.0, 0.9, 0.0]}, start="2024-03-04")  # old squeeze
        return ingest.IngestResult(symbols_skipped=list(symbols))

    monkeypatch.setattr(routine.ingest, "ingest", backfilling_ingest)
    report = routine.run(db_path=_db(ready))
    recheck = next(s for s in report.steps if s.name == "recheck")
    assert recheck.status == "warn"
    assert "backfill" in recheck.detail
    assert any(s.name == "predict" for s in report.steps)  # run continued


def test_routine_preexisting_problems_warn_but_run(ready):
    # A pre-existing extreme bar (present in the pre-ingest baseline) must not
    # abort the run — it warns via the doctor step and the model proceeds.
    seed_history(ready, {"OLDX": [0.0] * 3 + [0.9]}, start="2026-06-01")
    report = routine.run(db_path=_db(ready))
    assert report.status == "warn"
    assert any(s.name == "predict" for s in report.steps)  # ran to completion
    doctor_step = next(s for s in report.steps if s.name == "doctor")
    assert doctor_step.status == "warn"


def test_routine_staleness_gate(ready):
    # Age the store by dropping everything newer than 60 days before the pinned clock.
    ready.execute("DELETE FROM prices WHERE date > ?", [CLOSED.date() - dt.timedelta(days=60)])
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert report.steps[-1].name == "freshness"
    assert "track record" in report.steps[-1].detail


def test_routine_ingest_failure_rate_boundary(ready, monkeypatch):
    # 20 symbols in the seeded universe: 1/20 = exactly 5% must pass (strict >),
    # 2/20 must abort.
    def fail_n(n):
        def fake(con_, symbols, **kw):
            symbols = list(symbols)
            return ingest.IngestResult(symbols_failed=symbols[:n], symbols_skipped=symbols[n:])

        return fake

    monkeypatch.setattr(routine.ingest, "ingest", fail_n(1))
    assert routine.run(db_path=_db(ready)).status in ("ok", "warn")

    monkeypatch.setattr(routine.ingest, "ingest", fail_n(2))
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert report.steps[-1].name == "ingest"


def test_routine_unaccounted_symbols_fail(ready, monkeypatch):
    def lossy_ingest(con_, symbols, **kw):
        return ingest.IngestResult(symbols_skipped=list(symbols)[:-3])  # drops 3

    monkeypatch.setattr(routine.ingest, "ingest", lossy_ingest)
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert "unaccounted" in report.steps[-1].detail


def test_routine_universe_refresh_failure_degrades_not_aborts(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [CLOSED.date() - dt.timedelta(days=30)])

    def broken_refresh(**kw):
        raise RuntimeError("screener down")

    monkeypatch.setattr(routine.universe, "refresh_universe", broken_refresh)
    report = routine.run(db_path=_db(ready))
    uni_step = next(s for s in report.steps if s.name == "universe")
    assert uni_step.status == "warn"
    assert "screener down" in uni_step.detail
    assert any(s.name == "predict" for s in report.steps)  # run continued
    assert report.exit_code == 1  # degraded, not failed


def test_routine_stale_universe_triggers_refresh(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [CLOSED.date() - dt.timedelta(days=30)])
    called = {}

    def fake_refresh(**kw):
        called["yes"] = True
        return pd.DataFrame(
            {"symbol": ["RUN00"], "name": ["RUN00"], "market_cap": [1e9], "sector": ["Tech"]}
        )

    monkeypatch.setattr(routine.universe, "refresh_universe", fake_refresh)
    routine.run(db_path=_db(ready))
    assert called.get("yes")
    assert store.latest_universe(ready)["as_of"].iloc[0].date() == CLOSED.date()


def test_predictions_record_universe_snapshot(ready):
    routine.run(db_path=_db(ready))
    as_of = ready.execute("SELECT DISTINCT universe_as_of FROM predictions").fetchall()
    assert as_of == [(CLOSED.date(),)]


def test_predict_mode_explicit_runs_predict_and_no_detector(ready):
    # Level 4 regression: --mode predict (and bare routine, above) is the
    # pre-open cycle — champion predict runs, score-mode steps never appear.
    report = routine.run(db_path=_db(ready), mode="predict")
    names = [s.name for s in report.steps]
    assert "predict" in names
    assert not {"score", "detector", "issue"} & set(names)


def test_summary_lines_shape(ready):
    report = routine.run(db_path=_db(ready))
    lines = report.summary_lines()
    assert lines[0].startswith("routine:")
    assert any("top candidates" in line for line in lines)


# --- notify step wiring -------------------------------------------------------


class _FakeSMTP:
    """Records the call sequence; never touches the network."""

    sent: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append("login")

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.calls.append("send_message")
        type(self).sent.append((msg, from_addr, to_addrs))


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _configure_common(monkeypatch):
    monkeypatch.setenv(notify.ENV_TO, "leon@example.com")
    monkeypatch.setenv(notify.ENV_FROM, "signals@example.com")


def _configure_resend(monkeypatch, key="re_test_key_123"):
    _configure_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_RESEND_KEY, key)


def _configure_smtp(monkeypatch, password="smtp-pw-123"):
    _configure_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_SMTP_HOST, "smtp.example.com")
    monkeypatch.setenv(notify.ENV_SMTP_USER, "sender@example.com")
    monkeypatch.setenv(notify.ENV_SMTP_PASSWORD, password)


def _capture_resend(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _FakeResponse()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_notify_unconfigured_skips_without_degrading_exit(ready):
    report = routine.run(db_path=_db(ready))
    step = report.steps[-1]
    assert step.name == "notify"
    assert step.status == "ok"  # deliberate non-setup is not an exception
    assert "not configured" in step.detail and "skipping" in step.detail
    assert notify.ENV_TO in step.detail  # names the missing variables
    assert notify.ENV_RESEND_KEY in step.detail


def test_notify_sends_via_resend_with_dashboard_attached(ready, monkeypatch, tmp_path):
    _configure_resend(monkeypatch)
    captured = _capture_resend(monkeypatch)
    out = tmp_path / "dashboard.html"
    report = routine.run(db_path=_db(ready), out_path=str(out))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "ok"
    assert "via resend" in step.detail
    assert "with dashboard attached" in step.detail
    payload = json.loads(captured["req"].data)
    assert payload["to"] == ["leon@example.com"]
    assert payload["attachments"][0]["filename"] == "dashboard.html"
    assert base64.b64decode(payload["attachments"][0]["content"]) == out.read_bytes()


def test_notify_missing_dashboard_warns_but_sends(ready, monkeypatch, tmp_path):
    _configure_resend(monkeypatch)
    captured = _capture_resend(monkeypatch)

    def broken_render(*a, **kw):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(routine.dashboard, "render", broken_render)
    report = routine.run(db_path=_db(ready), out_path=str(tmp_path / "dashboard.html"))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"
    assert "WITHOUT dashboard attachment" in step.detail
    payload = json.loads(captured["req"].data)  # the email still went out
    assert "attachments" not in payload


def test_notify_resend_rejection_warns_with_status_never_the_key(ready, monkeypatch, tmp_path):
    sentinel = "re_sentinel_key_XYZZY_99"
    _configure_resend(monkeypatch, key=sentinel)

    def fail_urlopen(req, timeout=None):
        body = io.BytesIO(json.dumps({"message": "domain is not verified"}).encode())
        raise urllib.error.HTTPError(notify.RESEND_URL, 403, "Forbidden", {}, body)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail_urlopen)
    report = routine.run(db_path=_db(ready), out_path=str(tmp_path / "dashboard.html"))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"
    assert "403" in step.detail and "domain is not verified" in step.detail
    assert sentinel not in "\n".join(report.summary_lines())
    assert report.exit_code == 1  # WARN class, never 2


def test_notify_both_transports_warns_smtp_ignored(ready, monkeypatch, tmp_path):
    _configure_resend(monkeypatch)
    _configure_smtp(monkeypatch)  # SMTP set too — Resend must win, loudly
    captured = _capture_resend(monkeypatch)
    report = routine.run(db_path=_db(ready), out_path=str(tmp_path / "dashboard.html"))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"
    assert "SMTP settings ignored" in step.detail
    assert "via resend" in step.detail
    assert "req" in captured  # sent through Resend, not SMTP (SMTP fake would fail the test)


def test_notify_smtp_auth_failure_warns_and_never_leaks_password(ready, monkeypatch, tmp_path):
    sentinel = "sekrit-sentinel-XYZZY-99"
    _configure_smtp(monkeypatch, password=sentinel)

    class AuthFailSMTP(_FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")

    monkeypatch.setattr(notify.smtplib, "SMTP", AuthFailSMTP)
    report = routine.run(db_path=_db(ready), out_path=str(tmp_path / "dashboard.html"))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"
    assert "authentication failed" in step.detail
    assert "check the SMTP password" in step.detail
    output = "\n".join(report.summary_lines())
    assert sentinel not in output
    assert "535" not in step.detail  # fixed message, not the exception repr
    assert report.exit_code == 1  # WARN class, never 2


def test_notify_generic_send_failure_scrubs_secrets(ready, monkeypatch, tmp_path):
    sentinel = "sekrit-sentinel-ABCD-42"
    _configure_smtp(monkeypatch, password=sentinel)

    class ExplodingSMTP(_FakeSMTP):
        def send_message(self, msg, from_addr=None, to_addrs=None):
            raise RuntimeError(f"transport wedged while sending with {sentinel}")

    monkeypatch.setattr(notify.smtplib, "SMTP", ExplodingSMTP)
    report = routine.run(db_path=_db(ready), out_path=str(tmp_path / "dashboard.html"))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"
    assert sentinel not in "\n".join(report.summary_lines())
    assert "[redacted]" in step.detail
    assert report.exit_code == 1


def test_notify_misconfigured_recipient_warns(ready, monkeypatch):
    _configure_resend(monkeypatch)
    monkeypatch.setenv(notify.ENV_TO, "not-an-address")
    report = routine.run(db_path=_db(ready))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"
    assert "misconfigured" in step.detail
    assert report.exit_code == 1


def test_notify_partial_smtp_config_warns_not_skips(ready, monkeypatch):
    _configure_common(monkeypatch)
    monkeypatch.setenv(notify.ENV_SMTP_HOST, "smtp.example.com")  # no user/password
    report = routine.run(db_path=_db(ready))
    step = next(s for s in report.steps if s.name == "notify")
    assert step.status == "warn"  # half-configured must never look like deliberate non-setup
    assert "partial SMTP configuration" in step.detail
