"""Exit-rule simulation for the dashboard's strategy explorer.

Deterministic what-if replay of simple DAILY exit rules over the model's picks,
as pure functions of a pick's target-day outcome returns:

    oh = (high - open) / open    ol = (low - open) / open    oc = (close - open) / open

plus `filled` — the guarded touch event (scan.touch_event_predicate: the high
reached +2% AND the bar is not a high-spike glitch). A glitch-suspect high must
NEVER count as a limit fill: a fake print can't fill your order. SIM rows reuse
the recorded `hit` flag as `filled`; LIVE rows compute it through the same
shared predicate (track.daily_rank_outcomes) — one definition everywhere.

Assumptions, stated once and surfaced on the dashboard: entries are at the
open; limit/stop fills happen at EXACTLY the trigger price; all returns are
GROSS — before trading costs (not estimated). When both the limit and the stop
could have triggered on the same day, daily data cannot tell which came first,
so the rule returns a (worst, best) BAND — a single number there would be
dishonestly precise (quant-skeptic must-fix). Trailing stops are NOT computable
from daily bars at all; the dropdown lists the rule disabled with that reason.

This module is the SERVER-SIDE half of a Python/JS lockstep pair: the inline
`tpMath` script in dashboard.py mirrors every function here and the two must
produce identical numbers (pinned by tests, including a node-executed one).
This is prediction-support tooling — NOT one of the model "strategies" in
twopercent.strategies (which rank tickers; these rules only exit positions).
"""

from __future__ import annotations

import math

from twopercent.scan import DEFAULT_THRESHOLD

# A filled +2% limit sells at exactly the threshold (the fill-at-trigger
# assumption). The SQL copy of the fill predicate lives in scan/store.
LIMIT_PROFIT = DEFAULT_THRESHOLD
# The stop rule sells at exactly −1% when the day's low reached it.
STOP_LEVEL = -0.01
# Epsilon on the stop trigger, mirroring scan._THRESHOLD_EPSILON: a true −1%
# low computed as (low - open) / open can land a few ULPs ABOVE −0.01 in double
# math (e.g. open 5.00, low 4.95 → −0.009999999999999964) and would silently
# fail to trigger without it.
_STOP_EPSILON = 1e-9

# Intraday path verdicts for a day that touched BOTH triggers. Defined here
# rather than imported from intraday.py so this module stays dependency-free
# for the JS-lockstep tests; intraday.py re-exports these as LIMIT_FIRST /
# STOP_FIRST and a test pins the two definitions together.
SEQ_LIMIT_FIRST = 1
SEQ_STOP_FIRST = 2

# (key, dropdown label, enabled). `reach` is the DEFAULT prediction-quality
# view (not an exit rule); the trailing stop is listed but disabled because
# daily bars cannot replay it — do not "approximate" it into existence.
STRATEGY_CHOICES = [
    ("reach", "Reach rate (prediction quality)", True),
    ("hold_close", "Buy open → sell close", True),
    ("limit_2pct", "Buy open → +2% limit, else close", True),
    ("limit_stop", "+2% limit + −1% stop (best/worst case)", True),
    ("trailing", "Trailing stop (needs intraday data)", False),
]
PNL_STRATEGIES = ("hold_close", "limit_2pct", "limit_stop")


def stop_triggered(ol: float) -> bool:
    """Did the day's open-to-low move reach the −1% stop (epsilon-guarded)?"""
    return ol <= STOP_LEVEL + _STOP_EPSILON


def pick_return_band(
    strategy: str,
    ol: float,
    oc: float,
    filled: bool,
    seq: int | None = None,
    fill: float | None = None,
) -> tuple[float, float]:
    """(worst, best) day return of one pick under an exit rule, buy-at-open.

    Equal endpoints mean the rule's outcome is exact; they differ only for
    `limit_stop` on a day where BOTH the limit and the stop were touched and
    the ORDER is unknown — worst assumes the stop filled first (−1%), best
    assumes the limit did (+2%). Always worst <= best. `filled` is the guarded
    touch event (see module docstring) — never a raw high comparison.

    `seq` is that day's intraday verdict (intraday.LIMIT_FIRST / STOP_FIRST,
    None when unresolved). Daily bars carry no clock, so `seq` is the ONLY way
    the band can collapse; when it is None the band is returned exactly as
    before, which keeps every pre-intraday row rendering unchanged. A verdict
    is only ever produced by intraday.resolve() after the session passed its
    agreement check, so a sparse or split-shifted record cannot collapse a band
    on the strength of bars it never saw.
    """
    if strategy == "hold_close":
        return (oc, oc)
    if strategy == "limit_2pct":
        ret = LIMIT_PROFIT if filled else oc
        return (ret, ret)
    if strategy == "limit_stop":
        stopped = stop_triggered(ol)
        # `fill` is the MEASURED stop exit (intraday.stop_fills). CLAMPED at the
        # trigger: a stop-market order fills within seconds at or below -1%, but
        # the proxy is the next 5m bar's open, up to five minutes later, so it
        # prices in any bounce that followed the air pocket. Unclamped, HALF the
        # measured fills came out better than the trigger and 17% were GAINS --
        # a stopped pick booking a profit, and the "worst-case" win rate jumping
        # +20pp from a change advertised as a conservatism fix (quant-skeptic).
        # min() keeps the change monotone in the direction it claims.
        exit_ret = STOP_LEVEL if fill is None else min(fill, STOP_LEVEL)
        if filled and stopped:
            if seq == SEQ_LIMIT_FIRST:
                return (LIMIT_PROFIT, LIMIT_PROFIT)
            if seq == SEQ_STOP_FIRST:
                return (exit_ret, exit_ret)
            # Order unknown: worst is the stop exit (at or below the trigger,
            # so always below the limit), best is the limit.
            return (exit_ret, LIMIT_PROFIT)
        if stopped:
            return (exit_ret, exit_ret)
        if filled:
            return (LIMIT_PROFIT, LIMIT_PROFIT)
        return (oc, oc)
    raise ValueError(f"unknown exit-rule strategy: {strategy!r}")


def path_note(prefix: str, s: dict | None) -> str | None:
    """Coverage disclosure for a collapsed band — mirror of JS tpMath.pathNote().

    Rendered beside the number it qualifies, for the SELECTED basket and window.
    A static server-side note would describe a different selection than the one
    on screen and would be destroyed by the first selector change.
    """
    if not s or not s.get("amb"):
        return None
    amb, res = s["amb"], s["res"]
    if not res:
        return (
            f"{prefix}: 0 of {amb} both-triggered picks resolved "
            "— every band below is the daily worst/best case"
        )
    # floor(x + 0.5), NOT round(): Python rounds half-even (1/8 -> "12%") while
    # JS Math.round is half-up ("13%"), so the server render would flicker on the
    # first selector change. Same drift dashboard._pct_half_up exists for.
    pct = math.floor(100 * res / amb + 0.5)
    return (
        f"{prefix}: {res} of {amb} both-triggered picks ({pct}%) ordered from 5m "
        "bars; the rest stay a worst/best band. A collapsed band is conditional "
        "on that replay, not proven from daily bars, and resolvable days skew liquid."
    )


def summarize_strategy_days(days: list[dict], n: int, strategy: str) -> dict:
    """Compounded gross growth + pick win rates of an exit rule over payload days.

    EXACT server-side mirror of the inline JS tpMath.stratSummarize() — same
    inputs (the embedded-payload day dicts, picks as [rank, oh, ol, oc, hit]),
    same iteration order, so Python and JS produce bit-identical numbers.
    Basket = first n picks of each day (rank order; taking the next available
    name IS the substitution rule, counted and disclosed like the reach view).
    Day return = equal-weight mean of the basket's per-pick returns; growth
    compounds (1 + day return) across days — GROSS, no cost model. Win rate =
    share of individual PICKS with return > 0; the dashboard must never show it
    without the growth number beside it. gw/ww are the worst-case band ends,
    gb/wb the best (equal unless `limit_stop` hit both triggers on some day).

    A day with any null oh/ol/oc (a pre-upgrade benchmark row) is counted in
    `missing` and the caller must show "no strategy data recorded — re-run
    twopercent benchmark" instead of a number computed from a silently
    different window. Non-finite inputs (impossible via JSON, guarded anyway)
    count as `corrupt` — same never-average-around rule as the reach view.
    `base_days` is fixed at 0 (no base-rate concept in a P&L view) so
    dashboard._row_notes can be shared unchanged.
    """
    growth_w = growth_b = 1.0
    wins_w = wins_b = n_picks = 0
    amb_n = res_n = 0
    short = subst = missing = corrupt = 0
    for day in days:
        picks = day["picks"][:n]
        if len(picks) < n:
            short += 1
        if any(p[0] != i + 1 for i, p in enumerate(picks)):
            subst += 1
        if any(p[1] is None or p[2] is None or p[3] is None for p in picks):
            missing += 1
            continue
        if not picks:
            corrupt += 1
            continue
        sum_w = sum_b = 0.0
        # Day-local win tallies: committed only after the whole day validates,
        # so a partially-corrupt day can never leak ghost wins into the rate
        # (numerator without its denominator — reviewer finding, PR #77).
        day_wins_w = day_wins_b = 0
        day_amb = day_res = 0
        ok = True
        for p in picks:
            # p[5] is the intraday verdict; absent on payloads written before
            # #79, which must keep rendering as the unresolved band rather than
            # raising or silently reading as "resolved".
            seq = p[5] if len(p) > 5 else None
            fill = p[6] if len(p) > 6 else None
            # Ambiguity/resolution tallies for the disclosure: a band is a point
            # here ONLY because the intraday replay ordered the two triggers, so
            # the page must be able to say how much of the selection that covers.
            if bool(p[4]) and stop_triggered(p[2]):
                day_amb += 1
                if seq in (SEQ_LIMIT_FIRST, SEQ_STOP_FIRST):
                    day_res += 1
            worst, best = pick_return_band(strategy, p[2], p[3], bool(p[4]), seq, fill)
            if not (math.isfinite(worst) and math.isfinite(best)):
                ok = False
                break
            sum_w += worst
            sum_b += best
            if worst > 0:
                day_wins_w += 1
            if best > 0:
                day_wins_b += 1
        if not ok:
            corrupt += 1
            continue
        wins_w += day_wins_w
        wins_b += day_wins_b
        amb_n += day_amb
        res_n += day_res
        n_picks += len(picks)
        growth_w *= 1 + sum_w / len(picks)
        growth_b *= 1 + sum_b / len(picks)
    clean = len(days) - missing - corrupt
    return {
        "gw": growth_w if clean else float("nan"),
        "gb": growth_b if clean else float("nan"),
        "ww": wins_w / n_picks if n_picks else float("nan"),
        "wb": wins_b / n_picks if n_picks else float("nan"),
        "amb": amb_n,
        "res": res_n,
        "picks": n_picks,
        "days": len(days),
        "clean": clean,
        "short": short,
        "subst": subst,
        "base_days": 0,
        "missing": missing,
        "corrupt": corrupt,
    }
