"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Reads one make.fun case (schema makefun/gta6-trophies-case/v1): the history
of Exophase reads of every Grand Theft Auto VI PS5 trophy before the case's
checkpoint, and writes five quantiles (5/25/50/75/95%) of each trophy's earn
rate at the case's target, as /output/<id>.out.

Method, per trophy (standard library only, deterministic, no network):
  * Work on the logit scale, logit(p) = ln(p / (1 - p)), p in percent.
    Trophy growth is modelled against log-time since release,
    x = ln(1 + hours since 2026-11-19T00:00Z): early playthrough progress is
    fast and then flattens in percent terms, which a line in (x, logit) space
    captures, and it saturates naturally towards 100%.
  * A second candidate, a line in (hours, logit) space, is fitted on the same
    recent window (last 48 h of reads, or all reads when shorter). A backtest
    on the last 12 h of history picks the candidate whose median predicted
    the held-out reads better; ties go to the log-time model. Flat histories
    (no movement yet) fall back to persistence.
  * Slopes are clipped so no trophy can run away, medians are clamped to
    [0.01, 99.9], and the median's move from the last read is capped at
    2.5 logit units over the horizon. The spread around the median comes from the fit's
    residual scatter plus a term that grows with the horizon and with the
    backtest error -- narrow where the curve has been steady, wide where the
    record is jumpy (Exophase refreshes arrive in steps) or the trophy has
    never been earned yet.
  * Story order is enforced: the Chapter 1..4 + final-mission trophies cannot
    have a rising median along the story, and the platinum cannot sit above
    the lowest other trophy's median (it needs every other trophy first).

Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, one case per run, <= 8 s, 1 CPU, 1024 MB.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry ships one file only.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Release instant the log-time clock runs from (2026-11-19T00:00:00Z).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Recent window the trend is fitted on; older reads are pre-refresh history.
WINDOW_HOURS = 48.0
# Backtest horizon: fit before this cut, score the median on reads after it.
BACKTEST_HOURS = 12.0
# Slope clips: logit per log-time unit, and logit per hour.
B_LOGTIME_MIN, B_LOGTIME_MAX = -1.0, 4.0
B_TIME_MIN, B_TIME_MAX = -0.05, 0.20
# Spread (logit) floor / cap, and the horizon term's scale.
SIGMA_MIN, SIGMA_MAX = 0.25, 1.50
SIGMA_HORIZON = 0.35
# A trophy nobody has earned yet borrows this log-time slope from its peers.
ZERO_RATE_SLOPE = 0.30

CHAPTER_RE = re.compile(r"Complete Chapter (\d)\b")
FINAL_RE = re.compile(r"Complete the final mission\b")


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, like JavaScript Math.round(x*1e4)/1e4."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def wls(points, xs_key):
    """Weighted least squares y ~ a + b * xs_key(point); returns (a, b)."""
    sw = sx = sy = sxx = sxy = 0.0
    for t, x, y, w in points:
        xv = xs_key(t, x)
        sw += w
        sx += w * xv
        sy += w * y
        sxx += w * xv * xv
        sxy += w * xv * y
    denom = sw * sxx - sx * sx
    if sw <= 0 or abs(denom) < 1e-12:
        mean_y = sy / sw if sw > 0 else 0.0
        return mean_y, 0.0
    b = (sw * sxy - sx * sy) / denom
    a = (sy - b * sx) / sw
    return a, b


def robust_sd(values):
    """Median absolute deviation scaled to a standard deviation."""
    if not values:
        return 0.0
    s = sorted(values)
    med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    devs = sorted(abs(v - med) for v in values)
    mad = devs[len(devs) // 2] if len(devs) % 2 else 0.5 * (devs[len(devs) // 2 - 1] + devs[len(devs) // 2])
    return 1.4826 * mad


def fit_trophy(series, t_target_h, horizon_days):
    """series: [(t_hours, rate_pct)] oldest first, non-null only.

    Returns (median_pct, sigma_logit)."""
    last_rate = series[-1][1]
    t_last = series[-1][0]
    # Recent window; keep every read in it (Exophase refreshes are steps,
    # so consecutive equal rates are real information, not duplicates).
    window = [(t, r) for (t, r) in series if t >= t_last - WINDOW_HOURS]
    if len(window) < 3:
        window = list(series)
    pts = []
    t0 = window[0][0]
    span = max(t_last - t0, 1e-9)
    for (t, r) in window:
        x = math.log1p(max(t, 0.0))
        y = logit(r)
        w = 1.0 + 2.0 * (t - t0) / span  # recent reads count up to 3x
        pts.append((t, x, y, w))

    def predict_logtime(t):
        a, b = wls(pts, lambda tt, xx: xx)
        b = min(B_LOGTIME_MAX, max(B_LOGTIME_MIN, b))
        return a + b * math.log1p(max(t, 0.0)), b

    def predict_time(t):
        a, b = wls(pts, lambda tt, xx: tt)
        b = min(B_TIME_MAX, max(B_TIME_MIN, b))
        return a + b * t, b

    # Backtest: fit on reads before the cut, score the median on reads after.
    cut = t_last - BACKTEST_HOURS
    before = [p for p in pts if p[0] <= cut]
    after = [p for p in pts if p[0] > cut]
    use_logtime = True
    backtest_err = 0.0
    if len(before) >= 3 and after:
        a1, b1 = wls(before, lambda tt, xx: xx)
        b1 = min(B_LOGTIME_MAX, max(B_LOGTIME_MIN, b1))
        a2, b2 = wls(before, lambda tt, xx: tt)
        b2 = min(B_TIME_MAX, max(B_TIME_MIN, b2))
        e1 = e2 = 0.0
        for (t, x, y, _w) in after:
            e1 += abs((a1 + b1 * x) - y)
            e2 += abs((a2 + b2 * t) - y)
        e1 /= len(after)
        e2 /= len(after)
        backtest_err = min(e1, e2)
        # Prefer log-time unless the time-linear model is clearly better.
        use_logtime = not (e2 < 0.95 * e1)
    elif len(set(round(p[2], 6) for p in pts)) <= 1:
        # No movement at all in the window: persistence, no trend to chase.
        return clamp_pct(last_rate), SIGMA_MIN + SIGMA_HORIZON * math.sqrt(horizon_days)

    if use_logtime:
        med_logit, _b = predict_logtime(t_target_h)
        a, b = wls(pts, lambda tt, xx: xx)
        b = min(B_LOGTIME_MAX, max(B_LOGTIME_MIN, b))
        resid = [(a + b * x) - y for (_t, x, y, _w) in pts]
    else:
        med_logit, _b = predict_time(t_target_h)
        a, b = wls(pts, lambda tt, xx: tt)
        b = min(B_TIME_MAX, max(B_TIME_MIN, b))
        resid = [(a + b * t) - y for (t, _x, y, _w) in pts]

    resid_sd = robust_sd(resid)
    # Extrapolation leverage: how far past the window the target sits,
    # in units of the window's own log-time span.
    xs = [p[1] for p in pts]
    x_span = max(max(xs) - min(xs), 1e-9)
    x_t = math.log1p(max(t_target_h, 0.0))
    leverage = (x_t - max(xs)) / x_span if x_t > max(xs) else 0.0
    sigma = math.sqrt(
        (resid_sd ** 2) * (1.0 + leverage ** 2)
        + (SIGMA_HORIZON * math.sqrt(horizon_days)) ** 2
        + (0.5 * backtest_err) ** 2
    )
    sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
    # Robustness guard: however steep the fitted line, the median may not
    # move more than 2.5 logit units from the last read over the horizon.
    # (Never binds on smooth synthetic weeks; stops a noisy short window
    # from exploding a rare trophy's median.)
    med_logit = min(logit(last_rate) + 2.5,
                    max(logit(last_rate) - 2.5, med_logit))
    median = min(99.9, max(0.01, inv_logit(med_logit)))
    return median, sigma


def story_order(trophies):
    """Chapter trophies in story order: chapter 1..4 then final mission."""
    order = []
    for i, tr in enumerate(trophies):
        m = CHAPTER_RE.search(tr.get("description", ""))
        if m:
            order.append((int(m.group(1)), i))
        elif FINAL_RE.search(tr.get("description", "")):
            order.append((5, i))
    return [i for (_n, i) in sorted(order)]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    target = parse_time(case["target"])
    t_target_h = (target - RELEASE).total_seconds() / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0

    series_by_trophy = []
    for i in range(len(trophies)):
        series = []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            t_h = (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0
            series.append((t_h, float(rate)))
        series_by_trophy.append(series)

    medians = []
    sigmas = []
    for i, series in enumerate(series_by_trophy):
        if not series:
            # Trophy never appeared in any read: midpoint with max spread.
            medians.append(50.0)
            sigmas.append(SIGMA_MAX)
            continue
        last_rate = series[-1][1]
        if all(r <= 0.01 for (_t, r) in series):
            # Never earned by anyone yet (platinum, 100% completion, ...):
            # borrow a modest peer slope instead of a flat zero.
            t_last = series[-1][0]
            med_logit = logit(0.01) + ZERO_RATE_SLOPE * (
                math.log1p(max(t_target_h, 0.0)) - math.log1p(max(t_last, 0.0))
            )
            medians.append(min(99.9, max(0.01, inv_logit(med_logit))))
            sigmas.append(SIGMA_MAX)
            continue
        median, sigma = fit_trophy(series, t_target_h, horizon_days)
        if last_rate >= 99.99 and median < 99.0:
            # Everybody already holds it: it stays (nearly) universal.
            median, sigma = 99.9, SIGMA_MIN
        medians.append(median)
        sigmas.append(sigma)

    # Platinum can never sit above the lowest other trophy's median: it
    # requires every other trophy first.
    for i, tr in enumerate(trophies):
        if tr.get("tier") == "platinum" and len(medians) > 1:
            cap = min(m for j, m in enumerate(medians) if j != i)
            if medians[i] > cap:
                medians[i] = cap
    # Story trophies cannot rise along the story.
    order = story_order(trophies)
    running = None
    for i in order:
        if running is None or medians[i] < running:
            running = medians[i]
        else:
            medians[i] = running

    rows = []
    for i, tr in enumerate(trophies):
        m_logit = logit(medians[i])
        sigma = sigmas[i]
        qs = []
        prev = None
        for z in Z:
            q = round4(min(99.99, max(0.01, inv_logit(m_logit + z * sigma))))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
