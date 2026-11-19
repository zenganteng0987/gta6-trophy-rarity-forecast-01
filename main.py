"""Forecast GTA VI trophy earn rates on Exophase at a target instant.

Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (default /input) and writes
one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (default /output).

Model, in brief
---------------
The history is first collapsed into distinct page states: Exophase refreshes
a game's counts on its own schedule, so consecutive reads with the same
players-tracked count and the same rates are one frozen page, not new data.

* Rich history (at least 4 distinct states, spanning at least 6 hours, and
  players tracked up by more than 5%): players tracked N is extrapolated to
  the target by fitting ln N against ln(1 + hours since release).  Each
  trophy then gets two rate estimates -- an earners-elasticity fit of
  ln(earners + 0.5) against ln N, and a trend fit of logit(rate) against
  ln(1 + hours) -- with both slopes shrunk toward their rarity-tier mean.
  The median is the average of the two rates on the logit scale; the spread
  combines the trend fit's residual, a random-walk term over the horizon,
  and small-earner noise.
* Frozen history (the practice shape: the page never changed before the
  checkpoint): a refresh is overdue, so rates are expected to fall.  New
  tracked players hold each trophy at a lower rate than the old cohort, so
  every rate shifts down on the logit scale by the gap between earners
  growth and players growth over the page age plus the horizon, with a
  wide spread.  A trophy already at 100% stays put.

Nothing here touches the network and nothing is random, so two runs over
the same input produce byte-identical output.  Any failure on a case falls
back to persistence (every quantile equal to the trophy's last rate), so
the program never crashes without writing a valid forecast.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Winning measured parameters: SHRINK_K=8, RW=0.2, STALE_SIGMA=0.9.
PARAMS = {
    "RICH_MIN_STATES": 4,
    "RICH_MIN_SPAN_H": 6.0,
    "SHRINK_K": 8.0,
    "BETA_LO": -0.25,
    "BETA_HI": 1.25,
    "DELTA_LO": -6.0,
    "DELTA_HI": 6.0,
    "ENS_W": 0.5,
    "RW": 0.2,
    "SPREAD_SCALE": 1.0,
    "STALE_LN_N_RATE": 0.383,
    "STALE_N_MULT_CAP": 30.0,
    "STALE_G": 0.089,
    "STALE_SHIFT_CAP": 2.2,
    "STALE_SIGMA": 0.9,
}


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _clamp_pct(value):
    return min(99.99, max(0.01, value))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _round4(value):
    stepped = math.floor(value * 10000.0 + 0.5) / 10000.0
    return int(stepped) if float(stepped).is_integer() else stepped


def _fit_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x.

    Returns (intercept, slope, residual_sd) or None when there are too
    few points or no weight.  A flat x-range gives slope 0.
    """
    total_w = sum(ws)
    if total_w <= 0 or len(xs) < 2:
        return None
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (mean_y, 0.0, 0.0)
    slope = sum(w * (x - mean_x) * (y - mean_y)
                for x, y, w in zip(xs, ys, ws)) / sxx
    intercept = mean_y - slope * mean_x
    if len(xs) >= 3:
        rss = sum(w * (y - (intercept + slope * x)) ** 2
                  for x, y, w in zip(xs, ys, ws))
        resid_sd = math.sqrt(rss / max(total_w - 2.0, 1e-9)
                             * (len(xs) / max(len(xs) - 2, 1)))
    else:
        resid_sd = 0.0
    return (intercept, slope, resid_sd)


def _page_states(case, epoch):
    """Collapse consecutive identical reads into distinct page states.

    Each state records when the page first showed those numbers, the
    players tracked, and the rates.  A state's weight grows with how
    long the page sat unchanged: w = 1 + ln(1 + hours shown).
    """
    checkpoint_h = (_parse_time(case["at"]) - epoch).total_seconds() / 3600.0
    states = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        shown_h = (_parse_time(read["slot"]) - epoch).total_seconds() / 3600.0
        if states and states[-1]["key"] == key:
            continue
        states.append({"key": key, "start_h": shown_h,
                       "players": read["players_tracked"],
                       "rates": read["rates"]})
    for j, state in enumerate(states):
        end_h = states[j + 1]["start_h"] if j + 1 < len(states) else checkpoint_h
        shown_for = max(end_h - state["start_h"], 0.0)
        state["weight"] = 1.0 + math.log(1.0 + shown_for)
    return states, checkpoint_h


def _last_seen(states, idx):
    """Most recent non-null rate for trophy idx and the N it came with."""
    for state in reversed(states):
        rate = state["rates"][idx]
        if rate is not None:
            return rate, state["players"]
    return None, states[-1]["players"]


def _rich_context(case, states, params, checkpoint_h):
    """Shared quantities for the rich-history branch."""
    horizon_h = case["horizon_minutes"] / 60.0
    first, last = states[0], states[-1]
    span_h = last["start_h"] - first["start_h"]
    ctx = {
        "horizon_h": horizon_h,
        "horizon_days": horizon_h / 24.0,
        "span_h": span_h,
        "target_h": checkpoint_h + horizon_h,
        "target_n": None,
        "pooled_resid": 0.45,
        "beta": {},
        "delta": {},
        "rich": False,
    }
    rich = (len(states) >= params["RICH_MIN_STATES"]
            and span_h >= params["RICH_MIN_SPAN_H"]
            and last["players"] > first["players"] * 1.05)
    if not rich:
        return ctx
    ctx["rich"] = True

    # Players tracked at the target: ln N on ln(1 + hours), slope in [0, 8].
    traj = _fit_line(
        [math.log(1.0 + max(s["start_h"], 0.0)) for s in states],
        [math.log(max(s["players"], 1)) for s in states],
        [s["weight"] for s in states])
    if traj is not None:
        slope = min(8.0, max(0.0, traj[1]))
        n_target = math.exp(traj[0] + slope
                            * math.log(1.0 + max(ctx["target_h"], 0.0)))
        ctx["target_n"] = min(max(n_target, last["players"]),
                              last["players"] * 300.0)

    # First pass: raw per-trophy slopes, tier means, pooled residual.
    raw_beta, raw_delta, resids = {}, {}, []
    by_tier = {}
    for idx, trophy in enumerate(case["trophies"]):
        pts = [(s["start_h"], s["players"], s["rates"][idx]) for s in states
               if s["rates"][idx] is not None]
        if len(pts) < 3:
            continue
        ws = [s["weight"] for s in states if s["rates"][idx] is not None]
        el = _fit_line([math.log(max(n, 1)) for _, n, _ in pts],
                       [math.log(r * n / 100.0 + 0.5) for _, n, r in pts], ws)
        tr = _fit_line([math.log(1.0 + max(h, 0.0)) for h, _, _ in pts],
                       [_logit(max(r, 50.0 / max(n, 1))) for _, n, r in pts],
                       ws)
        if el is not None:
            raw_beta[idx] = el[1]
        if tr is not None:
            raw_delta[idx] = tr[1]
            if len(pts) >= 5 and tr[2] > 0:
                resids.append(tr[2])
        by_tier.setdefault(trophy["tier"], []).append(idx)
    if resids:
        ordered = sorted(resids)
        ctx["pooled_resid"] = max(0.15, ordered[len(ordered) // 2])

    def _tier_mean(values, fallback):
        return sum(values) / len(values) if values else fallback

    all_beta = list(raw_beta.values())
    all_delta = list(raw_delta.values())
    global_beta = _tier_mean(all_beta, 0.7)
    global_delta = _tier_mean(all_delta, 0.0)
    shrink = params["SHRINK_K"]
    for tier, members in by_tier.items():
        tier_beta = _tier_mean([raw_beta[i] for i in members if i in raw_beta],
                               global_beta)
        tier_delta = _tier_mean([raw_delta[i] for i in members if i in raw_delta],
                                global_delta)
        for idx in members:
            npts = sum(1 for s in states if s["rates"][idx] is not None)
            m = max(npts - 2, 1)
            if idx in raw_beta:
                shrunk = (raw_beta[idx] * m + shrink * tier_beta) / (m + shrink)
                ctx["beta"][(tier, idx)] = min(params["BETA_HI"],
                                               max(params["BETA_LO"], shrunk))
            if idx in raw_delta:
                shrunk = (raw_delta[idx] * m + shrink * tier_delta) / (m + shrink)
                ctx["delta"][(tier, idx)] = min(params["DELTA_HI"],
                                                max(params["DELTA_LO"], shrunk))
    return ctx


def _rich_row(case, states, idx, tier, params, ctx):
    """Median + sigma for one trophy from the rich-history fits, or None."""
    pts = [(s["start_h"], s["players"], s["rates"][idx]) for s in states
           if s["rates"][idx] is not None]
    if len(pts) < 3:
        return None
    ws = [s["weight"] for s in states if s["rates"][idx] is not None]
    el = _fit_line([math.log(max(n, 1)) for _, n, _ in pts],
                   [math.log(r * n / 100.0 + 0.5) for _, n, r in pts], ws)
    tr = _fit_line([math.log(1.0 + max(h, 0.0)) for h, _, _ in pts],
                   [_logit(max(r, 50.0 / max(n, 1))) for _, n, r in pts], ws)
    if el is None and tr is None:
        return None
    last_rate, ref_n = _last_seen(states, idx)
    earners_last = last_rate * ref_n / 100.0
    logits, weights = [], []
    if el is not None and ctx["target_n"] is not None:
        beta = ctx["beta"].get((tier, idx))
        if beta is None:
            beta = min(params["BETA_HI"], max(params["BETA_LO"], el[1]))
        growth = ctx["target_n"] / max(ref_n, 1)
        earners_t = (earners_last + 0.5) * (growth ** beta) - 0.5
        rate_el = 100.0 * max(earners_t, 0.005) / ctx["target_n"]
        logits.append(_logit(rate_el))
        weights.append(params["ENS_W"])
    if tr is not None:
        delta = ctx["delta"].get((tier, idx))
        if delta is None:
            delta = min(params["DELTA_HI"], max(params["DELTA_LO"], tr[1]))
        rate_tr = _inv_logit(tr[0] + delta
                             * math.log(1.0 + max(ctx["target_h"], 0.0)))
        logits.append(_logit(rate_tr))
        weights.append(1.0 - params["ENS_W"])
    if not logits:
        return None
    median_logit = sum(l * w for l, w in zip(logits, weights)) / sum(weights)
    resid = tr[2] if tr is not None else ctx["pooled_resid"]
    if resid <= 0:
        resid = ctx["pooled_resid"]
    var = (resid ** 2) * (1.0 + ctx["horizon_h"] / max(ctx["span_h"], 1.0)) \
        + (params["RW"] ** 2) * ctx["horizon_days"] \
        + 1.0 / (earners_last + 1.0)
    sigma = params["SPREAD_SCALE"] * math.sqrt(var)
    return _inv_logit(median_logit), min(3.5, max(0.12, sigma))


def _stale_row(case, states, idx, params, checkpoint_h):
    """Median + sigma for one trophy when the page is frozen / not rich."""
    last_rate, ref_n = _last_seen(states, idx)
    if last_rate is None:
        return 0.05, 1.0
    if last_rate >= 99.9:
        return last_rate, 0.06
    page_age = max(checkpoint_h - states[-1]["start_h"], 0.0)
    eff = page_age + case["horizon_minutes"] / 60.0
    n_mult = math.exp(min(params["STALE_LN_N_RATE"] * eff,
                          math.log(params["STALE_N_MULT_CAP"])))
    resp = 0.6 + 0.4 * min(last_rate / 5.0, 1.0)
    e_mult = math.exp(params["STALE_G"] * resp * eff)
    shift = math.log(e_mult) - math.log(n_mult)
    shift = max(shift, -params["STALE_SHIFT_CAP"])
    median = _inv_logit(_logit(last_rate) + shift)
    earners_last = last_rate * ref_n / 100.0
    sigma = params["STALE_SIGMA"] + 0.3 / math.sqrt(earners_last + 1.0)
    return median, min(3.5, sigma)


def forecast_case(case, overrides=None):
    """Forecast one case. Returns {"trophies": [{"id": n, "q": [...]}]}."""
    params = {**PARAMS, **(overrides or {})}
    if not case.get("history"):
        return _persistence_forecast(case)
    release = case["game"]["release_date"]
    epoch = _parse_time(release + "T00:00:00Z")
    states, checkpoint_h = _page_states(case, epoch)
    ctx = _rich_context(case, states, params, checkpoint_h)
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        got = None
        if ctx["rich"]:
            got = _rich_row(case, states, idx, trophy["tier"], params, ctx)
        if got is None:
            got = _stale_row(case, states, idx, params, checkpoint_h)
        median, sigma = got
        centre = _logit(median)
        qs = [_round4(_inv_logit(centre + z * sigma)) for z in Z_SCORES]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def _persistence_forecast(case):
    """Fallback that never fails: every quantile is the last seen rate."""
    rows = []
    if case.get("history"):
        last_rates = case["history"][-1]["rates"]
    else:
        last_rates = []
    for idx, trophy in enumerate(case["trophies"]):
        rate = last_rates[idx] if idx < len(last_rates) and last_rates[idx] is not None else 0.05
        rows.append({"id": trophy["id"], "q": [_round4(rate)] * 5})
    return {"trophies": rows}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_path in sorted(input_dir.glob("*.in")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = _persistence_forecast(case)
        out_path = output_dir / (case_path.stem + ".out")
        out_path.write_text(json.dumps(forecast), encoding="utf-8")


if __name__ == "__main__":
    main()
