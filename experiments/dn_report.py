"""DN-WTA v2 metric aggregation and report writing (spec: 评价指标体系_精简
三层版.md §8 compatibility notes + DN-WTA_v2_数据集说明.md §8).

Metrics implemented on top of dn_env.run records (per seed) - all first-layer
core metrics plus the two DN-specific ones:

    (i)   leak rate          leaked value / total value (+ leaked count)
    (ii)  expected cost      per-decision-step cost_shot (spec §8.1 scope:
                             engaged targets only), episode value = mean
                             over decision steps; cost_all (all visible
                             targets, CPLEX-objective scope) also reported
    (iii) optimality gap     per-step (cost_all - reference) / reference,
                             reference = centralised myopic CPLEX optimum on
                             the same state; cplex policy gap is 0 by
                             definition, greedy computes it when run with
                             --dn-reference
    (iv)  ammo efficiency    expected destroyed value per fired shot
    (v)   engagement age     mean age at first engagement
    (vi)  window coverage    threatened targets engaged at least once / n
                             (v2: all targets are threatened)
    (vii) decision latency   P50/P90/max wall time per decision step +
                             dt-satisfaction rate (single-step <= dt)
    (viii)success rate       legal agent-slots / decision slots
    (xvii)invalid engagement invalid shots / total shots
    (xviii)ammo trajectory   pool curve per step + end-of-episode leftover

Statistics protocol: mean +- std over seeds with worst case; deterministic
result fingerprint (QA gate: two identical runs -> identical hash).
"""

import hashlib
import json
import os
import time


# ----------------------------------------------------------------------
# per-run metrics
# ----------------------------------------------------------------------

def _pctl(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = min(int(len(s) * q), len(s) - 1)
    return s[k]


def run_metrics(run, dn):
    """Extract the metric dict of ONE replication from its run record."""
    steps = run["steps"]
    dec = [s for s in steps if s["decision_step"]]
    walls = [s["wall_time"] for s in dec]

    cost_shots = [s["cost_shot"] for s in dec]
    cost_alls = [s["cost_all"] for s in dec]
    gaps = []
    for s in dec:
        ref = s.get("reference_cost")
        if ref is not None and ref > 1e-12 and s["cost_all"] is not None:
            # clamp floating-point noise: cost_all >= pool-feasible optimum
            gaps.append(max(0.0, (s["cost_all"] - ref) / ref))

    exp_kills = [s["expected_kill"] for s in dec if s["expected_kill"]]
    shots_total = run["shots_total"]

    ages = list(run["first_engagement_age"].values())
    engaged = set(run["first_engagement_age"])
    threatened = set(dn.T)  # v2 endpoint semantics: all targets threaten

    slots = len(dec) * dn.m
    legal = sum(s["legal_actions"] for s in dec)
    illegal = sum(s["illegal_actions"] for s in dec)

    dt = dn.dt
    return {
        # (i)
        "leak_rate": run["leak_rate"],
        "leak_value": run["leak_value"],
        "leak_count": run["leak_count"],
        # (ii)
        "cost_shot_mean": sum(cost_shots) / len(cost_shots) if cost_shots else None,
        "cost_all_mean": sum(cost_alls) / len(cost_alls) if cost_alls else None,
        # (iii)
        "gap_mean": sum(gaps) / len(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "gap_steps": len(gaps),
        # (iv)
        "ammo_efficiency": (sum(exp_kills) / shots_total) if shots_total else None,
        "shots_total": shots_total,
        # (v)
        "engagement_age_mean": sum(ages) / len(ages) if ages else None,
        # (vi)
        "coverage": len(engaged & threatened) / float(len(threatened)),
        # (vii)
        "latency_p50": _pctl(walls, 0.50),
        "latency_p90": _pctl(walls, 0.90),
        "latency_max": max(walls) if walls else None,
        "dt_meet_rate": (sum(1 for w in walls if w <= dt) / len(walls)
                         if walls else None),
        # (viii)
        "success_rate": (legal / float(slots)) if slots else None,
        "illegal_actions": illegal,
        # (xvii)
        "invalid_engagement_rate": (run["invalid_shots"] / float(shots_total)
                                    if shots_total else 0.0),
        # (xviii)
        "ammo_end": run["ammo_end"],
        "destroyed_count": run["destroyed_count"],
        "destroyed_value": run["destroyed_value"],
    }


# ----------------------------------------------------------------------
# aggregation across seeds
# ----------------------------------------------------------------------

def _msw(vals):
    """mean/std/min/max of a list (None entries skipped)."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    mean = sum(xs) / len(xs)
    std = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
    return {"mean": mean, "std": std, "min": min(xs), "max": max(xs)}


AGG_KEYS = [
    "leak_rate", "leak_value", "leak_count",
    "cost_shot_mean", "cost_all_mean", "gap_mean", "gap_max", "gap_steps",
    "ammo_efficiency", "shots_total", "engagement_age_mean", "coverage",
    "latency_p50", "latency_p90", "latency_max", "dt_meet_rate",
    "success_rate", "invalid_engagement_rate", "ammo_end",
    "destroyed_count", "destroyed_value",
]


def aggregate(runs, dn):
    """Aggregate metric dicts + per-step averages across seeds."""
    mets = [run_metrics(r, dn) for r in runs]
    agg = {k: _msw([m[k] for m in mets]) for k in AGG_KEYS}

    n = len(runs)
    K1 = len(runs[0]["steps"])
    by_step = []
    for t in range(K1):
        ss = [r["steps"][t] for r in runs]
        by_step.append({
            "t": t,
            "decision_step": ss[0]["decision_step"],
            "avg_new": sum(len(s["new"]) for s in ss) / n,
            "avg_visible": sum(s["n_visible"] for s in ss) / n,
            "avg_shots": sum(s["shots"] for s in ss) / n,
            "avg_destroyed": sum(len(s["destroyed"]) for s in ss) / n,
            "avg_destroyed_value": sum(s["destroyed_value"] for s in ss) / n,
            "avg_breakthrough": sum(len(s["breakthrough"]) for s in ss) / n,
            "avg_breakthrough_value": sum(s["breakthrough_value"] for s in ss) / n,
            "avg_cost_shot": (sum(s["cost_shot"] for s in ss
                                  if s["cost_shot"] is not None) / n
                              if ss[0]["cost_shot"] is not None else None),
            "avg_cost_all": (sum(s["cost_all"] for s in ss
                                 if s["cost_all"] is not None) / n
                             if ss[0]["cost_all"] is not None else None),
            "avg_reference": (sum(s["reference_cost"] for s in ss
                                  if s.get("reference_cost") is not None) / n
                              if any(s.get("reference_cost") is not None
                                     for s in ss) else None),
            "avg_pool": sum(s["pool_after"] for s in ss) / n,
            "avg_wall_time": (sum(s["wall_time"] for s in ss
                                  if s["wall_time"] is not None) / n),
            "solved_count": sum(1 for s in ss if s["solved"]),
        })
    return {"runs": n, "metrics": agg, "by_step": by_step}


# ----------------------------------------------------------------------
# report assembly
# ----------------------------------------------------------------------

def _fingerprint(runs):
    """Outcome-only fingerprint: random stream + decisions, no timings."""
    fp = []
    for r in runs:
        fp.append({
            "seed": r["seed"],
            "leak_value": r["leak_value"],
            "leak_ids": r["leaked_ids"],
            "destroyed_ids": r["destroyed_ids"],
            "shots": [[s["t"], s["shots"]] for s in r["steps"]],
            "pool_curve": r["pool_curve"],
            "invalid": r["invalid_shots"],
        })
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_report(params, environment, source_md5, instance_info, runs, dn):
    return {
        "meta": {
            "instance": instance_info["path"],
            "m": dn.m, "n": dn.n, "K": dn.K, "mu": dn.mu,
            "pool_total": dn.m * dn.mu,
            "dt": dn.dt, "delta_d": dn.delta_d, "v_m": dn.v_m,
            "pcap": dn.pcap, "total_value": dn.total_value(),
        },
        "params": params,
        "environment": environment,
        "source_md5": source_md5,
        "runs": runs,
        "aggregates": aggregate(runs, dn),
        "metrics_per_run": [run_metrics(r, dn) for r in runs],
        "result_hash": _fingerprint(runs),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_json(path, report):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)


def _fmt(v, spec=".4f"):
    return ("%" + spec) % v if v is not None else "n/a"


def _ms(x):
    if x is None:
        return "n/a"
    return "%.4f +- %.4f" % (x["mean"], x["std"])


def write_md(path, report):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    M = report["meta"]
    P = report["params"]
    A = report["aggregates"]
    met = A["metrics"]
    L = []
    L.append("# DN-WTA v2 simulation report")
    L.append("")
    L.append("- instance: `%s` (m=%d platforms, n=%d targets, K=%d steps, "
             "mu=%d/platform, global pool %d)" % (M["instance"], M["m"], M["n"],
                                                  M["K"], M["mu"], M["pool_total"]))
    L.append("- dynamics: dt=%gs (episode %gs), delta_d=%g km/step, v_m=%g km/s, "
            "pcap=%.2f, total value %d" % (M["dt"], M["dt"] * M["K"],
                                           M["delta_d"], M["v_m"], M["pcap"],
                                           M["total_value"]))
    L.append("- policy: **%s** | seeds: %d (base %d) | solver timelimit %ss, "
            "delta %g, threads %d%s" % (P["policy"], P["seeds"], P["seed_base"],
                                        P["timelimit"], P["delta"], P["threads"],
                                        ", per-step CPLEX gap reference"
                                        if P.get("dn_reference") else ""))
    L.append("- result hash: `%s`" % report["result_hash"])
    L.append("- generated at: %s" % report["generated_at"])
    L.append("")

    # ------------------------------------------------------------------
    L.append("## Core metrics (mean +- std over %d seeds, worst in brackets)" % A["runs"])
    L.append("")
    L.append("| # | metric | value |")
    L.append("|---|---|---|")
    if met["gap_mean"] is None:
        gap_row = "n/a (no reference runs)"
    else:
        gap_row = "%s / %s (%.1f ref steps/run)" % (
            _ms(met["gap_mean"]), _fmt(met["gap_max"]["max"]),
            met["gap_steps"]["mean"] if met["gap_steps"] else 0.0)

    rows = [
        ("(i)", "leak rate [leaked value / %d]" % M["total_value"],
         "%s [max %.4f] | leaked count %s" % (_ms(met["leak_rate"]),
                                              met["leak_rate"]["max"],
                                              _ms(met["leak_count"]))),
        ("(ii)", "expected cost / step (engaged scope)",
         "%s" % _ms(met["cost_shot_mean"])),
        ("(ii)", "expected surviving value / step (CPLEX scope)",
         "%s" % _ms(met["cost_all_mean"])),
        ("(iii)", "optimality gap vs per-step CPLEX (mean / max)",
         gap_row),
        ("(iv)", "ammo efficiency [expected kill value / shot]",
         "%s | shots/run %s" % (_ms(met["ammo_efficiency"]),
                                _ms(met["shots_total"]))),
        ("(v)", "mean engagement age (steps)",
         "%s" % _ms(met["engagement_age_mean"])),
        ("(vi)", "window coverage (threatened engaged >= once)",
         "%s [min %.4f]" % (_ms(met["coverage"]), met["coverage"]["min"])),
        ("(vii)", "decision latency P50 / P90 / max (s)",
         "%s / %s / %s | dt(%gs) meet rate %s" % (
             _ms(met["latency_p50"]), _ms(met["latency_p90"]),
             _fmt(met["latency_max"]["max"], ".3f") if met["latency_max"] else "n/a",
             M["dt"], _ms(met["dt_meet_rate"]))),
        ("(viii)", "success rate (legal action slots)",
         "%s [min %.4f]" % (_ms(met["success_rate"]),
                            met["success_rate"]["min"])),
        ("(xvii)", "invalid engagement rate",
         "%s" % _ms(met["invalid_engagement_rate"])),
        ("(xviii)", "ammo leftover at end / destroyed count / destroyed value",
         "%s | %s | %s" % (_ms(met["ammo_end"]), _ms(met["destroyed_count"]),
                           _ms(met["destroyed_value"]))),
    ]
    for num, name, val in rows:
        L.append("| %s | %s | %s |" % (num, name, val))
    L.append("")

    # ------------------------------------------------------------------
    L.append("## Per-step averages (over %d MC runs)" % A["runs"])
    L.append("")
    L.append("| t | new | visible | shots | cost_shot | cost_all | ref | "
             "kills | kill value | bt | bt value | pool after | wall (s) | solved |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in A["by_step"]:
        L.append("| %d%s | %.2f | %.2f | %.2f | %s | %s | %s | %.2f | %.1f | "
                 "%.2f | %.1f | %.2f | %s | %d/%d |"
                 % (s["t"], "" if s["decision_step"] else " (settle)",
                    s["avg_new"], s["avg_visible"], s["avg_shots"],
                    _fmt(s["avg_cost_shot"], ".2f"), _fmt(s["avg_cost_all"], ".2f"),
                    _fmt(s["avg_reference"], ".2f"),
                    s["avg_destroyed"], s["avg_destroyed_value"],
                    s["avg_breakthrough"], s["avg_breakthrough_value"],
                    s["avg_pool"], _fmt(s["avg_wall_time"], ".3f"),
                    s["solved_count"], A["runs"]))
    L.append("")

    # ------------------------------------------------------------------
    L.append("## Monte-Carlo summary")
    L.append("")
    L.append("| seed | leak value | leak rate | leak count | bt / end | "
             "destroyed | shots | invalid | ammo left |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r, m_ in zip(report["runs"], report["metrics_per_run"]):
        L.append("| %d | %d | %.6f | %d | %d / %d | %d | %d | %d | %d |"
                 % (r["seed"], r["leak_value"], r["leak_rate"], r["leak_count"],
                    r["leak_count_breakthrough"], r["leak_count_end"],
                    r["destroyed_count"], r["shots_total"], r["invalid_shots"],
                    r["ammo_end"]))
    L.append("")
    L.append("Field notes: `cost_shot` = expected surviving value of ENGAGED "
             "targets only (spec (ii) scope); `cost_all` = expected surviving "
             "value over all visible targets (same scope as the per-step "
             "CPLEX objective, used for the gap); `ref` = per-step centralised "
             "myopic CPLEX optimum on the same state (gap reference for "
             "non-CPLEX policies); decisions happen only in t <= K-2, the "
             "last window is settlement-only; alive-at-K targets leak "
             "(v2 endpoint semantics).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
