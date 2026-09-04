"""Monte-Carlo aggregation and JSON / Markdown report writing."""

import hashlib
import json
import os
import time


def _outcome_fingerprint(runs):
    """Deterministic outcome-only fingerprint (timings & solver floats excluded).

    Covers exactly what the random stream + assignment decisions produce, so
    two runs with identical parameters must yield identical hashes.
    """
    fp = []
    for r in runs:
        fp.append({
            "seed": r["seed"],
            "leak_value": r["leak_value"],
            "leak_rate": round(r["leak_rate"], 12),
            "final_alive": r["final_alive_targets"],
            "waves": [
                {
                    "wave": w["wave"],
                    "n_new": w["n_new"],
                    "n_stay": w["n_stay"],
                    "destroyed": w["destroyed"],
                    "survived": w["survived"],
                    "destroyed_value": w["destroyed_value"],
                    "cumulative_leak": w["cumulative_leak"],
                }
                for w in r["waves"]
            ],
        })
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def aggregate(runs):
    n = len(runs)
    leak_rates = [r["leak_rate"] for r in runs]
    leak_values = [r["leak_value"] for r in runs]
    mean = sum(leak_rates) / n
    std = (sum((x - mean) ** 2 for x in leak_rates) / n) ** 0.5

    K = len(runs[0]["waves"]) if runs else 0
    # union of observed ages (distance instances: 0..L-1), fixed order
    all_ages = sorted({a for r in runs for w in r["waves"]
                       for a in w.get("age_hist", {})}, key=int)
    by_wave = []
    for k in range(K):
        ws = [r["waves"][k] for r in runs]
        exp_costs = [w["expected_cost"] for w in ws if w["expected_cost"] is not None]
        walls = [w["wall_time"] for w in ws if w["wall_time"] is not None]
        age_mix = {a: sum(w.get("age_hist", {}).get(a, 0) for w in ws) / n
                   for a in all_ages}
        by_wave.append({
            "wave": k,
            "avg_new": sum(w["n_new"] for w in ws) / n,
            "avg_stay": sum(w["n_stay"] for w in ws) / n,
            "avg_targets": sum(w["n_targets"] for w in ws) / n,
            "avg_expected_cost": (sum(exp_costs) / len(exp_costs)) if exp_costs else None,
            "avg_destroyed_value": sum(w["destroyed_value"] for w in ws) / n,
            "avg_cumulative_leak": sum(w["cumulative_leak"] for w in ws) / n,
            "avg_distance": sum(w.get("avg_distance", 0.0) for w in ws) / n,
            "avg_best_p": sum(w.get("avg_best_p", 0.0) for w in ws) / n,
            "avg_age_mix": age_mix,
            "avg_breakthrough": sum(len(w.get("breakthrough", [])) for w in ws) / n,
            "avg_breakthrough_leak": sum(w.get("breakthrough_leak", 0) for w in ws) / n,
            "avg_wall_time": (sum(walls) / len(walls)) if walls else None,
            "solved_count": sum(1 for w in ws if w["solved"]),
        })
    return {
        "runs": n,
        "leak_rate_mean": mean,
        "leak_rate_std": std,
        "leak_rate_min": min(leak_rates),
        "leak_rate_max": max(leak_rates),
        "leak_value_mean": sum(leak_values) / n,
        "breakthrough_count_total_mean":
            sum(r.get("breakthrough_count_total", 0) for r in runs) / n,
        "breakthrough_leak_total_mean":
            sum(r.get("breakthrough_leak_total", 0.0) for r in runs) / n,
        "avg_total_solver_runtime": sum(r["total_solver_runtime"] for r in runs) / n,
        "avg_total_wall_time": sum(r["total_wall_time"] for r in runs) / n,
        "by_wave": by_wave,
    }


def build_report(params, environment, source_md5, instance_info, runs):
    return {
        "params": params,
        "environment": environment,
        "source_md5": source_md5,
        "instance": instance_info,
        "runs": runs,
        "aggregates": aggregate(runs),
        "result_hash": _outcome_fingerprint(runs),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_json(path, report):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)


def _fmt(v, spec=".4f"):
    return ("%" + spec) % v if v is not None else "n/a"


def write_md(path, report):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    P = report["params"]
    A = report["aggregates"]
    L = []
    L.append("# DWTA dynamic simulation report")
    L.append("")
    L.append("- instance: `%s`" % report["instance"]["path"])
    L.append("- m (weapons): %d, n (targets): %d, K (waves): %d, mu: %d, "
             "total value: %d" % (report["instance"]["m"], report["instance"]["n"],
                                  report["instance"]["K"], report["instance"]["mu"],
                                  report["instance"]["total_value"]))
    L.append("- seeds: %d (seed base %d), delta: %g, timelimit: %s, threads: %d, "
             "branching: %s"
             % (P["seeds"], P["seed_base"], P["delta"],
                str(P["timelimit"]), P["threads"], P.get("branching", "probabilities")))
    L.append("- result hash: `%s`" % report["result_hash"])
    L.append("- generated at: %s" % report["generated_at"])
    L.append("")
    L.append("## Per-wave averages (over %d MC runs)" % A["runs"])
    L.append("")
    L.append("| wave | avg new | avg stay | avg targets | avg dist (km) | avg best p | "
             "age mix | avg expected cost | avg destroyed value | bt | avg bt leak | "
             "avg cumulative leak | avg wall time (s) | solved |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for w in A["by_wave"]:
        ages = w.get("avg_age_mix", {})
        age_str = "/".join("%s" % _fmt(ages[a], ".2f")
                           for a in sorted(ages, key=int)) or "n/a"
        L.append("| %d | %.2f | %.2f | %.2f | %.2f | %.4f | %s | %s | %.1f | %.2f | %.1f "
                 "| %.1f | %s | %d/%d |"
                 % (w["wave"], w["avg_new"], w["avg_stay"], w["avg_targets"],
                    w.get("avg_distance", 0.0), w.get("avg_best_p", 0.0), age_str,
                    _fmt(w["avg_expected_cost"]), w["avg_destroyed_value"],
                    w.get("avg_breakthrough", 0.0), w.get("avg_breakthrough_leak", 0.0),
                    w["avg_cumulative_leak"], _fmt(w["avg_wall_time"], ".2f"),
                    w["solved_count"], A["runs"]))
    L.append("")
    L.append("Age mix column: mean number of active targets per age "
             "(age 0 = arriving wave, 1, 2, ...; distance instances cap at L-1).")
    L.append("")
    L.append("## Monte-Carlo summary")
    L.append("")
    L.append("| seed | leak value | leak rate | breakthrough count | breakthrough leak | solver runtime (s) |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for r in report["runs"]:
        L.append("| %d | %d | %.6f | %d | %d | %.2f |"
                 % (r["seed"], r["leak_value"], r["leak_rate"],
                    r.get("breakthrough_count_total", 0),
                    r.get("breakthrough_leak_total", 0),
                    r["total_solver_runtime"]))
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append("| leak rate (mean +- std) | %.6f +- %.6f |" % (A["leak_rate_mean"], A["leak_rate_std"]))
    L.append("| leak rate min / max | %.6f / %.6f |" % (A["leak_rate_min"], A["leak_rate_max"]))
    L.append("| mean leak value | %.1f of %d |" % (A["leak_value_mean"], report["instance"]["total_value"]))
    L.append("| mean breakthrough count (total) | %.2f |" % A.get("breakthrough_count_total_mean", 0.0))
    L.append("| mean breakthrough leak (total) | %.1f |" % A.get("breakthrough_leak_total_mean", 0.0))
    L.append("| avg total solver runtime (s) | %.2f |" % A["avg_total_solver_runtime"])
    L.append("| avg total wall time (s) | %.2f |" % A["avg_total_wall_time"])
    L.append("")
    L.append("Field notes: `expected cost` = solver's optimised expected surviving value "
             "for the wave's target set (effective, distance-scaled probabilities); "
             "`avg dist` / `avg best p` = mean over active targets of the current "
             "distance d_j(k) and best single-weapon effective hit probability; "
             "`cumulative leak` = weight of targets still alive after the wave plus "
             "all breakthrough value so far; `bt` / `bt leak` = targets that broke "
             "through in that wave (alive after their L-th wave) and their leaked "
             "weight; final leak value is the leak after the last wave. "
             "Legacy instances without the d0 column: distance stays 1 km, "
             "no breakthrough occurs.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
