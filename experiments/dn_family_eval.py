"""CLI: evaluate one policy over a fixed DN-WTA v3 split (thin wrapper).

Reads the split assignment from data/dn-data-v3/MANIFEST.md, re-uses the DN
pipeline as-is (dwta.dn_env / dwta.dn_policies / experiments.dn_report -
no pipeline code duplicated here), and writes ONE flat output folder:

    family_report.json   per-instance metrics + cross-instance family
                         aggregates; raw per-seed run records embedded
                         (no sub-folders)
    family_report.md     human-readable summary (instances x metrics)

Splits are FIXED by the MANIFEST (test = s01-s02, train = s03-s26,
val = s27-s30). Formal algorithm comparison runs on test only (2
instances x 30 MC seeds); train/val exist for MARL training/early stop.

Usage (from the project root):

    python experiments/dn_family_eval.py --split test --policy cplex \
        --seeds 30 --seed-base 42 --timelimit 30 --output output/e13_dn3_cplex

Policies: none / greedy / cplex (same registry as main.py --policy for DN
instances). greedy automatically runs with the per-step CPLEX reference
(--dn-reference semantics) so its gap (metric iii) is computed.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dwta.dn_instance import DNInstance  # noqa: E402
from dwta import dn_env, dn_policies  # noqa: E402
from dwta.wave_runner import DEFAULT_PYTHON  # noqa: E402
from experiments import dn_report  # noqa: E402

MANIFEST = os.path.join(PROJECT_ROOT, "data", "dn-data-v3", "MANIFEST.md")

# family-level aggregation keys: (metric key, worst direction)
FAMILY_KEYS = [
    ("leak_rate", "max"),
    ("gap_mean", "max"),
    ("invalid_engagement_rate", "max"),
    ("ammo_efficiency", "min"),
    ("latency_p50", "max"),
    ("latency_p90", "max"),
    ("shots_total", "max"),
    ("destroyed_value", "min"),
]


def read_split(manifest_path, split):
    """Parse the per-instance index table of the MANIFEST and return the
    file names assigned to `split` (sorted)."""
    insts = []
    for ln in open(manifest_path):
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        if len(parts) < 3:
            continue
        name = parts[0].strip("`")
        if name.startswith("dn_") and name.endswith(".txt") and parts[1] == split:
            insts.append(name)
    if not insts:
        raise SystemExit("[ERROR] no '%s' instances found in %s"
                         % (split, manifest_path))
    return sorted(insts)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_log(path):
    """Console echo + optional log file (same behaviour as main.py's Tee)."""
    if path is None:
        return lambda msg="": print(msg, flush=True)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    f = open(path, "w")

    def log(msg=""):
        print(str(msg), flush=True)
        f.write(str(msg) + "\n")
        f.flush()
    return log


def family_aggregate(per_instance):
    """Cross-instance aggregation over the per-seed means of each instance.

    For every FAMILY_KEY: collect instance-level means (already averaged
    over the MC seeds inside dn_report.aggregate), report mean/std and the
    worst instance-level mean (direction per key)."""
    fam = {}
    for key, worst in FAMILY_KEYS:
        vals = []
        for inst in per_instance:
            m = inst["aggregates"]["metrics"].get(key)
            if m is not None and m["mean"] is not None:
                vals.append(m["mean"])
        if not vals:
            fam[key] = None
            continue
        mean = sum(vals) / len(vals)
        std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        fam[key] = {
            "mean": mean, "std": std,
            "worst": max(vals) if worst == "max" else min(vals),
            "n_instances": len(vals),
        }
    return fam


def _ms(x):
    if x is None:
        return "n/a"
    return "%.4f +- %.4f [worst %.4f]" % (x["mean"], x["std"], x["worst"])


def write_md(path, rep):
    P, F = rep["params"], rep["family"]
    L = []
    L.append("# DN-WTA v3 family evaluation report")
    L.append("")
    L.append("- split: **%s** (%d instances: %s) | policy: **%s** | "
             "seeds: %d (base %d) | solver timelimit %ss"
             % (P["split"], len(rep["instances"]), P["instances"],
                P["policy"], P["seeds"], P["seed_base"], P["timelimit"]))
    L.append("- generated at: %s" % rep["generated_at"])
    L.append("")
    L.append("## Per-instance metrics (mean +- std over %d MC seeds)" % P["seeds"])
    L.append("")
    L.append("| instance | total value | leak rate | gap mean | invalid | "
             "ammo eff | latency p50 (s) | shots | destroyed value | result hash |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for inst in rep["instances"]:
        m = inst["aggregates"]["metrics"]

        def s(k, spec=".4f"):
            x = m.get(k)
            return ("%" + spec) % x["mean"] if x and x["mean"] is not None else "n/a"
        L.append("| `%s` | %d | %s | %s | %s | %s | %s | %s | %s | `%s` |"
                 % (inst["instance"], inst["meta"]["total_value"],
                    s("leak_rate"), s("gap_mean"), s("invalid_engagement_rate"),
                    s("ammo_efficiency"), s("latency_p50", ".3f"),
                    s("shots_total", ".1f"), s("destroyed_value", ".1f"),
                    inst["result_hash"][:12]))
    L.append("")
    L.append("## Family aggregates (across %d instances, per-key worst)"
             % len(rep["instances"]))
    L.append("")
    L.append("| metric | mean +- std | worst instance mean |")
    L.append("|---|---|---|")
    for key, _ in FAMILY_KEYS:
        L.append("| %s | %s | %s |" % (key, _ms(F.get(key)),
                                       ("%.4f" % F[key]["worst"])
                                       if F.get(key) else "n/a"))
    L.append("")
    L.append("Notes: `gap mean` is the per-step optimality gap vs the "
             "centralised myopic CPLEX reference (0 by definition for the "
             "cplex policy; n/a when no reference runs); evaluation protocol "
             "and split are fixed by `data/dn-data-v3/MANIFEST.md`.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="DN-WTA v3 family evaluation (one policy over one split)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="instance split from the MANIFEST (test = s01-s02, "
                         "the formal comparison benchmark)")
    ap.add_argument("--policy", choices=["none", "greedy", "cplex"],
                    required=True)
    ap.add_argument("--seeds", type=int, default=30,
                    help="Monte-Carlo seeds per instance (v3 protocol: 30)")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--timelimit", type=int, default=30,
                    help="per-step CPLEX timelimit (s)")
    ap.add_argument("--delta", type=float, default=0.001)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="python interpreter for the solver subprocess")
    ap.add_argument("--data-dir", default="data/dn-data-v3")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--output", required=True,
                    help="flat output folder, e.g. output/e13_dn3_cplex")
    ap.add_argument("--log", default=None,
                    help="optional terminal log file, e.g. logs/e13_run.log")
    args = ap.parse_args(argv)

    instances = read_split(args.manifest, args.split)
    log = make_log(args.log)
    os.makedirs(args.output, exist_ok=True)
    tmp_dir = os.path.join(args.output, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    solver = {"delta": args.delta, "timelimit": args.timelimit,
              "threads": args.threads, "python": args.python}
    # greedy automatically runs with the per-step CPLEX reference
    # (--dn-reference semantics); cplex IS the reference; none needs nothing
    with_ref = args.policy == "greedy"
    policy = dn_policies.build_policy(args.policy, solver=solver,
                                      tmp_dir=tmp_dir,
                                      with_reference=with_ref)

    log("DN-WTA v3 family evaluation")
    log("  split=%s instances=%s" % (args.split, ", ".join(instances)))
    log("  policy=%s%s seeds=%d (base %d) timelimit=%ss"
        % (args.policy, " (with per-step CPLEX reference)" if with_ref else "",
           args.seeds, args.seed_base, args.timelimit))

    per_instance = []
    try:
        for name in instances:
            path = os.path.join(args.data_dir, name)
            dn = DNInstance(path)
            log("")
            log("=== %s (m=%d n=%d K=%d mu=%d pool=%d total_value=%d) ==="
                % (name, dn.m, dn.n, dn.K, dn.mu, dn.m * dn.mu,
                   dn.total_value()))
            runs = []
            for r in range(args.seeds):
                seed = args.seed_base + r
                runs.append(dn_env.simulate_dn(dn, seed, policy))
                if (r + 1) % 10 == 0 or r + 1 == args.seeds:
                    log("  MC runs done: %d/%d" % (r + 1, args.seeds))
            agg = dn_report.aggregate(runs, dn)
            m = agg["metrics"]
            log("  leak rate %.6f +- %.6f | shots %.1f | destroyed value %.1f"
                % (m["leak_rate"]["mean"], m["leak_rate"]["std"],
                   m["shots_total"]["mean"], m["destroyed_value"]["mean"]))
            per_instance.append({
                "instance": name,
                "meta": {"m": dn.m, "n": dn.n, "K": dn.K, "mu": dn.mu,
                         "pool_total": dn.m * dn.mu,
                         "total_value": dn.total_value()},
                "runs": runs,
                "metrics_per_run": [dn_report.run_metrics(r_, dn)
                                    for r_ in runs],
                "aggregates": agg,
                "result_hash": dn_report._fingerprint(runs),
            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log("[cleanup] removed tmp dir %s" % tmp_dir)

    family = family_aggregate(per_instance)
    rep = {
        "meta": {"dataset": "DN-WTA v3", "split": args.split,
                 "manifest": os.path.relpath(args.manifest, PROJECT_ROOT)},
        "params": {"split": args.split, "policy": args.policy,
                   "seeds": args.seeds, "seed_base": args.seed_base,
                   "timelimit": args.timelimit, "delta": args.delta,
                   "threads": args.threads, "dn_reference": with_ref,
                   "data_dir": args.data_dir, "instances": instances},
        "environment": {"python": args.python, "argv": sys.argv},
        "source_md5": {
            "cplex/wta_cplex.py": md5_of(os.path.join(PROJECT_ROOT, "cplex",
                                                      "wta_cplex.py")),
            "cplex/validator.py": md5_of(os.path.join(PROJECT_ROOT, "cplex",
                                                      "validator.py")),
            "dwta/dn_env.py": md5_of(os.path.join(PROJECT_ROOT, "dwta",
                                                  "dn_env.py")),
            "dwta/dn_policies.py": md5_of(os.path.join(PROJECT_ROOT, "dwta",
                                                       "dn_policies.py")),
            "experiments/dn_family_eval.py": md5_of(os.path.abspath(__file__)),
        },
        "instances": per_instance,
        "family": family,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = os.path.join(args.output, "family_report.json")
    md_path = os.path.join(args.output, "family_report.md")
    with open(json_path, "w") as f:
        json.dump(rep, f, indent=2, sort_keys=False)
    write_md(md_path, rep)
    log("")
    log("reports written: %s, %s" % (json_path, md_path))
    log("family leak rate: %s" % _ms(family["leak_rate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
