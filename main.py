"""CLI entry point for the DN-WTA multi-agent time-stepped pipeline.

DN-WTA instances (header 'm n K mu dt delta_d v_m pcap', e.g.
data/dn-data-v3/dn_3x50_K10_s01.txt) run through the multi-agent
time-stepped simulation environment (dwta/dn_env.py) which exposes the
local-observation interface (spec DN-WTA_v2_数据集说明.md §7) that MARL
policies are built on:

    python main.py --instance data/dn-data-v3/dn_3x50_K10_s01.txt \
        --policy greedy --seeds 30
    python main.py --instance data/dn-data-v3/dn_3x50_K10_s01.txt \
        --policy cplex --seeds 30 --timelimit 30
"""

import argparse
import hashlib
import os
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dwta.dn_instance import DNFormatError  # noqa: E402
from dwta.wave_runner import CplexLimitError, DEFAULT_PYTHON  # noqa: E402


class Tee:
    """Log to console and to a log file simultaneously."""

    def __init__(self, path):
        # second-granularity timestamps collide when quick runs (e.g. the
        # no-defense 'none' policy finishes in < 1s) are chained back to
        # back - open("w") would silently truncate the previous run's
        # audit log, so disambiguate with a numeric suffix instead
        base, ext = os.path.splitext(path)
        n = 0
        while os.path.exists(path):
            n += 1
            path = "%s_%d%s" % (base, n, ext)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w")

    def __call__(self, msg=""):
        line = str(msg)
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def handle_cplex_limit(log, err):
    log("")
    log("[ABORT] CPLEX problem-size limit reached (community edition):")
    log("  %s" % err.detail_line)
    log("The community edition of CPLEX limits models to 1000 variables and")
    log("1000 constraints. Options:")
    log("  1) apply for the free IBM CPLEX academic edition (IBM Academic")
    log("     Initiative) and install it into this environment, then re-run;")
    log("  2) or use a smaller instance, e.g. the smoke-sized demo:")
    log("     python main.py --instance data/dy-data-v1/dn_3x10_K10_s1.txt")
    log("        --policy cplex --seeds 3 --timelimit 30")
    log("This simulation is aborted; partial temp files (if any) are cleaned.")


def is_dn_instance(path):
    """DN-WTA files have an 8-field header (m n K mu dt delta_d v_m pcap)."""
    try:
        with open(path, "r") as f:
            head = f.readline().split()
    except OSError:
        return False
    return len(head) == 8


def run_dn(args, log, tmp_dir):
    """DN-WTA pipeline: multi-agent time-stepped simulation."""
    from dwta.dn_instance import DNInstance  # noqa: F401
    from dwta import dn_env, dn_policies
    from experiments import dn_report

    dn = DNInstance(args.instance)
    log("DN-WTA instance: %s" % args.instance)
    log("  m=%d platforms, n=%d targets, K=%d steps (episode %gs), "
        "mu=%d/platform -> global pool %d" % (dn.m, dn.n, dn.K, dn.K * dn.dt,
                                              dn.mu, dn.m * dn.mu))
    log("  dt=%gs delta_d=%gkm/step v_m=%gkm/s pcap=%.2f total_value=%d"
        % (dn.dt, dn.delta_d, dn.v_m, dn.pcap, dn.total_value()))

    solver = {"delta": args.delta, "timelimit": args.timelimit,
              "threads": args.threads, "python": args.python}
    if args.branching != "probabilities":
        solver["extra_args"] = ["-branching", args.branching]

    policy = dn_policies.build_policy(args.policy, solver=solver,
                                      tmp_dir=tmp_dir,
                                      with_reference=args.dn_reference)
    log("policy: %s%s" % (policy.name,
                          " (with per-step CPLEX gap reference)"
                          if args.dn_reference else ""))

    runs = []
    for r in range(args.seeds):
        seed = args.seed_base + r
        log("")
        log("=== MC run %d/%d (seed %d) ===" % (r + 1, args.seeds, seed))
        runs.append(dn_env.simulate_dn(dn, seed, policy, log=log))

    # summary -----------------------------------------------------------
    mets = [dn_report.run_metrics(r_, dn) for r_ in runs]
    log("")
    log("=== Monte-Carlo summary (%d runs) ===" % len(runs))
    log("%6s | %10s | %10s | %9s | %7s | %8s | %7s" %
        ("seed", "leak value", "leak rate", "kills", "shots", "invalid",
         "ammo<"))
    for r_, m_ in zip(runs, mets):
        log("%6d | %10d | %10.6f | %9d | %7d | %8d | %7d" %
            (r_["seed"], r_["leak_value"], r_["leak_rate"],
             r_["destroyed_count"], r_["shots_total"], r_["invalid_shots"],
             r_["ammo_end"]))
    lr = dn_report._msw([m_["leak_rate"] for m_ in mets])
    log("leak rate: %.6f +- %.6f (min %.6f, max %.6f)"
        % (lr["mean"], lr["std"], lr["min"], lr["max"]))

    # report -------------------------------------------------------------
    params = {
        "instance": args.instance, "policy": args.policy,
        "seeds": args.seeds, "seed_base": args.seed_base,
        "delta": args.delta, "timelimit": args.timelimit,
        "threads": args.threads, "branching": args.branching,
        "dn_reference": args.dn_reference,
    }
    environment = {"python": args.python, "project_root": PROJECT_ROOT,
                   "argv": sys.argv}
    source_md5 = {
        "cplex/wta_cplex.py": md5_of(os.path.join(PROJECT_ROOT, "cplex",
                                                  "wta_cplex.py")),
        "cplex/validator.py": md5_of(os.path.join(PROJECT_ROOT, "cplex",
                                                  "validator.py")),
        "dwta/dn_env.py": md5_of(os.path.join(PROJECT_ROOT, "dwta",
                                              "dn_env.py")),
        "dwta/dn_policies.py": md5_of(os.path.join(PROJECT_ROOT, "dwta",
                                                   "dn_policies.py")),
        "instance": md5_of(args.instance),
    }
    instance_info = {"path": args.instance}
    rep = dn_report.build_report(params, environment, source_md5,
                                 instance_info, runs, dn)
    json_path = os.path.join(args.output, "report.json")
    md_path = os.path.join(args.output, "report.md")
    dn_report.write_json(json_path, rep)
    dn_report.write_md(md_path, rep)
    log("")
    log("reports written: %s, %s" % (json_path, md_path))
    log("result hash: %s" % rep["result_hash"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="DN-WTA multi-agent time-stepped simulation")
    ap.add_argument("--instance", required=True,
                    help="DN-WTA instance file (8-field header 'm n K mu "
                         "dt delta_d v_m pcap', see data/ and "
                         "DN-WTA_v3_数据集说明.md)")
    ap.add_argument("--seeds", type=int, default=20, help="number of Monte-Carlo runs")
    ap.add_argument("--seed-base", type=int, default=42, help="base seed (run r uses base+r)")
    ap.add_argument("--timelimit", type=int, default=60, help="per-step solver timelimit (s)")
    ap.add_argument("--delta", type=float, default=0.001, help="piecewise-linear accuracy")
    ap.add_argument("--threads", type=int, default=1, help="CPLEX threads")
    ap.add_argument("--branching", choices=["probabilities", "cplex"],
                    default="probabilities",
                    help="branching strategy forwarded to wta_cplex.py "
                         "(-branching); use 'cplex' for mu>=3 instances where "
                         "the built-in 'probabilities' branch callback asserts "
                         "on integral values (documented deviation)")
    ap.add_argument("--output", default="output", help="output directory")
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="python interpreter used for the solver subprocess")
    ap.add_argument("--policy", choices=["none", "greedy", "cplex"],
                    default="greedy",
                    help="decision policy: none (no-defense lower bound), "
                         "greedy (distributed local greedy, observation-only, "
                         "no solver needed - the interface MARL policies "
                         "plug into), cplex (centralised myopic optimum, "
                         "reference upper bound)")
    ap.add_argument("--dn-reference", action="store_true",
                    help="compute the per-step centralised CPLEX optimum "
                         "alongside a non-CPLEX policy to measure its "
                         "optimality gap (metric iii)")
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    log = Tee(os.path.join(PROJECT_ROOT, "logs",
                           "dwta_%s.log" % time.strftime("%Y%m%d_%H%M%S")))
    tmp_dir = os.path.join(args.output, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    exit_code = 0

    try:
        if not is_dn_instance(args.instance):
            log("[ERROR] not a DN-WTA instance file (header must have 8 "
                "fields 'm n K mu dt delta_d v_m pcap'): %s"
                % args.instance)
            return 1
        return run_dn(args, log, tmp_dir)
    except CplexLimitError as e:
        handle_cplex_limit(log, e)
        exit_code = 2
    except DNFormatError as e:
        log("[ERROR] DN-WTA format: %s" % e)
        exit_code = 1
    except FileNotFoundError as e:
        log("[ERROR] file not found: %s" % e)
        exit_code = 1
    finally:
        try:
            if os.path.isdir(tmp_dir):
                leftovers = sorted(f for f in os.listdir(tmp_dir)
                                   if os.path.isfile(os.path.join(tmp_dir, f)))
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if os.path.isdir(tmp_dir):  # cleanup blocked by environment
                    log("[cleanup] WARNING: could not fully remove %s; "
                        "leftover files: %s" % (tmp_dir, ", ".join(leftovers) or "?"))
                else:
                    log("[cleanup] removed tmp dir %s (%d temp file(s): %s)"
                        % (tmp_dir, len(leftovers),
                           ", ".join(leftovers) if leftovers else "none"))
        except OSError as e:
            log("[cleanup] WARNING: %s" % e)
    log.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
