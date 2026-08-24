"""CLI entry point for dynamic multi-wave WTA simulation (DWTA).

Examples (run from the project root):
    python main.py --smoke --seeds 3 --timelimit 30
    python main.py --instance data/dyn_wta_50x100x1_K10.txt \
        --seeds 20 --timelimit 60
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

from dwta.instance import DynInstance, StaticInstanceError  # noqa: E402
from dwta.wave_runner import CplexLimitError, DEFAULT_PYTHON  # noqa: E402
from dwta import simulator  # noqa: E402
from experiments import gen_dynamic_data as gdd, report  # noqa: E402


class Tee:
    """Log to console and to a log file simultaneously."""

    def __init__(self, path):
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


def print_mc_summary(log, runs, aggregates):
    log("")
    log("=== Monte-Carlo summary (%d runs) ===" % aggregates["runs"])
    log("%6s | %10s | %10s | %12s" % ("seed", "leak value", "leak rate", "solver (s)"))
    for r in runs:
        log("%6d | %10d | %10.6f | %12.2f"
            % (r["seed"], r["leak_value"], r["leak_rate"], r["total_solver_runtime"]))
    a = aggregates
    log("leak rate: %.6f +- %.6f (min %.6f, max %.6f)"
        % (a["leak_rate_mean"], a["leak_rate_std"], a["leak_rate_min"], a["leak_rate_max"]))
    log("avg total solver runtime: %.2f s | avg total wall time: %.2f s"
        % (a["avg_total_solver_runtime"], a["avg_total_wall_time"]))
    log("avg stayover count per wave: %s"
        % ["%.1f" % w["avg_stay"] for w in a["by_wave"]])


def handle_cplex_limit(log, err):
    log("")
    log("[ABORT] CPLEX problem-size limit reached (community edition):")
    log("  %s" % err.detail_line)
    log("The community edition of CPLEX limits models to 1000 variables and")
    log("1000 constraints. Options:")
    log("  1) apply for the free IBM CPLEX academic edition (IBM Academic")
    log("     Initiative) and install it into this environment, then re-run;")
    log("  2) or use a smaller instance, e.g. the smoke demo:")
    log("     python main.py --smoke --seeds 3 --timelimit 30")
    log("This simulation is aborted; partial temp files (if any) are cleaned.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dynamic multi-wave WTA simulation demo")
    ap.add_argument("--instance", help="dynamic instance file (see gen_dynamic_data.py)")
    ap.add_argument("--seeds", type=int, default=20, help="number of Monte-Carlo runs")
    ap.add_argument("--seed-base", type=int, default=42, help="base seed (run r uses base+r)")
    ap.add_argument("--timelimit", type=int, default=60, help="per-wave solver timelimit (s)")
    ap.add_argument("--delta", type=float, default=0.001, help="piecewise-linear accuracy")
    ap.add_argument("--threads", type=int, default=1, help="CPLEX threads")
    ap.add_argument("--branching", choices=["probabilities", "cplex"],
                    default="probabilities",
                    help="branching strategy forwarded to wta_cplex.py "
                         "(-branching); use 'cplex' for mu>=3 instances where "
                         "the built-in 'probabilities' branch callback asserts "
                         "on integral values (documented deviation)")
    ap.add_argument("--output", default="output", help="output directory")
    ap.add_argument("--smoke", action="store_true",
                    help="auto-generate and use a small 20x50 K=10 instance")
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="python interpreter used for the solver subprocess")
    ap.add_argument("--policy", choices=["base", "llm"], default="base",
                    help="per-wave decision policy: base (plain CPLEX per wave, "
                         "default) or llm (LLM-assisted strategy, M3)")
    ap.add_argument("--llm-modules", default="",
                    help="comma-separated LLM strategy modules for --policy llm "
                         "(subset of a,b,c,d; default from DWTA_LLM_MODULES env)")
    ap.add_argument("--llm-model", default=os.environ.get("DWTA_LLM_MODEL",
                                                          "deepseek-v4-flash"),
                    help="LLM model name for --policy llm (default DWTA_LLM_MODEL "
                         "env or deepseek-v4-flash)")
    ap.add_argument("--llm-timeout", type=int, default=60,
                    help="per-call LLM API timeout in seconds (default 60)")
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    log = Tee(os.path.join(PROJECT_ROOT, "logs",
                           "dwta_%s.log" % time.strftime("%Y%m%d_%H%M%S")))
    tmp_dir = os.path.join(args.output, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    exit_code = 0

    try:
        smoke = args.smoke or not args.instance
        if smoke:
            log("smoke mode: generating 20 weapons x 50 targets, K=10 (5 targets/wave),"
                " seed=123")
            static = os.path.join(tmp_dir, "smoke_static_20x50x1.txt")
            dyn_path = os.path.join(tmp_dir, "smoke_dyn_20x50x1_K10.txt")
            gdd.make_smoke_static(static, seed=123)
            gdd.convert(static, 10, 123, dyn_path, shuffle=True)
        else:
            dyn_path = args.instance

        log("loading dynamic instance: %s" % dyn_path)
        dyn = DynInstance(dyn_path)
        log("m=%d n=%d K=%d mu=%d total_value=%d"
            % (dyn.m, dyn.n, dyn.K, dyn.mu, dyn.total_value()))

        solver = {"delta": args.delta, "timelimit": args.timelimit,
                  "threads": args.threads, "python": args.python}
        if args.branching != "probabilities":
            # forwarded verbatim to wta_cplex.py via run_solver(extra_args=...)
            solver["extra_args"] = ["-branching", args.branching]
        runs = []

        # per-wave decision policy injection (M3: --policy llm)
        decide_fn = None
        llm_ctx = None
        if args.policy == "llm":
            from dwta import llm_agent
            # isolate the audit stream per experiment directory so the
            # concurrent E2/E3/E4 runs never interleave jsonl lines; the
            # canonical run (--output output) keeps the documented path
            # logs/llm_calls.jsonl (doc deliverable #7)
            llm_ctx = llm_agent.LLMContext(model=args.llm_model,
                                           timeout=args.llm_timeout,
                                           modules=args.llm_modules,
                                           log_dir=(None if args.output == "output"
                                                    else args.output))
            decide_fn = llm_agent.build_policy(llm_ctx)
            log("LLM policy active: model=%s modules=%s timeout=%ds"
                % (llm_ctx.model, "+".join(llm_ctx.modules) or "(none)",
                   llm_ctx.timeout))

        for r in range(args.seeds):
            seed = args.seed_base + r
            log("")
            log("=== MC run %d/%d (seed %d) ===" % (r + 1, args.seeds, seed))
            runs.append(simulator.simulate(dyn, seed, solver, tmp_dir, log,
                                           decide_fn=decide_fn))

        aggregates = report.aggregate(runs)
        print_mc_summary(log, runs, aggregates)

        params = {
            "instance": dyn_path, "smoke": smoke, "seeds": args.seeds,
            "seed_base": args.seed_base, "delta": args.delta,
            "timelimit": args.timelimit, "threads": args.threads,
            "branching": args.branching,
            "policy": args.policy,
            "llm_modules": (llm_ctx.modules if llm_ctx else
                            [m.strip() for m in args.llm_modules.split(",")
                             if m.strip()]),
            "llm_model": args.llm_model, "llm_timeout": args.llm_timeout,
        }
        environment = {"python": args.python, "project_root": PROJECT_ROOT,
                       "argv": sys.argv}
        source_md5 = {
            "cplex/wta_cplex.py": md5_of(os.path.join(PROJECT_ROOT, "cplex",
                                                      "wta_cplex.py")),
            "cplex/validator.py": md5_of(os.path.join(PROJECT_ROOT, "cplex",
                                                      "validator.py")),
            "instance": md5_of(dyn_path),
        }
        instance_info = {"path": dyn_path, "m": dyn.m, "n": dyn.n, "K": dyn.K,
                         "mu": dyn.mu, "total_value": dyn.total_value(),
                         "has_dist": dyn.has_dist, "L": dyn.L, "pcap": dyn.pcap}
        rep = report.build_report(params, environment, source_md5, instance_info, runs)
        # report naming: smoke keeps the legacy demo_report.*; regular runs are
        # named after the policy so E1/E4 land on the documented file names
        # (doc 3.3/5.3: baseline_dist_K10_report / llm_dist_K10_report)
        policy_tag = "baseline" if args.policy == "base" else args.policy
        if smoke:
            stem = "demo_report"
        elif dyn.has_dist:
            stem = "%s_dist_K%d_report" % (policy_tag, dyn.K)
        else:
            stem = "%s_K%d_report" % (policy_tag, dyn.K)
        json_path = os.path.join(args.output, stem + ".json")
        md_path = os.path.join(args.output, stem + ".md")
        report.write_json(json_path, rep)
        report.write_md(md_path, rep)
        log("")
        log("reports written: %s, %s" % (json_path, md_path))
        log("result hash: %s" % rep["result_hash"])
    except CplexLimitError as e:
        handle_cplex_limit(log, e)
        exit_code = 2
    except StaticInstanceError as e:
        log("[ERROR] %s" % e)
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
