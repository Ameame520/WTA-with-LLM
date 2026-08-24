"""Multi-wave simulation loop with BDA (Bernoulli damage assessment) sampling.

Wave coupling rules (this stage):
  * ammunition: each weapon is reset to `mu` shots per wave (no cross-wave ammo);
  * targets: surviving targets stay (with their original weight) into the next
    wave; destroyed targets are removed permanently.

Distance extension (instance has a d0 column): targets close 1 km per wave.
The per-wave sub-instance and the Bernoulli settlement both use the effective
hit rate p_eff(i,j,k) = min(pcap, p_ij*d0_j/d_j(k)). A target still alive
with age >= L-1 after the settlement of its L-th wave breaks through: it
leaks its full weight and leaves the scenario. Cumulative leak = alive stock
+ breakthrough value so far. Legacy files (no d0 column) behave exactly as
before (no closing, no breakthrough).
"""

import os

import numpy as np

from . import wave_runner


def decide(state):
    """Pure decision interface: state -> assignment.

    This is the single per-wave decision entry point, kept functionally pure
    (no hidden simulator state is consulted) so that it can later be replaced
    by an RL / LLM policy without touching the simulation loop.

    state keys:
        wave_idx       : current wave index k
        alive_targets  : list of (orig_j, w_j) present this wave (new + stayovers)
        p              : dict {(weapon_i, orig_j): effective prob at wave k
        ages           : dict {orig_j: k - k_arr(j)}  (0 at arrival)
        distances      : dict {orig_j: d_j(k) in km}
        ammo           : list of shots per weapon this wave
        m, mu          : weapon count / per-weapon capacity
        dyn            : DynInstance (read-only)
        tmp_inst       : path for the temporary static instance
        tmp_sol        : path for the temporary solution file
        solver         : dict(delta, timelimit, threads, python[, extra_args])

    Returns (assignment, solve_info):
        assignment : {orig_j: {weapon_i: shots}} or {} when the wave was not solved
        solve_info : dict(objective, solver_runtime, wall_time, solved, warning)
    """
    dyn = state["dyn"]
    target_ids = sorted(j for j, _ in state["alive_targets"])

    wave_runner.write_wave_instance(state["tmp_inst"], dyn, target_ids,
                                    wave_idx=state["wave_idx"])

    info = {"objective": None, "solver_runtime": None, "wall_time": None,
            "solved": False, "warning": None}
    rc, output, wall = wave_runner.run_solver(
        state["tmp_inst"], state["tmp_sol"],
        delta=state["solver"]["delta"], timelimit=state["solver"]["timelimit"],
        threads=state["solver"]["threads"], python_exe=state["solver"]["python"],
        extra_args=state["solver"].get("extra_args"))

    info["wall_time"] = wall
    parsed = wave_runner.parse_wave_solution(state["tmp_inst"], state["tmp_sol"], target_ids)
    if parsed is None:
        info["warning"] = ("no solution file produced (rc=%s) - wave not engaged, "
                           "all targets stay over" % rc)
        return {}, info
    info["objective"] = parsed["objective"]
    info["solver_runtime"] = parsed["runtime"]
    info["solved"] = True
    return parsed["assignment"], info


def survival_probability(dyn, j, assignment_j, wave_idx=None):
    """Probability that target j survives its current-wave assignment.

    wave_idx given: effective hit rates dyn.effective_p(i, j, wave_idx);
    omitted (legacy calls): base dyn.p.
    """
    prod = 1.0
    for i, v in assignment_j.items():
        pij = dyn.p[i, j] if wave_idx is None else dyn.effective_p(i, j, wave_idx)
        prod *= (1.0 - pij) ** v
    return prod


def simulate(dyn, seed, solver, tmp_dir, log, decide_fn=None):
    """Run one full Monte-Carlo replication (all K waves) for one random stream.

    decide_fn: per-wave decision policy with the decide(state) interface;
    defaults to the built-in CPLEX-per-wave policy. This is the injection
    point for the M3 LLM strategies.
    Returns a per-run record dict (JSON-serialisable).
    """
    if decide_fn is None:
        decide_fn = decide
    rng = np.random.RandomState(seed)
    alive = {}  # orig_j -> weight
    total_value = dyn.total_value()
    breakthrough_leak_total = 0  # value leaked via breakthrough (dist instances)
    breakthrough_count_total = 0
    waves_rec = []
    os.makedirs(tmp_dir, exist_ok=True)

    for k in range(dyn.K):
        new_ids = dyn.targets_in_wave(k)
        for j in new_ids:
            alive[j] = dyn.w[j]
        stay_ids = [j for j in alive if j not in set(new_ids)]
        target_ids = sorted(alive)
        ages = {j: k - dyn.wave[j] for j in target_ids}
        distances = {j: dyn.dist(j, k) for j in target_ids}

        tmp_inst = os.path.join(tmp_dir, "wave%d_inst.txt" % k)
        tmp_sol = os.path.join(tmp_dir, "wave%d.sol" % k)
        state = {
            "wave_idx": k,
            "alive_targets": [(j, alive[j]) for j in target_ids],
            "p": {(i, j): dyn.effective_p(i, j, k)
                  for i in dyn.W for j in target_ids},
            "ages": ages,
            "distances": distances,
            "ammo": [dyn.mu] * dyn.m,
            "m": dyn.m,
            "mu": dyn.mu,
            "dyn": dyn,
            "tmp_inst": tmp_inst,
            "tmp_sol": tmp_sol,
            "solver": solver,
        }
        assignment, solve_info = decide_fn(state)

        # Bernoulli damage settlement, targets visited in ascending id order
        # so the random stream is fully reproducible. Effective hit rates
        # (distance-scaled) are used; legacy instances fall back to base p.
        destroyed = []
        survived = []
        for j in target_ids:
            surv = survival_probability(dyn, j, assignment.get(j, {}), wave_idx=k)
            u = rng.random_sample()
            if u < surv:
                survived.append(j)
            else:
                destroyed.append(j)
        for j in destroyed:
            del alive[j]

        # Breakthrough: a target alive after the settlement of its L-th wave
        # (age >= L-1) leaks its full weight and leaves the scenario.
        # Only applies to distance instances; legacy files never trigger it.
        breakthrough = [j for j in survived if dyn.has_dist and ages[j] >= dyn.L - 1]
        breakthrough_set = set(breakthrough)
        for j in breakthrough:
            del alive[j]
        breakthrough_leak = sum(dyn.w[j] for j in breakthrough)
        breakthrough_leak_total += breakthrough_leak
        breakthrough_count_total += len(breakthrough)
        # targets still in the scenario after this wave (breakthrough removed)
        survived_in_scenario = [j for j in survived if j not in breakthrough_set]

        destroyed_value = sum(dyn.w[j] for j in destroyed)
        cumulative_leak = sum(alive.values()) + breakthrough_leak_total

        # per-wave dynamics statistics (hit rate / distance)
        n_t = len(target_ids)
        avg_distance = (sum(distances.values()) / float(n_t)) if n_t else 0.0
        best_p = {j: max(dyn.effective_p(i, j, k) for i in dyn.W) for j in target_ids}
        avg_best_p = (sum(best_p.values()) / float(n_t)) if n_t else 0.0
        age_hist = {}
        for j in target_ids:
            age_hist[str(ages[j])] = age_hist.get(str(ages[j]), 0) + 1

        rec = {
            "wave": k,
            "new_targets": new_ids,
            "stayover_targets": stay_ids,
            "n_new": len(new_ids),
            "n_stay": len(stay_ids),
            "n_targets": len(target_ids),
            "solved": solve_info["solved"],
            "objective": solve_info["objective"],
            "solver_runtime": solve_info["solver_runtime"],
            "wall_time": solve_info["wall_time"],
            "warning": solve_info["warning"],
            "expected_cost": (solve_info["objective"]
                              if solve_info["solved"] else None),
            "survival_probs": {str(j): survival_probability(dyn, j, assignment.get(j, {}),
                                                            wave_idx=k)
                               for j in target_ids},
            "assignment": {str(j): {str(i): v for i, v in a.items()}
                           for j, a in assignment.items()},
            "destroyed": destroyed,
            "survived": survived_in_scenario,
            "breakthrough": breakthrough,
            "breakthrough_leak": breakthrough_leak,
            "destroyed_value": destroyed_value,
            "cumulative_leak": cumulative_leak,
            "avg_distance": avg_distance,
            "avg_best_p": avg_best_p,
            "age_hist": age_hist,
        }
        # M3 LLM hook: after the settlement the policy may run its post-solve
        # module (decision explanation) and attach keys to the wave record
        # (e.g. rec["llm"]). It must never touch the random stream, so the
        # replication stays reproducible; failures are recorded, not raised.
        if hasattr(decide_fn, "on_wave_end"):
            try:
                decide_fn.on_wave_end(rec, state)
            except Exception as exc:  # noqa: BLE001 - hook must not kill the run
                rec.setdefault("llm", {})["post_warning"] = str(exc)[:300]

        waves_rec.append(rec)

        log("  wave %2d | new %3d | stay %3d | avg d %5.2f | exp cost %10s | "
            "destroyed %8d | bt %3d | cum leak %8d | %6.2fs%s"
            % (k, rec["n_new"], rec["n_stay"], avg_distance,
               ("%.4f" % rec["expected_cost"]) if rec["expected_cost"] is not None else "n/a",
               destroyed_value, len(breakthrough), cumulative_leak,
               rec["wall_time"] if rec["wall_time"] is not None else float("nan"),
               "  [WARNING: %s]" % rec["warning"] if rec["warning"] else ""))

        # Temp files are intentionally kept till the end of the whole run and
        # removed once by the caller's try/finally (see run_demo.py): per-wave
        # names wave<k>_inst.txt / wave<k>.sol are simply overwritten in the
        # next MC replication, so no residue accumulates mid-run either.

    # Final leak = stock still alive at scenario end + all breakthrough value.
    leak_value = sum(alive.values()) + breakthrough_leak_total
    run_rec = {
        "seed": seed,
        "waves": waves_rec,
        "leak_value": leak_value,
        "total_value": total_value,
        "leak_rate": leak_value / total_value if total_value else 0.0,
        "breakthrough_count_total": breakthrough_count_total,
        "breakthrough_leak_total": breakthrough_leak_total,
        "total_solver_runtime": sum(w["solver_runtime"] or 0.0 for w in waves_rec),
        "total_wall_time": sum(w["wall_time"] or 0.0 for w in waves_rec),
        "final_alive_targets": sorted(alive),
    }
    return run_rec
