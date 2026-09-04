"""M5: per-step team reward, potential shaping and kill credit.

All events are read from the post-run DNEnv object and run_rec (zero env
changes). Normalisation base = dyn.total_value().

    R(t) = sum_{breakthrough/end leak @t} (-w_j/total)
         + sum_{kills settled @t}        (+w_j/total)
         - c_invalid * |{invalid settlements @t}|

    R'(t) = R(t) + 0.99 * Phi(s_{t+1}) - Phi(s_t)
    Phi(s) = -sum_{j alive} w_j * pbar_j / total,  Phi(terminal) = 0
    pbar_j(t) = 1 - prod(1 - p_shot) over in-flight interceptors on j
    (in-flight set = shots with t_fire <= t < t_hit, i.e. including this
    step's launches - training-side global information, critic-scope).

Credit attribution: for every shot with outcome == 'kill', its +w_j/total
is booked to the firing slot (t_fire, i).

Reconciliation (selftest-asserted):
    * sum_t R_team == (destroyed_value - leak_value) / total   < 1e-9
    * kill-shot count == len(env.destroyed_at)
    * credit keys subset of env.shots (t_fire, i) keys
"""

import torch

C_INVALID = 0.01        # per invalid settlement (locked, 0.005-0.05)
GAMMA_SHAPE = 0.99


def build_rewards(env, run_rec: dict, dyn,
                  c_invalid: float = C_INVALID) -> dict:
    total = float(dyn.total_value())
    K = dyn.K
    steps = K + 1                                  # t = 0..K

    # ---- raw event streams from the post-run env ----------------------
    kills_at = {}                                  # t -> value sum
    for j, t in env.destroyed_at.items():
        kills_at[t] = kills_at.get(t, 0.0) + dyn.w[j]
    leak_at = {}                                   # t -> value sum
    for j, (t, _cause) in env.leaked_at.items():
        leak_at[t] = leak_at.get(t, 0.0) + dyn.w[j]
    invalid_at = {}                                # t -> count
    for ev in env.shots:
        if ev.get("outcome") == "invalid":
            t = ev["t_hit"]
            invalid_at[t] = invalid_at.get(t, 0) + 1

    R_team = torch.zeros(steps, dtype=torch.float64)
    R_pure = torch.zeros(steps, dtype=torch.float64)   # no invalid penalty
    for t in range(steps):
        r = 0.0
        r -= leak_at.get(t, 0.0) / total
        r += kills_at.get(t, 0.0) / total
        R_pure[t] = r
        R_team[t] = r - c_invalid * invalid_at.get(t, 0)

    # ---- potential Phi(t): state AFTER step t's events ----------------
    # replay alive set + in-flight set from the recorded outcomes
    Phi = torch.zeros(steps + 1, dtype=torch.float64)   # Phi[steps] = terminal = 0
    alive = set()
    fired = sorted(env.shots, key=lambda e: (e["t_fire"], e["i"]))
    for t in range(steps):
        for j in dyn.targets_arriving(t):
            alive.add(j)
        alive.discard(None)
        # remove destroyed / leaked at t
        for j, td in env.destroyed_at.items():
            if td == t:
                alive.discard(j)
        for j, (tl, _c) in env.leaked_at.items():
            if tl == t:
                alive.discard(j)
        # in-flight including this step's launches (t_fire <= t < t_hit)
        phi = 0.0
        for j in list(alive):
            p_surv = 1.0
            for ev in fired:
                if ev["j"] == j and ev["t_fire"] <= t < ev["t_hit"]:
                    p_surv *= (1.0 - ev["p_shot"])
            pbar = 1.0 - p_surv
            if pbar > 0.0:
                phi -= dyn.w[j] * pbar / total
        Phi[t] = phi

    R_shaped = torch.zeros(steps, dtype=torch.float64)
    for t in range(steps):
        nxt = Phi[t + 1] if t + 1 <= steps else 0.0
        R_shaped[t] = R_team[t] + GAMMA_SHAPE * nxt - Phi[t]

    # ---- credit attribution -------------------------------------------
    credit = {}
    for ev in env.shots:
        if ev.get("outcome") == "kill":
            key = (ev["t_fire"], ev["i"])
            credit[key] = credit.get(key, 0.0) + dyn.w[ev["j"]] / total

    shots_detail = [dict(ev) for ev in env.shots]

    return {
        "R_team": R_team,
        "R_pure": R_pure,
        "R_shaped": R_shaped,
        "credit": credit,
        "shots_detail": shots_detail,
    }


# ----------------------------------------------------------------------
# selftest: offline replay reconciliation on real episodes
# ----------------------------------------------------------------------

def _selftest(instances):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dwta.dn_instance import DNInstance
    from dwta.dn_env import DNEnv
    from dwta.dn_policies import build_policy

    total_checks = 0
    for path in instances:
        dn = DNInstance(path)
        for seed in (42, 43):
            env = DNEnv(dn, seed)
            pol = build_policy("greedy")
            run_rec = env.run(pol)
            out = build_rewards(env, run_rec, dn)
            total_val = float(dn.total_value())
            lhs = float(out["R_pure"].sum())
            rhs = (run_rec["destroyed_value"] - run_rec["leak_value"]) \
                / total_val
            err = abs(lhs - rhs)
            assert err < 1e-9, "reconciliation error %g on %s seed %d" \
                % (err, path, seed)
            n_kills = sum(1 for s in out["shots_detail"]
                          if s.get("outcome") == "kill")
            assert n_kills == len(env.destroyed_at), "kill count mismatch"
            valid_keys = {(s["t_fire"], s["i"]) for s in env.shots}
            assert set(out["credit"]).issubset(valid_keys), "credit leak"
            total_checks += 1
            print("  %s seed %d: R_team sum=%.6f reconciled (err %.2e), "
                  "kills=%d, credit slots=%d"
                  % (os.path.basename(path), seed, lhs, err, n_kills,
                     len(out["credit"])))
    print("marl/reward.py selftest: ALL PASS (%d episodes)" % total_checks)


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--instances", default=None)
    args = ap.parse_args()
    if args.selftest:
        insts = [p.strip() for p in args.instances.split(",")] \
            if args.instances else []
        if not insts:
            raise SystemExit("--selftest requires --instances")
        _selftest(insts)
    else:
        print("use --selftest --instances <files>")
