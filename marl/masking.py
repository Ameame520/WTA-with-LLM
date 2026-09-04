"""M4: feasibility masking for the DN-WTA learning policy.

mask[j] = alive_j AND (h_ij <= u_j) AND (B(t) > 0); the hold action is
always feasible. Same recipe as the greedy baseline (dn_policies.py:97):
h_ij = ceil(d_ij/v_m/dt), u_j = ceil(r_j/delta_d), both derived from the
§7 observation + instance public priors only.
"""

import math


def _ceil_int(x, eps=1e-9):
    return int(math.ceil(x - eps))


def feasible_mask(obs_i: dict, t: int, dyn) -> list:
    """Boolean mask over obs_i['targets'] (aligned with that list).

    Returns a list of booleans, one per target entry in obs order; the
    hold action ⊥ is NOT part of this list (always available).
    """
    pub = obs_i["public"]
    own = obs_i["own"]
    pool_ok = obs_i["pool"] > 0
    K = dyn.K
    masks = []
    for tr in obs_i["targets"]:
        if not tr["alive"] or not pool_ok:
            masks.append(False)
            continue
        j = tr["id"]
        r = tr["r"]
        d = own["d"][j]
        u = _ceil_int(r / pub["delta_d"] - 1e-9) if r > 0 else 0
        h = max(1, _ceil_int(d / pub["v_m"] / pub["dt"] - 1e-9)) \
            if d > 0 else 1
        # interceptor must arrive before the target breaks through AND
        # before/at the final decision window (t_hit <= K)
        masks.append(bool(h <= u and t + h <= K))
    return masks


def _selftest():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dwta.dn_instance import DNInstance
    from dwta.dn_env import DNEnv

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dn = DNInstance(os.path.join(here, "data", "dn-data-v3",
                                 "dn_3x50_K10_s01.txt"))
    env = DNEnv(dn, 42)
    for t in range(0, dn.K - 1):
        for i in range(dn.m):
            obs = env.get_observation(i, t)
            m = feasible_mask(obs, t, dn)
            assert len(m) == len(obs["targets"])
            for tr, ok in zip(obs["targets"], m):
                if ok:
                    assert tr["alive"] and obs["pool"] > 0
                    # masked actions must be env-legal
                    assert env.can_fire(i, tr["id"], t)
        # fire on the first feasible target to advance state diversity
        obs = env.get_observation(0, t)
        m = feasible_mask(obs, t, dn)
        for tr, ok in zip(obs["targets"], m):
            if ok:
                env.fire(0, tr["id"], t)
                break
        # manual env advance (same order as DNEnv.run)
        if t + 1 <= dn.K:
            for j in dn.targets_arriving(t + 1):
                env.appeared.add(j)
                env.alive.add(j)
            env._settle_inflight(t + 1)
            env._settle_breakthrough(t + 1)
    print("marl/masking.py selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
