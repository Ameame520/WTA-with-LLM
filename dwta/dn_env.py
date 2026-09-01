"""DN-WTA v2 multi-agent time-stepped simulation environment (spec: DN-WTA_v2
数据集说明.md §4 / §7 / §9).

Differences vs the legacy wave simulator (dwta/simulator.py, dyn_* instances):

  * time structure : decisions only in t = 0..K-2; window K-1 is settlement
                     and statistics only; t = K is the end-of-episode
                     settlement (alive & not destroyed -> leak);
  * ammo           : ONE global shared pool (m * mu shots for the whole
                     episode, never replenished); each agent fires at most
                     1 interceptor per step;
  * settlement     : DELAYED - a shot fired at step t locks p_shot =
                     p_eff(i, j, t) and settles Bernoulli at t_hit = t + h;
  * targets        : appear at k_arr, close delta_d km per step to the
                     boundary; r_j(t) = 0 while alive -> breakthrough leak;
  * info boundary  : policies must use get_observation(i, t) (§7) - only
                     shared target info, own p/d/inflight, public env
                     constants; n / future arrivals / other agents hidden.

Per-step order (fixed, reproducible random stream):
    1. targets with k_arr = t appear;
    2. inflight shots with t_hit = t settle (ascending (t_fire, i));
       shots arriving at an already destroyed / leaked target count as
       invalid engagements (metric xvii), ammo NOT returned;
    3. alive targets with r_j(t) <= 0 break through (full weight leaks;
       only t < K - targets whose breakthrough step is >= K stay in the
       end settlement, matching the dataset spec §5 "期末" accounting);
    4. if t <= K-2: the policy acts (per-agent target choice or hold);
    5. at t = K: every still-alive target leaks (end settlement).

This module is the environment only; policies live in dwta/dn_policies.py.
"""

import time

import numpy as np

from .dn_instance import DNInstance


class DNEnv(object):
    """One full-episode environment for one Monte-Carlo replication."""

    def __init__(self, dn: DNInstance, seed: int):
        self.dn = dn
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # global shared ammo pool (spec §3.1: m * mu shots, no replenish)
        self.pool0 = dn.m * dn.mu
        self.pool = self.pool0

        # target bookkeeping
        self.appeared = set()          # ids detected so far
        self.alive = set()             # appeared, not destroyed, not leaked
        self.destroyed_at = {}         # j -> settlement step
        self.leaked_at = {}            # j -> (step, 'breakthrough' | 'end')

        # ordnance bookkeeping
        self.inflight = []             # pending shot events (dicts)
        self.shots = []                # every fired shot (all stay here)
        self.invalid_shots = 0         # settled on dead/leaked target (xvii)

        # engagement bookkeeping (metrics v / vi)
        self.first_engagement_age = {}  # j -> age at first shot
        self.engaged_targets = set()    # ids shot at least once

    # ------------------------------------------------------------------
    # information boundary (spec §7)
    # ------------------------------------------------------------------

    def visible_targets(self, t: int):
        """Ids appeared, alive (not destroyed / leaked) at decision time."""
        return sorted(self.alive)

    def get_observation(self, i: int, t: int):
        """Local observation of agent i at step t (spec §7).

        Shared: current t, appeared target ids (with w / r / alive status /
        age), global pool level (shared resource counter).
        Own: base p_ij, current d_ij(t), own inflight interceptors.
        Public constants: K, dt, delta_d, v_m, pcap, m.
        Hidden: n, future k_arr, other agents' p/d/ammo/actions/inflight,
        how many interceptors attack each target.
        """
        dn = self.dn
        targets = []
        for j in sorted(self.appeared):
            targets.append({
                "id": j,
                "w": dn.w[j],
                "r": dn.r(j, t),
                "age": dn.age(j, t),
                "alive": j in self.alive,
            })
        return {
            "t": t,
            "public": {"K": dn.K, "m": dn.m, "dt": dn.dt, "delta_d": dn.delta_d,
                       "v_m": dn.v_m, "pcap": dn.pcap},
            "pool": self.pool,
            "targets": targets,
            "own": {
                "p": {j: dn.p[i, j] for j in sorted(self.appeared)},
                "d": {j: dn.dist(i, j, t) for j in sorted(self.appeared)},
                "inflight": [dict(ev) for ev in self.inflight if ev["i"] == i],
            },
        }

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def can_fire(self, i: int, j: int, t: int):
        """Legality of agent i engaging target j at step t."""
        return (0 <= i < self.dn.m and j in self.alive
                and self.pool > 0 and t <= self.dn.K - 2)

    def fire(self, i: int, j: int, t: int):
        """Execute one launch (caller must have checked can_fire).

        p_shot is locked at fire time; settlement happens at t_hit = t + h.
        """
        dn = self.dn
        ev = {
            "i": i, "j": j, "t_fire": t,
            "h": dn.flight_steps(i, j, t),
            "p_shot": dn.p_eff(i, j, t),
        }
        ev["t_hit"] = t + ev["h"]
        self.inflight.append(ev)
        self.shots.append(ev)
        self.pool -= 1
        self.engaged_targets.add(j)
        if j not in self.first_engagement_age:
            self.first_engagement_age[j] = dn.age(j, t)
        return ev

    def _settle_inflight(self, t: int):
        """Bernoulli settlement of all inflight shots with t_hit == t.

        Fixed order (t_fire, i) keeps the random stream reproducible.
        Returns list of targets destroyed at this step.
        """
        due = sorted((ev for ev in self.inflight if ev["t_hit"] == t),
                     key=lambda e: (e["t_fire"], e["i"]))
        destroyed = []
        for ev in due:
            self.inflight.remove(ev)
            j = ev["j"]
            if j not in self.alive:
                # target already destroyed (earlier shot) or leaked:
                # invalid engagement, ammo not returned (spec §4)
                self.invalid_shots += 1
                ev["outcome"] = "invalid"
                continue
            u = self.rng.random_sample()
            if u < ev["p_shot"]:
                self.alive.discard(j)
                self.destroyed_at[j] = t
                destroyed.append(j)
                ev["outcome"] = "kill"
            else:
                ev["outcome"] = "miss"
        return destroyed

    def _settle_breakthrough(self, t: int):
        """Alive targets with r_j(t) <= 0 leak their full weight."""
        leaked = []
        for j in sorted(self.alive):
            if self.dn.r(j, t) <= 0.0:
                self.alive.discard(j)
                self.leaked_at[j] = (t, "breakthrough")
                leaked.append(j)
        return leaked

    # ------------------------------------------------------------------
    # expected-cost bookkeeping (metric ii / iii, analytic - no sampling)
    # ------------------------------------------------------------------

    def _step_costs(self, t: int, actions):
        """Analytic per-step costs of the shots fired at step t.

        cost_shot : spec §8.1 (ii) - sum over ENGAGED targets only,
                    w_j * prod(1 - p_shot) for this step's shots;
        cost_all  : expected surviving value over ALL visible targets
                    (unengaged targets keep full w_j) - same scope as the
                    per-step CPLEX objective, used for gap (iii);
        expected_kill : expected destroyed value of this step's shots.
        """
        dn = self.dn
        on_target = {}
        for i, j in actions.items():
            if j is None:
                continue
            on_target.setdefault(j, []).append(dn.p_eff(i, j, t))
        cost_shot = cost_all = expected_kill = 0.0
        for j in sorted(self.alive):
            ps = on_target.get(j)
            if ps:
                surv = 1.0
                for p in ps:
                    surv *= (1.0 - p)
                cost_all += dn.w[j] * surv
                cost_shot += dn.w[j] * surv
                expected_kill += dn.w[j] * (1.0 - surv)
            else:
                cost_all += dn.w[j]
        return cost_shot, cost_all, expected_kill

    # ------------------------------------------------------------------
    # full-episode run
    # ------------------------------------------------------------------

    def run(self, policy, log=None):
        """Run the whole episode under `policy` (dn_policies interface).

        policy.act(env, t) -> (actions, info) with actions = {i: j or None}
        and info = dict(solved, failed_agents, wall_time, detail...).
        Returns the per-run record (JSON-serialisable).
        """
        dn = self.dn
        steps_rec = []
        shots_per_step = {}
        destroyed_total_value = 0

        for t in range(dn.K + 1):
            # 1) arrivals ------------------------------------------------
            new_ids = []
            if t < dn.K:
                new_ids = dn.targets_arriving(t)
                for j in new_ids:
                    self.appeared.add(j)
                    self.alive.add(j)

            # 2) delayed settlement of inflight interceptors -------------
            destroyed = self._settle_inflight(t)
            destroyed_value = sum(dn.w[j] for j in destroyed)
            destroyed_total_value += destroyed_value

            # 3) breakthrough --------------------------------------------
            # (t < K only: at t = K everything alive goes to the end
            #  settlement below - dataset spec §5 期末 accounting)
            breakthrough = (self._settle_breakthrough(t) if t < dn.K else [])
            breakthrough_value = sum(dn.w[j] for j in breakthrough)

            # 4) decision window (t <= K-2) ------------------------------
            actions, info = {}, {"solved": True, "failed_agents": 0,
                                 "wall_time": 0.0}
            n_legal = n_illegal = 0
            if t <= dn.K - 2:
                t0 = time.time()
                actions, info = policy.act(self, t)
                info["wall_time"] = time.time() - t0
                # legality check per agent action (failed convention:
                # illegal action -> hold fire, that agent's slot failed)
                for i in range(dn.m):
                    j = actions.get(i)
                    if j is None:
                        n_legal += 1  # explicit hold is a legal action
                    elif self.can_fire(i, j, t):
                        n_legal += 1
                    else:
                        n_illegal += 1
                        actions[i] = None
                fired = {i: j for i, j in actions.items() if j is not None}
                for i in sorted(fired):
                    self.fire(i, fired[i], t)

            cost_shot, cost_all, expected_kill = self._step_costs(
                t, {i: j for i, j in actions.items()})

            rec = {
                "t": t,
                "decision_step": bool(t <= dn.K - 2),
                "new": new_ids,
                "n_visible": len(self.alive),
                "destroyed": destroyed,
                "destroyed_value": destroyed_value,
                "breakthrough": breakthrough,
                "breakthrough_value": breakthrough_value,
                "shots": sum(1 for s in self.shots if s["t_fire"] == t),
                "pool_after": self.pool,
                "cost_shot": cost_shot if t <= dn.K - 2 else None,
                "cost_all": cost_all if t <= dn.K - 2 else None,
                "expected_kill": expected_kill if t <= dn.K - 2 else None,
                "reference_cost": (info.get("reference_cost")
                                   if info.get("reference_cost") is not None
                                   else info.get("objective")),
                "solved": info.get("solved", True),
                "failed_agents": info.get("failed_agents", 0),
                "wall_time": info.get("wall_time", 0.0),
                "legal_actions": n_legal,
                "illegal_actions": n_illegal,
                "detail": info.get("detail"),
            }
            steps_rec.append(rec)

            if log is not None:
                log("  t=%2d | new %d | vis %2d | shots %d | kills %2d | "
                    "bt %2d | pool %2d | cost_all %s%s"
                    % (t, len(new_ids), rec["n_visible"], rec["shots"],
                       len(destroyed), len(breakthrough), self.pool,
                       ("%.1f" % cost_all) if rec["cost_all"] is not None
                       else "n/a",
                       "  [WARNING: %s]" % info["detail"]
                       if info.get("detail") else ""))

        # end-of-episode settlement: alive at t=K -> leak (spec §4 v2) ----
        end_leak = sorted(self.alive)
        for j in end_leak:
            self.leaked_at[j] = (dn.K, "end")
        self.alive = set()

        leak_value = sum(dn.w[j] for j in self.leaked_at)
        leak_count_bt = sum(1 for (_, cause) in self.leaked_at.values()
                            if cause == "breakthrough")
        leak_count_end = len(end_leak)

        run_rec = {
            "seed": self.seed,
            "steps": steps_rec,
            "leak_value": leak_value,
            "leak_count": len(self.leaked_at),
            "leak_count_breakthrough": leak_count_bt,
            "leak_count_end": leak_count_end,
            "leak_rate": leak_value / dn.total_value(),
            "destroyed_count": len(self.destroyed_at),
            "destroyed_value": destroyed_total_value,
            "shots_total": len(self.shots),
            "invalid_shots": self.invalid_shots,
            "ammo_end": self.pool,
            "pool_curve": [s["pool_after"] for s in steps_rec],
            "first_engagement_age": dict(self.first_engagement_age),
            "destroyed_ids": sorted(self.destroyed_at),
            "leaked_ids": sorted(self.leaked_at),
        }
        return run_rec


def simulate_dn(dn: DNInstance, seed: int, policy, log=None):
    """Convenience wrapper: one Monte-Carlo replication -> run record."""
    env = DNEnv(dn, seed)
    return env.run(policy, log=log)
