"""Baseline decision policies for the DN-WTA v2 environment.

All policies implement the same interface consumed by DNEnv.run:

    policy.act(env, t) -> (actions, info)
        actions : {agent_i: target_j or None}   (<= 1 shot per agent/step)
        info    : dict(solved, failed_agents, wall_time, objective,
                       detail) - objective is the pool-feasible centralised
                       myopic CPLEX optimum on the same state (metric iii
                       gap reference; re-scored after global-pool trimming,
                       so the cplex policy's own gap is 0 by definition).

Policies:

  NonePolicy   ('none')   : never fire - sanity lower bound, reproduces the
                            no-defense leak histogram of the dataset spec;
  GreedyPolicy ('greedy') : DISTRIBUTED local greedy. Each agent uses only
                            get_observation(i, t) (spec §7): score =
                            w_j * p_eff_ij(t) / remaining_windows, filtered
                            to targets the interceptor can still reach before
                            breakthrough (t_hit <= breakthrough step). No
                            communication -> duplicate engagements possible
                            (shows up in metric xvii);
  CplexPolicy  ('cplex')  : CENTRALISED myopic optimum. Full joint state
                            (violates §7 on purpose - it is the reference
                            upper bound): builds the per-step static
                            sub-instance (mu = 1 per platform, effective
                            probabilities) and calls the untouched
                            cplex/wta_cplex.py via subprocess, exactly like
                            the legacy wave_runner. Global-pool aware
                            marginal-value trimming when the pool cannot
                            cover the solver's plan.

The CPLEX subprocess machinery (run_solver / parse_wave_solution /
CplexLimitError) is re-used as-is from dwta/wave_runner.py; only the
sub-instance writer is DN-specific (effective p at time t).
"""

import math
import os

from . import wave_runner


# ----------------------------------------------------------------------
# policy 1: no defense
# ----------------------------------------------------------------------

class NonePolicy(object):
    name = "none"
    needs_solver = False

    def act(self, env, t):
        return {i: None for i in range(env.dn.m)}, {
            "solved": True, "failed_agents": 0}


# ----------------------------------------------------------------------
# policy 2: distributed local greedy (observation-only)
# ----------------------------------------------------------------------

class GreedyPolicy(object):
    name = "greedy"
    needs_solver = False

    def __init__(self, with_reference=False, solver=None, tmp_dir=None):
        """with_reference: additionally solve the per-step CPLEX optimum on
        the SAME state to compute the gap (iii) of this policy's decisions.
        Needs solver config dict + tmp_dir (see main.py)."""
        self.with_reference = with_reference
        self._solver = solver
        self._tmp_dir = tmp_dir
        self._ref = CplexPolicy(solver, tmp_dir) if with_reference else None

    @staticmethod
    def _obs_lookup(obs, j, key):
        return next(tr[key] for tr in obs["targets"] if tr["id"] == j)

    def _score(self, obs, i, j, t):
        """w_j * p_eff_ij(t) / windows_left - observation-only (spec §7).

        p_eff needs d0_ij which is NOT in the observation; the agent
        reconstructs it from the public closing dynamics:
        d0_ij = d_ij(t) + delta_d * age_j(t). windows_left and the flight
        time h derive from the shared r_j(t) / own d_ij(t) respectively.
        """
        pub, own = obs["public"], obs["own"]
        d = own["d"][j]
        age = self._obs_lookup(obs, j, "age")
        d0 = d + pub["delta_d"] * age
        p_eff = min(pub["pcap"], own["p"][j] * d0 / max(d, 1e-9))
        r = self._obs_lookup(obs, j, "r")
        windows_left = max(1, int(math.ceil(r / pub["delta_d"] - 1e-9)))
        w = self._obs_lookup(obs, j, "w")
        return w * p_eff / windows_left

    def act(self, env, t):
        dn = env.dn
        actions = {}
        if env.pool <= 0 or t > dn.K - 2:
            return {i: None for i in range(dn.m)}, {
                "solved": True, "failed_agents": 0}

        for i in range(dn.m):
            obs = env.get_observation(i, t)
            pub, own = obs["public"], obs["own"]
            best_j, best_s = None, 0.0
            for tr in obs["targets"]:
                if not tr["alive"]:
                    continue
                j, r = tr["id"], tr["r"]
                windows_left = math.ceil(r / pub["delta_d"] - 1e-9)
                if windows_left < 1:
                    continue  # cannot be engaged any more
                # own interceptor must arrive before/at breakthrough
                d = own["d"][j]
                h = max(1, int(math.ceil(d / pub["v_m"] / pub["dt"] - 1e-9)))
                if t + h > t + windows_left:
                    continue
                s = self._score(obs, i, j, t)
                if s > best_s:
                    best_j, best_s = j, s
            actions[i] = best_j

        info = {"solved": True, "failed_agents": 0}
        if self._ref is not None:
            _, ref_info = self._ref.act(env, t)
            info["reference_cost"] = ref_info.get("objective")
            info["reference_solved"] = ref_info.get("solved", False)
            info["detail"] = ref_info.get("detail")
        return actions, info


# ----------------------------------------------------------------------
# policy 3: centralised myopic CPLEX (reference upper bound)
# ----------------------------------------------------------------------

class CplexPolicy(object):
    name = "cplex"
    needs_solver = True

    def __init__(self, solver=None, tmp_dir=None):
        """solver: dict(delta, timelimit, threads, python[, extra_args]).
        tmp_dir: directory for the per-step temp instance / solution files."""
        self.solver = solver or {"delta": 0.001, "timelimit": 60,
                                 "threads": 1,
                                 "python": wave_runner.DEFAULT_PYTHON}
        self.tmp_dir = tmp_dir or os.path.join("output", "tmp")

    def _write_step_instance(self, path, env, t):
        """Static sub-instance for the CURRENT joint state (mu = 1):

            m n 1 / w_j lines / p_eff(i, j, t) lines (local target ids)
        """
        dn = env.dn
        target_ids = env.visible_targets(t)
        out = ["%d %d %d" % (dn.m, len(target_ids), 1)]
        for j in target_ids:
            out.append(str(dn.w[j]))
        for i in range(dn.m):
            for local_j, j in enumerate(target_ids):
                out.append("%d %d %.12f" % (i, local_j, dn.p_eff(i, j, t)))
        with open(path, "w") as f:
            f.write("\n".join(out) + "\n")
        return target_ids

    def _solve(self, env, t):
        """One subprocess solve on the current state.

        Returns (assignment {i: j}, objective, solved, detail) where
        objective = optimal expected surviving value over visible targets
        (metric iii reference), assignment possibly {} when unsolved.
        """
        os.makedirs(self.tmp_dir, exist_ok=True)
        inst = os.path.join(self.tmp_dir, "dn_t%d_inst.txt" % t)
        sol = os.path.join(self.tmp_dir, "dn_t%d.sol" % t)
        if os.path.exists(sol):
            os.remove(sol)
        target_ids = self._write_step_instance(inst, env, t)

        rc, output, _wall = wave_runner.run_solver(
            inst, sol,
            delta=self.solver["delta"], timelimit=self.solver["timelimit"],
            threads=self.solver["threads"], python_exe=self.solver["python"],
            extra_args=self.solver.get("extra_args"))

        parsed = wave_runner.parse_wave_solution(inst, sol, target_ids)
        if parsed is None:
            return {}, None, False, ("no solution file (rc=%s) - hold fire "
                                     "this step" % rc)
        assignment = {}
        for j, per_i in parsed["assignment"].items():
            for i in per_i:
                assignment[i] = j  # mu = 1 -> at most one shot per (i, j)
        return assignment, parsed["objective"], True, None

    @staticmethod
    def _expected_surviving(env, t, assignment):
        """Expected surviving value over ALL visible targets under
        `assignment` ({i: j}) - same formula and inputs as env cost_all,
        so the reported objective matches what the environment scores
        (cplex policy gap == 0 by definition)."""
        dn = env.dn
        per_target = {}
        for i, j in assignment.items():
            per_target.setdefault(j, []).append(i)
        cost = 0.0
        for j in env.visible_targets(t):
            shooters = per_target.get(j)
            if shooters:
                surv = 1.0
                for i in shooters:
                    surv *= (1.0 - dn.p_eff(i, j, t))
                cost += dn.w[j] * surv
            else:
                cost += dn.w[j]
        return cost

    def act(self, env, t):
        dn = env.dn
        no_action = {i: None for i in range(dn.m)}
        if t > dn.K - 2:
            return no_action, {"solved": True, "failed_agents": 0}

        assignment, objective, solved, detail = self._solve(env, t)
        info = {"solved": solved, "detail": detail, "objective": objective}
        if not solved:
            info["failed_agents"] = dn.m
            return no_action, info
        info["failed_agents"] = 0

        # global-pool trimming: keep the best-shots the pool can still
        # cover, ranked by marginal expected destroyed value; re-score the
        # surviving plan so `objective` stays the pool-feasible optimum
        # (same resource budget any policy faces at this state)
        if len(assignment) > env.pool:
            per_target = {}
            for i, j in sorted(assignment.items()):
                per_target.setdefault(j, []).append(i)
            marginal = []
            for j, shooters in per_target.items():
                surv = 1.0
                for i in shooters:  # ascending platform id: deterministic
                    p = dn.p_eff(i, j, t)
                    marginal.append((dn.w[j] * surv * p, i, j))
                    surv *= (1.0 - p)
            marginal.sort(key=lambda x: (-x[0], x[1], x[2]))
            keep = set(i for _, i, _ in marginal[:max(env.pool, 0)])
            assignment = {i: j for i, j in assignment.items() if i in keep}
            objective = self._expected_surviving(env, t, assignment)
            info["objective"] = objective

        actions = {i: None for i in range(dn.m)}
        for i, j in assignment.items():
            actions[i] = j
        return actions, info


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------

def build_policy(name, solver=None, tmp_dir=None, with_reference=False):
    if name == "none":
        return NonePolicy()
    if name == "greedy":
        return GreedyPolicy(with_reference=with_reference, solver=solver,
                            tmp_dir=tmp_dir)
    if name == "cplex":
        return CplexPolicy(solver, tmp_dir)
    raise ValueError("unknown DN policy: %r (expected none/greedy/cplex)"
                     % name)
