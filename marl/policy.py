"""MarlPolicy: dn_policies-interface adapter for the learned actor.

Execution path (evaluation / deployment):
    per step t, per agent i (information boundary §7 only):
        obs_i = env.get_observation(i, t)          # shared + own
        mem_i.update(obs_i, t)                     # M1 lambda recursion
        feats  = build_inputs(obs_i, mem_i, dyn)   # public priors only
        logits = MarlNet(feats)                    # M2+M3
        action = argmax(mask(logits))              # M4, greedy mode
        mem_i.note_own_shot(...)                   # private memory

Red lines honoured here:
    * NEVER touches env.rng - sampling uses an isolated torch.Generator
      (training only; evaluation runs greedy argmax => deterministic
      per-seed result_hash);
    * reset_episode() + automatic t-rewind detection (cross-seed reuse
      safety, conflict #5);
    * device auto = MPS with try/except fallback to CPU.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl.perceive import AgentMemory, build_inputs          # noqa: E402
from marl.network import MarlNet, assert_params              # noqa: E402
from marl.masking import feasible_mask                       # noqa: E402


def _pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MarlPolicy(object):
    name = "marl"
    needs_solver = False

    def __init__(self, model_path: str = None, device: str = "auto",
                 greedy: bool = True, seed: int = 0,
                 training: bool = False, with_reference=False,
                 solver=None, tmp_dir=None):
        self.device = _pick_device(device)
        self.greedy = greedy
        self.training = training
        self.net = MarlNet().to(self.device)
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=self.device,
                              weights_only=True)
            self.net.load_state_dict(ckpt["state_dict"])
        if training:
            self.net.train()
        else:
            self.net.eval()
        self.params_count = assert_params(self.net)
        self._gen = torch.Generator()          # isolated sampling stream
        self._gen.manual_seed(int(seed))
        self.tau = 1.0                          # sampling temperature
        # per-step CPLEX reference (gap metric iii) - same mechanism as
        # GreedyPolicy; the reference solver reads the joint state but its
        # objective is only REPORTED, never fed back into decisions
        self._ref = None
        if with_reference:
            from dwta.dn_policies import CplexPolicy
            self._ref = CplexPolicy(solver, tmp_dir)
        self.reset_episode()

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._mems = None
        self._last_t = -1
        self.last_step_collect = None

    def _ensure_mems(self, m: int):
        if self._mems is None or len(self._mems) != m:
            self._mems = [AgentMemory() for _ in range(m)]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_logits(self, feats):
        """One actor forward on self.device, CPU-fallback on any MPS
        error (returns (logits, on_device_str))."""
        try:
            logits, *_ = self.net(feats["x"].to(self.device),
                                  feats["q"].to(self.device),
                                  feats["g"].to(self.device))
            return logits.cpu()
        except RuntimeError as e:                     # MPS op incompatibility
            if self.device.type != "mps":
                raise
            print("[marl] MPS forward failed (%s) - falling back to CPU"
                  % e)
            self.device = torch.device("cpu")
            self.net.to(self.device)
            logits, *_ = self.net(feats["x"], feats["q"], feats["g"])
            return logits.cpu()

    # ------------------------------------------------------------------
    def act(self, env, t: int):
        dn = env.dn
        m = dn.m
        if t <= self._last_t:
            self.reset_episode()          # rewind -> fresh episode
        self._ensure_mems(m)
        self._last_t = t

        actions = {}
        collect = [] if self.training else None
        for i in range(m):
            obs_i = env.get_observation(i, t)
            mem = self._mems[i]
            mem.update(obs_i, t)
            feats = build_inputs(obs_i, mem, dn)      # public priors only
            logits = self._forward_logits(feats)      # [1+L] hold first
            L = len(feats["alive_ids"])
            if L == 0:
                actions[i] = None
                if collect is not None:
                    collect.append({"agent": i, "empty": True})
                continue
            # mask over obs targets -> align to the alive-only order of
            # alive_ids (feats["x"] rows were built from alive entries)
            full_mask = feasible_mask(obs_i, t, dn)
            ok_by_id = {tr["id"]: ok for tr, ok in
                        zip(obs_i["targets"], full_mask)}
            mask = [ok_by_id[j] for j in feats["alive_ids"]]
            masked = logits.clone()
            for jj, ok in enumerate(mask):
                if not ok:
                    masked[1 + jj] = -float("inf")
            logp_all = torch.log_softmax(masked / self.tau, dim=0)
            if self.greedy or (not any(mask)) or env.pool <= 0:
                pick = int(torch.argmax(masked).item())
            else:
                probs = torch.exp(logp_all)
                # guard: all-zero (fully masked) -> hold
                if not torch.isfinite(probs).all():
                    pick = 0
                else:
                    pick = int(torch.multinomial(
                        probs, 1, generator=self._gen).item())
            actions[i] = None if pick == 0 else feats["alive_ids"][pick - 1]
            if collect is not None:
                collect.append({
                    "agent": i, "empty": False,
                    "x": feats["x"], "q": feats["q"], "g": feats["g"],
                    "mask": torch.tensor(mask, dtype=torch.bool),
                    "pick": pick,
                    "logp": float(logp_all[pick].item()),
                    "tau": self.tau,
                })
        if collect is not None:
            self.last_step_collect = collect

        # private memory: register own shots AFTER deciding all agents
        for i, j in actions.items():
            if j is not None:
                self._mems[i].note_own_shot(j, t)

        info = {"solved": True, "failed_agents": 0}
        if self._ref is not None:
            _, ref_info = self._ref.act(env, t)
            info["reference_cost"] = ref_info.get("objective")
            info["reference_solved"] = ref_info.get("solved", False)
            info["detail"] = ref_info.get("detail")
        return actions, info


# ----------------------------------------------------------------------
# smoke test: 3 seeds on s01 with a random-init checkpoint
# ----------------------------------------------------------------------

def _selftest():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(here, "output", "e14_smoke")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "init.pt")
    net = MarlNet()
    torch.manual_seed(0)
    for p in net.parameters():     # deterministic random init
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    torch.save({"state_dict": net.state_dict(),
                "feature_spec": {"x": 10, "q": 5, "g": 3},
                "params_count": net.params_count()}, ckpt_path)

    from dwta.dn_instance import DNInstance
    from dwta.dn_env import DNEnv
    dn = DNInstance(os.path.join(here, "data", "dn-data-v3",
                                 "dn_3x50_K10_s01.txt"))
    pol = MarlPolicy(ckpt_path, device="auto", greedy=True, seed=0)
    print("device:", pol.device, "| params:", pol.params_count)
    for seed in (42, 43, 44):
        env = DNEnv(dn, seed)
        pol.reset_episode()
        rec = env.run(pol)
        illegal = sum(s["illegal_actions"] for s in rec["steps"])
        wall = max(s["wall_time"] for s in rec["steps"]
                   if s["decision_step"])
        print("seed %d: leak=%.4f illegal=%d max_act_wall=%.4fs"
              % (seed, rec["leak_rate"], illegal, wall))
        assert illegal == 0, "illegal fire detected"
        assert wall < 1.0, "act wall >= 1s"
    print("marl/policy.py selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
