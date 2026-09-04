"""M6: CTDE training loop (MAPPO-style PPO) for the DN-WTA marl policy.

Training-side wrapper collects, per decision step:
    * actor inputs strictly from per-agent §7 observations (red line:
      the execution path never sees global state);
    * critic inputs from GLOBAL truth (env.inflight registry, true
      per-target occupancy column, B(t), t) - allowed here only.

Critic: state+joint-action conditioned V(s, a); counterfactual baselines
V(s, a_{-i}, hold_i) give each agent a COMA-style differential signal:
    A_i(t) = GAE_team(t) + kill_credit((t, i))
             + [V(s, a) - V(s, a_{-i}, hold_i)]

PPO: clip=0.2, entropy 0.01, GAE(lambda=0.95, gamma=0.99), Adam lr 3e-4,
32 episodes per batch (~864 joint samples), 4 epochs per update.

Early stop: val = s27-s30 x seeds 42-51 (leak rate only, no CPLEX),
evaluated every --eval-every iters, patience in EVAL POINTS.

Seed isolation: collection seeds from 100001 upward; eval 42-71; val 42-51.
Sampling temperature anneals 1.0 -> 0.5 over the first --anneal-iters.

Products (flat dir --output):
    best.pt            {state_dict, feature_spec, params_count}
    train_log.jsonl    one line per eval point
    train_summary.json {total_wall_sec, env_steps, params_count, best_val}

CLI (module-level, precedent experiments/*.py):
    python marl/train.py --iters 50 --eval-every 10 --device auto \
        --output output/e14_train_smoke
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dwta.dn_instance import DNInstance                     # noqa: E402
from dwta.dn_env import DNEnv                               # noqa: E402
from marl.policy import MarlPolicy, _pick_device            # noqa: E402
from marl.network import MarlNet, assert_params             # noqa: E402
from marl.reward import build_rewards                       # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dn-data-v3")

# split file lists (fixed by MANIFEST)
TRAIN_INSTS = ["dn_3x50_K10_s%02d.txt" % s for s in range(3, 27)]
VAL_INSTS = ["dn_3x50_K10_s%02d.txt" % s for s in range(27, 31)]

# PPO hyper-parameters (locked by the implementation spec)
PPO_CLIP = 0.2
ENT_COEF = 0.01
GAE_LAMBDA = 0.95
GAMMA = 0.99
LR = 3e-4
EPISODES_PER_ITER = 32
PPO_EPOCHS = 2
MINIBATCH = 256
WALL_LIMIT_SEC = 24 * 3600.0


# ----------------------------------------------------------------------
# critic: global-state + joint-action value (training side ONLY)
# ----------------------------------------------------------------------

class CriticNet(nn.Module):
    """V(s, a): pooled true target features + global row + per-agent
    action summaries -> MLP -> scalar."""

    STATE_DIM = 32 + 3 + 3 * 3     # pooled targets + global + 3 agents x 3

    def __init__(self):
        super().__init__()
        self.tgt_proj = nn.Linear(5, 32)
        self.head = nn.Sequential(nn.Linear(self.STATE_DIM, 128), nn.Tanh(),
                                  nn.Linear(128, 1))

    def forward(self, tgt, glob, act):
        """tgt: [B, L, 5] (padded, mask via zeros), glob: [B, 3],
        act: [B, 9]. Returns [B] values."""
        e = torch.tanh(self.tgt_proj(tgt))            # [B, L, 32]
        pooled = e.mean(dim=1)                        # [B, 32] (0-safe)
        z = torch.cat([pooled, glob, act], dim=-1)
        return self.head(z).squeeze(-1)


def _ceil_int(x, eps=1e-9):
    return int(math.ceil(x - eps))


# ----------------------------------------------------------------------
# training-side collectors (global truth - NOT part of the actor path)
# ----------------------------------------------------------------------

def critic_inputs(env, t, actions, dn):
    """Build the critic input tensors for one step (before fire)."""
    total = float(dn.total_value())
    pool0 = float(dn.m * dn.mu)
    rows = []
    for j in sorted(env.alive):
        occ = sum(1 for ev in env.inflight if ev["j"] == j)
        p_surv = 1.0
        for ev in env.inflight:
            if ev["j"] == j:
                p_surv *= (1.0 - ev["p_shot"])
        pbar = 1.0 - p_surv
        r = dn.r(j, t)
        rows.append([dn.w[j] / total,
                     r / max(1.0, float(dn.r0[j])),
                     min(1.0, occ / 6.0),
                     pbar,
                     occ * 0.0 + _ceil_int(r / dn.delta_d - 1e-9) / 10.0])
    if not rows:
        rows = [[0.0] * 5]
    tgt = torch.tensor(rows, dtype=torch.float32)
    glob = torch.tensor([env.pool / pool0, t / float(dn.K),
                         len(env.alive) / float(dn.n)], dtype=torch.float32)
    act_rows = []
    for i in range(dn.m):
        j = actions.get(i)
        if j is None:
            act_rows += [1.0, 0.0, 0.0]
        else:
            act_rows += [0.0, dn.w[j] / total, dn.p_eff(i, j, t)]
    act = torch.tensor(act_rows, dtype=torch.float32)
    return tgt, glob, act


# ----------------------------------------------------------------------
# main trainer
# ----------------------------------------------------------------------

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = _pick_device(args.device)
        self.actor = MarlNet().to(self.device)
        self.n_params = assert_params(self.actor)
        self.critic = CriticNet().to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=LR)
        # evaluation policy (greedy, shares the actor module)
        self.eval_pol = MarlPolicy(model_path=None, device=args.device,
                                   greedy=True, seed=0)
        self.eval_pol.net = self.actor
        self.collector = MarlPolicy(model_path=None, device=args.device,
                                    greedy=False, seed=args.seed,
                                    training=True)
        self.collector.net = self.actor
        self.train_dns = [DNInstance(os.path.join(DATA_DIR, f))
                          for f in TRAIN_INSTS]
        self.val_dns = [DNInstance(os.path.join(DATA_DIR, f))
                        for f in VAL_INSTS]
        self.seed_counter = 100001
        self.env_steps = 0
        self.best_val = float("inf")
        self.log_path = os.path.join(args.output, "train_log.jsonl")
        self._log_f = open(self.log_path, "w")

    # ------------------------------------------------------------------
    def _anneal_tau(self, it):
        span = max(1, self.args.anneal_iters)
        self.collector.tau = max(0.5, 1.0 - 0.5 * it / span)

    # ------------------------------------------------------------------
    def collect_episode(self, dn):
        """Run one episode with the training collector; return the
        per-step sample list and the episode record."""
        env = DNEnv(dn, self.seed_counter)
        self.seed_counter += 1
        samples = []          # per decision step: {t, agents:[...],
                              # critic in (tensors), actions}
        orig_act = self.collector.act

        def wrapped_act(e, t):
            # NOTE: called by env.run BEFORE fires; we snapshot global
            # state pre-action for the critic, then act
            actions, info = orig_act(e, t)
            tgt, glob, act = critic_inputs(e, t, actions, dn)
            samples.append({
                "t": t,
                "agents": self.collector.last_step_collect,
                "tgt": tgt, "glob": glob, "act": act,
                "actions": dict(actions),
            })
            self.env_steps += 1
            return actions, info

        run_rec = env.run(type("W", (), {"act": staticmethod(wrapped_act)})())
        rew = build_rewards(env, run_rec, dn)
        return samples, rew, run_rec

    # ------------------------------------------------------------------
    def _gae(self, rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
        T = len(rewards)
        adv = [0.0] * T
        lastgaelam = 0.0
        for t in reversed(range(T)):
            next_v = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + gamma * next_v - values[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        return adv

    # ------------------------------------------------------------------
    def process_batch(self, episodes):
        """Turn collected episodes into flat PPO samples."""
        flat = []
        for (samples, rew, run_rec) in episodes:
            K = len(rew["R_shaped"]) - 1               # t = 0..K
            dec_steps = [s for s in samples if s["t"] <= K - 2]
            if not dec_steps:
                continue
            # decision-step reward series; fold tail (t=K-1, K) into the
            # last decision step
            r_series = [float(rew["R_shaped"][s["t"]]) for s in dec_steps]
            r_series[-1] += float(rew["R_shaped"][K - 1]) \
                if K - 1 > dec_steps[-1]["t"] else 0.0
            r_series[-1] += float(rew["R_shaped"][K])
            with torch.no_grad():
                vals = []
                for s in dec_steps:
                    try:
                        v = self.critic(
                            s["tgt"].unsqueeze(0).to(self.device),
                            s["glob"].unsqueeze(0).to(self.device),
                            s["act"].unsqueeze(0).to(self.device)).item()
                    except RuntimeError as e:
                        if self.device.type != "mps":
                            raise
                        print("[marl-train] MPS critic failed (%s) -> CPU"
                              % e)
                        self.device = torch.device("cpu")
                        self.critic.to(self.device)
                        self.actor.to(self.device)
                        v = self.critic(s["tgt"].unsqueeze(0),
                                        s["glob"].unsqueeze(0),
                                        s["act"].unsqueeze(0)).item()
                    vals.append(v)
            gae = self._gae(r_series, vals)
            credit = rew["credit"]
            for idx, s in enumerate(dec_steps):
                t = s["t"]
                # counterfactual differential per agent
                cf = {}
                with torch.no_grad():
                    base_v = vals[idx]
                    for i in range(len(s["agents"])):
                        act_cf = s["act"].clone().view(-1)
                        act_cf[i * 3:i * 3 + 3] = torch.tensor(
                            [1.0, 0.0, 0.0])
                        try:
                            v = self.critic(
                                s["tgt"].unsqueeze(0).to(self.device),
                                s["glob"].unsqueeze(0).to(self.device),
                                act_cf.view(1, -1).to(self.device)
                            ).item()
                        except RuntimeError:
                            v = self.critic(
                                s["tgt"].unsqueeze(0),
                                s["glob"].unsqueeze(0),
                                act_cf.view(1, -1)).item()
                        cf[i] = base_v - v
                for entry in s["agents"]:
                    i = entry["agent"]
                    if entry.get("empty"):
                        continue
                    adv_i = gae[idx] + credit.get((t, i), 0.0) + cf[i]
                    flat.append({
                        "x": entry["x"], "q": entry["q"], "g": entry["g"],
                        "mask": entry["mask"], "pick": entry["pick"],
                        "logp_old": entry["logp"],
                        "adv": adv_i,
                        "ret": gae[idx] + vals[idx],
                        "tgt": s["tgt"], "glob": s["glob"], "act": s["act"],
                    })
        return flat

    # ------------------------------------------------------------------
    def ppo_update(self, flat):
        if not flat:
            return 0.0, 0.0, 0.0
        advs = torch.tensor([f["adv"] for f in flat], dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        rets = torch.tensor([f["ret"] for f in flat], dtype=torch.float32)
        n = len(flat)
        idx_all = torch.randperm(n)
        stats = [0.0, 0.0, 0.0]
        nb = 0
        for _epoch in range(PPO_EPOCHS):
            for start in range(0, n, MINIBATCH):
                idx = idx_all[start:start + MINIBATCH]
                pol_loss = 0.0
                self.opt_a.zero_grad()
                for k in idx.tolist():
                    f = flat[k]
                    logits, *_ = self.actor(f["x"].to(self.device),
                                            f["q"].to(self.device),
                                            f["g"].to(self.device))
                    masked = logits.clone()
                    masked[1:][~f["mask"]] = -float("inf")
                    # PPO ratio must compare SAME distribution: replay with
                    # the temperature used at collection time
                    logp_all = torch.log_softmax(
                        masked / f.get("tau", self.collector.tau), dim=0)
                    logp_new = logp_all[f["pick"]]
                    ratio = torch.exp(logp_new - f["logp_old"])
                    a = advs[k]
                    surr = torch.min(ratio * a,
                                     torch.clamp(ratio, 1 - PPO_CLIP,
                                                 1 + PPO_CLIP) * a)
                    fin = torch.isfinite(logp_all)
                    ent = -(torch.exp(logp_all[fin]) * logp_all[fin]).sum()
                    pol_loss = pol_loss - surr - ENT_COEF * ent
                pol_loss = pol_loss / len(idx)
                pol_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               0.5)
                self.opt_a.step()
                stats[0] += float(pol_loss.item())
                nb += 1
            # ---- critic regression (MSE to returns, grad-accumulated) --
            self.opt_c.zero_grad()
            v_loss = 0.0
            for k in idx_all.tolist():
                f = flat[k]
                v = self.critic(f["tgt"].unsqueeze(0).to(self.device),
                                f["glob"].unsqueeze(0).to(self.device),
                                f["act"].unsqueeze(0).to(self.device))
                loss = (v[0] - rets[k]) ** 2 / n
                loss.backward()
                v_loss += float(loss.item()) * n
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                           0.5)
            self.opt_c.step()
            stats[1] += v_loss / n
        return (stats[0] / max(1, nb), stats[1] / max(1, PPO_EPOCHS),
                stats[2])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_val(self):
        """val split x seeds 42-51, greedy argmax, leak rate only."""
        rates = []
        for dn in self.val_dns:
            for seed in range(42, 52):
                env = DNEnv(dn, seed)
                self.eval_pol.reset_episode()
                rec = env.run(self.eval_pol)
                rates.append(rec["leak_rate"])
        mean = sum(rates) / len(rates)
        std = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
        return mean, std

    # ------------------------------------------------------------------
    def save_ckpt(self, path):
        torch.save({
            "state_dict": self.actor.state_dict(),
            "feature_spec": {"x": 10, "q": 5, "g": 3},
            "params_count": self.n_params,
        }, path)

    # ------------------------------------------------------------------
    def run(self):
        args = self.args
        t_start = time.time()
        ckpt_path = os.path.join(args.output, "best.pt")
        it = 0
        eval_points = 0
        bad_points = 0
        stopped = None
        while it < args.iters:
            if time.time() - t_start > WALL_LIMIT_SEC:
                stopped = "wall_limit_24h"
                break
            # ---- collect one batch ----------------------------------
            episodes = []
            train_leaks = []
            for _ in range(EPISODES_PER_ITER):
                dn = self.train_dns[
                    (it * EPISODES_PER_ITER + len(episodes))
                    % len(self.train_dns)]
                samples, rew, run_rec = self.collect_episode(dn)
                episodes.append((samples, rew, run_rec))
                train_leaks.append(run_rec["leak_rate"])
            self._anneal_tau(it)
            flat = self.process_batch(episodes)
            pl, vl, el = self.ppo_update(flat)
            it += 1
            # ---- periodic evaluation -------------------------------
            if it % args.eval_every == 0 or it == args.iters:
                val_mean, val_std = self.evaluate_val()
                eval_points += 1
                row = {
                    "iter": it,
                    "env_steps": self.env_steps,
                    "train_leak": sum(train_leaks) / len(train_leaks),
                    "val_leak_mean": val_mean,
                    "val_leak_std": val_std,
                    "wall_sec": round(time.time() - t_start, 1),
                    "tau": round(self.collector.tau, 3),
                    "policy_loss": round(pl, 6),
                }
                self._log_f.write(json.dumps(row) + "\n")
                self._log_f.flush()
                print("[iter %6d] train %.4f | val %.4f+-%.4f | "
                      "tau %.2f | %.0fs"
                      % (it, row["train_leak"], val_mean, val_std,
                         self.collector.tau, row["wall_sec"]), flush=True)
                if not all(math.isfinite(v) for v in
                           (val_mean, row["train_leak"])):
                    stopped = stopped or "nan_guard"
                    break
                if val_mean < self.best_val - 1e-6:
                    self.best_val = val_mean
                    self.save_ckpt(ckpt_path)
                    bad_points = 0
                else:
                    bad_points += 1
                    if bad_points >= args.patience:
                        stopped = "early_stop"
                        break
        # ---- summary -------------------------------------------------
        wall = time.time() - t_start
        summary = {
            "total_wall_sec": round(wall, 1),
            "env_steps": self.env_steps,
            "params_count": self.n_params,
            "best_val": (None if self.best_val == float("inf")
                         else self.best_val),
            "final_metrics": {"iters_done": it, "eval_points": eval_points,
                              "stop_reason": stopped or "iters_done",
                              "device": str(self.device)},
        }
        with open(os.path.join(args.output, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self._log_f.close()
        print("training done: %s | best val %.4f | wall %.0fs"
              % (summary["final_metrics"]["stop_reason"],
                 self.best_val if self.best_val != float("inf") else -1,
                 wall))


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="CTDE training for marl")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=20,
                    help="early-stop patience in EVAL POINTS")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0,
                    help="actor sampling generator seed")
    ap.add_argument("--anneal-iters", type=int, default=2000,
                    help="iters to anneal tau 1.0 -> 0.5")
    ap.add_argument("--output", default=os.path.join(
        here, "..", "output", "e14_marl_train"))
    args = ap.parse_args(argv)
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)
    tr = Trainer(args)
    print("device=%s actor_params=%d" % (tr.device, tr.n_params))
    tr.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
