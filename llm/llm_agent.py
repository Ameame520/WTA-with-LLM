"""LLM-assisted per-wave strategy layer (M3, requirement doc section 4).

Two-phase LLM usage around the untouched per-wave CPLEX solve:

  call 1 (pre-solve)  - modules a/b/d:
      a  suggested_assignments -> validated 'weapon target shots' file passed
         to wta_cplex.py via the new -warmstart CLI flag (CPLEX MIP start;
         infeasible starts are repaired/discarded by CPLEX itself)
      b  solver_params          -> per-wave emphasis / timelimit_multiplier /
                                   branching overrides (whitelisted)
      d  delay_targets          -> low-value targets are removed from this
                                   wave's sub-instance (they stay over)
  call 2 (post-solve) - module c (always on for --policy llm):
      short natural-language explanation of the wave, stored in the wave
      record as rec["llm"]["explanation"] for the E4 report

Robustness contract (doc 4.2 validation rules): every LLM field is validated
against the live state; anything invalid is dropped field by field; if the
call or parsing fails the wave falls back to the plain base solve. LLM calls
never consume the simulator random stream, so the Monte-Carlo replication
stays reproducible for a fixed advice stream.

All calls are appended to logs/llm_calls.jsonl for audit (timestamp, wave,
phase, duration, tokens, raw response, validation outcome).
"""

import json
import os
import time

import requests

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
AUDIT_REL = os.path.join("logs", "llm_calls.jsonl")

_VALID_MODULES = ("a", "b", "c", "d")
# branching is locked to 'cplex': the built-in 'probabilities' branch callback
# asserts on integral values for mu>=3 instances (documented deviation), so any
# LLM request for 'probabilities' is rejected and forced back to 'cplex'.
_LOCKED_BRANCHING = "cplex"


class LLMContext(object):
    """Runtime configuration + audit trail for the LLM strategy layer."""

    def __init__(self, model=None, timeout=60, modules="", log_dir=None):
        self.model = model or os.environ.get("DWTA_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = int(timeout)
        mods = (modules or os.environ.get("DWTA_LLM_MODULES", "a,b,c,d"))
        seen = []
        for m in str(mods).split(","):
            m = m.strip().lower()
            if m in _VALID_MODULES and m not in seen:
                seen.append(m)
        # module c is always active alongside the llm policy (doc 5.1)
        if "c" not in seen:
            seen.append("c")
        self.modules = seen
        self.api_key = (os.environ.get("DEEPSEEK_API_KEY")
                        or os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY (or OPENAI_API_KEY) is not set - refusing to "
                "run the llm policy without credentials (doc section 4.3)")
        self.base_url = (os.environ.get("DEEPSEEK_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.calls = []  # in-memory audit trail (also mirrored to jsonl)
        root = log_dir or os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        self.audit_path = os.path.join(root, AUDIT_REL)
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        self._audit({"phase": "init", "model": self.model,
                     "modules": self.modules, "base_url": self.base_url,
                     "timeout": self.timeout})

    # ------------------------------------------------------------------ api
    def chat(self, messages, wave=None, phase="pre", json_mode=False,
             max_tokens=32768, retries=2):
        """One chat completion; returns response text or None on failure.

        Reasoning models spend tokens on 'reasoning_content' before the final
        'content', so the budget must cover both; if content comes back empty
        (truncated by thinking) the reasoning trace is returned as fallback -
        _extract_json still finds the trailing answer block."""
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                r = requests.post(
                    self.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.api_key,
                             "Content-Type": "application/json"},
                    json=body, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                text = msg.get("content") or msg.get("reasoning_content") or ""
                self._audit({"phase": phase, "wave": wave, "attempt": attempt,
                             "duration_s": round(time.time() - t0, 2),
                             "usage": data.get("usage"),
                             "finish_reason": choice.get("finish_reason"),
                             "empty_content": not msg.get("content"),
                             "response": text[:4000]})
                return text
            except Exception as exc:  # noqa: BLE001 - full fallback contract
                last_err = str(exc)
                self._audit({"phase": phase, "wave": wave, "attempt": attempt,
                             "duration_s": round(time.time() - t0, 2),
                             "error": last_err[:500]})
                time.sleep(1.0)
        return None

    def _audit(self, entry):
        entry = dict(entry, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.calls.append(entry)
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # auditing must never break the policy


def _extract_json(text):
    """Best-effort JSON extraction: direct parse, then the LAST balanced
    {...} block (reasoning traces put the final answer at the end)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    end = text.rfind("}")
    while end >= 0:
        depth = 0
        for start in range(end, -1, -1):
            ch = text[start]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except (ValueError, TypeError):
                        break  # try an earlier closing brace
        end = text.rfind("}", 0, end)
    return None


# --------------------------------------------------------------- validation
def _validate_assignments(raw, state, engaged):
    """-> list of (weapon, target, shots) tuples, or [] (doc 4.2 rules)."""
    if not isinstance(raw, list):
        return []
    dyn = state["dyn"]
    mu, engaged_set = dyn.mu, set(engaged)
    per_weapon, per_target, out = {}, {}, []
    for item in raw:
        try:
            if isinstance(item, dict):
                i, j, v = int(item["weapon"]), int(item["target"]), int(item["shots"])
            else:
                i, j, v = int(item[0]), int(item[1]), int(item[2])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if i not in dyn.W or j not in engaged_set or v <= 0:
            continue  # inactive pair - dropped
        if per_weapon.get(i, 0) + v > mu or per_target.get(j, 0) + v > mu:
            continue  # over capacity - dropped
        per_weapon[i] = per_weapon.get(i, 0) + v
        per_target[j] = per_target.get(j, 0) + v
        out.append((i, j, v))
    return out


def _validate_solver_params(raw, base_solver):
    """-> dict of overrides actually applied (whitelisted, doc 4.2 rules)."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    try:
        emph = int(raw.get("emphasis"))
        if 1 <= emph <= 5:
            out["emphasis"] = emph
    except (TypeError, ValueError):
        pass
    branch = str(raw.get("branching", "")).strip().lower()
    if branch == "probabilities":
        # known assert bug on mu>=3 instances - forced back (documented)
        out["branching_forced"] = ("probabilities rejected (known solver "
                                   "assert bug), kept cplex")
    tl_mult = raw.get("timelimit_multiplier")
    if base_solver.get("timelimit") and isinstance(tl_mult, (int, float)):
        mult = min(2.0, max(0.5, float(tl_mult)))
        if abs(mult - 1.0) > 1e-9:
            out["timelimit"] = max(30, int(round(base_solver["timelimit"] * mult)))
            out["timelimit_multiplier"] = mult
    return out


def _validate_delay(raw, state, target_ids):
    """-> list of delayed target ids (doc 4.2 rules)."""
    if not isinstance(raw, list) or not raw:
        return []
    dyn, alive = state["dyn"], set(target_ids)
    cap = max(0, len(target_ids) - 5)
    out = []
    for j in raw:
        try:
            j = int(j)
        except (TypeError, ValueError):
            continue
        if j not in alive or j in out:
            continue
        if state["ages"][j] >= dyn.L - 1:
            continue  # last stay wave - delaying means breakthrough, forbidden
        if len(out) >= cap:
            continue
        out.append(j)
    return out


def _write_warmstart(path, suggestions, engaged):
    """'weapon local_target shots' lines, local indices as in the wave file."""
    lines = ["%d %d %d" % (i, engaged.index(j), v) for i, j, v in suggestions]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _mipstart_status(output):
    """CPLEX MIP-start acceptance markers from the solver stdout."""
    marks = []
    for ln in (output or "").splitlines():
        low = ln.lower()
        if "mip start" in low or "warmstart" in low:
            marks.append(ln.strip())
    return marks[:3]


def _acceptance_stats(suggestions, assignment):
    """How many suggested pairs the final CPLEX solution actually used."""
    if not suggestions:
        return {"suggested": 0, "used": 0, "used_exact": 0}
    used = exact = 0
    for i, j, v in suggestions:
        if i in assignment.get(j, {}):
            used += 1
            if assignment[j][i] == v:
                exact += 1
    return {"suggested": len(suggestions), "used": used, "used_exact": exact}


# ------------------------------------------------------------------ prompts
def _top_weapons(state, j, k=3):
    dyn, w = state["dyn"], state["wave_idx"]
    ranked = sorted(dyn.W, key=lambda i: -state["p"][(i, j)])
    return ranked[:k]


def _situation_brief(state, mem):
    dyn, k = state["dyn"], state["wave_idx"]
    lines = [
        "Dynamic WTA situation, wave %d of %d." % (k + 1, dyn.K),
        "%d weapons, %d shots each this wave; at most %d waves per target "
        "(L=%d); effective hit rate cap %.2f." % (dyn.m, dyn.mu, dyn.L,
                                                  dyn.L, dyn.pcap),
    ]
    if mem.get("prev_rec") is not None:
        pr = mem["prev_rec"]
        acc = pr.get("llm", {}).get("acceptance", {})
        lines.append(
            "Last wave: destroyed %d targets (value %d), %d stayed over, "
            "%d breakthrough; advice usage %s/%s pairs."
            % (len(pr["destroyed"]), pr["destroyed_value"],
               len(pr["survived"]), len(pr["breakthrough"]),
               acc.get("used", 0), acc.get("suggested", 0)))
    lines.append("Cumulative destroyed value %d, cumulative leak %d."
                 % (mem["cum_destroyed_value"], mem["cum_leak"]))
    lines.append("Active targets this wave (id, value, age, remaining stay "
                 "waves, distance km, best hit rate, top weapons):")
    for j, w in state["alive_targets"]:
        top = _top_weapons(state, j)
        best = state["p"][(top[0], j)]
        lines.append("  target %-4d w=%-5d age=%d stay_left=%d d=%.0fkm "
                     "best_p=%.2f top_weapons=%s"
                     % (j, w, state["ages"][j],
                        dyn.L - 1 - state["ages"][j], state["distances"][j],
                        best, top))
    return "\n".join(lines)


_SYSTEM_PRE = (
    "You are an operations-research assistant for a dynamic weapon-target "
    "assignment problem. Before each wave's exact CPLEX solve you may (a) "
    "suggest assignments as a warm start, (b) tune solver parameters, and "
    "(d) propose delaying low-value targets. Answer with ONE json object "
    "exactly of this schema:\n"
    '{"suggested_assignments": [{"weapon": int, "target": int, "shots": int}],'
    ' "solver_params": {"emphasis": 1-5, "branching": "cplex",'
    ' "timelimit_multiplier": 0.5-2.0}, "delay_targets": [int],'
    ' "rationale": "one short sentence"}\n'
    "Rules (think the wave through carefully - reasoning depth is not\n"
    "rationed, correctness is what matters):\n"
    "1. Score each active target as value * best_p / distance. Suggest one "
    "entry per target for the top 15-20 scored targets, pairing each with its "
    "first top_weapon (shots=1).\n"
    "2. Only use weapon and target ids that appear in the situation; a weapon "
    "may appear at most mu times in total, a target receives at most mu "
    "shots.\n"
    "3. delay_targets: only targets with stay_left>0 scoring in the bottom "
    "quarter, at most (number of active targets - 5) in total; empty list if "
    "unsure.\n"
    "4. solver_params: keep emphasis=3, branching=\"cplex\", "
    "timelimit_multiplier=1.0 unless the last wave clearly hit the runtime "
    "limit.")


def _call_pre_solve(ctx, mem, state):
    user = _situation_brief(state, mem)
    mods = "active modules: %s" % ", ".join(m for m in ctx.modules if m != "c")
    text = ctx.chat(
        [{"role": "system", "content": _SYSTEM_PRE},
         {"role": "user", "content": user + "\n" + mods}],
        wave=state["wave_idx"], phase="pre", json_mode=True)
    advice = _extract_json(text)
    if not isinstance(advice, dict):
        return None
    return advice


_SYSTEM_POST = (
    "You are an operations-research analyst. In simplified Chinese, explain "
    "in 3-5 short sentences what happened in this wave of a dynamic "
    "weapon-target assignment engagement: what the solver did, how the LLM "
    "advice related to it (accepted/rejected), and what the settlement "
    "outcome means for the remaining scenario. Be concrete, no markdown.")


def _call_post_solve(ctx, mem, rec):
    acc = rec.get("llm", {}).get("acceptance", {})
    llm_info = rec.get("llm", {})
    lines = [
        "Wave %d settled." % (rec["wave"] + 1),
        "Final CPLEX solution: %d target(s) engaged, objective %.6g, "
        "solver runtime %s s."
        % (len(rec["assignment"]), rec["objective"] or 0.0,
           rec["solver_runtime"]),
        "LLM advice this wave: %d suggested pairs, %d used by solver, %d with "
        "exact shot count; %d target(s) delayed; warmstart status: %s."
        % (acc.get("suggested", 0), acc.get("used", 0), acc.get("used_exact", 0),
           len(llm_info.get("delayed", [])),
           "; ".join(llm_info.get("mipstart_status", [])) or "not used"),
        "Settlement: %d destroyed (value %d), %d stayed over, %d breakthrough "
        "(leak %d). Cumulative leak %d."
        % (len(rec["destroyed"]), rec["destroyed_value"], len(rec["survived"]),
           len(rec["breakthrough"]), rec["breakthrough_leak"],
           rec["cumulative_leak"]),
    ]
    text = ctx.chat(
        [{"role": "system", "content": _SYSTEM_POST},
         {"role": "user", "content": "\n".join(lines)}],
        wave=rec["wave"], phase="post", json_mode=False, max_tokens=8192)
    return (text or "").strip()


# ------------------------------------------------------------------ policy
def build_policy(ctx):
    """Wrap the base per-wave solve with the two-phase LLM strategy.

    Returns a callable with the simulator decide(state) interface plus an
    on_wave_end(rec, state) hook used for the post-solve module c call.
    """
    mem = {"cum_destroyed_value": 0, "cum_leak": 0, "prev_rec": None}

    def policy(state):
        from dwta import wave_runner  # local import avoids cycles at load time

        dyn, k = state["dyn"], state["wave_idx"]
        target_ids = sorted(j for j, _ in state["alive_targets"])
        llm_rec = {"modules": list(ctx.modules), "suggested": [],
                   "delayed": [], "mipstart_status": [], "acceptance": {},
                   "timelimit_used": state["solver"].get("timelimit"),
                   "advice": None}

        # ---- phase 1: pre-solve LLM call (modules a/b/d) -----------------
        advice = None
        try:
            advice = _call_pre_solve(ctx, mem, state)
        except Exception as exc:  # noqa: BLE001 - degrade to base solve
            llm_rec["error"] = str(exc)[:300]
        llm_rec["advice"] = advice

        engaged, delayed = target_ids, []
        solver = state["solver"]
        extra = list(solver.get("extra_args") or [])
        suggestions = []
        if advice:
            if "d" in ctx.modules:
                delayed = _validate_delay(advice.get("delay_targets"),
                                          state, target_ids)
                if delayed:
                    engaged = [j for j in target_ids if j not in set(delayed)]
                    llm_rec["delayed"] = delayed
            if "b" in ctx.modules:
                over = _validate_solver_params(advice.get("solver_params"),
                                               solver)
                if over:
                    solver = dict(solver)
                    for key in ("emphasis", "timelimit"):
                        if key in over:
                            solver[key] = over[key]
                    llm_rec["param_overrides"] = over
            if "a" in ctx.modules and engaged:
                suggestions = _validate_assignments(
                    advice.get("suggested_assignments"), state, engaged)
                if suggestions:
                    warm_path = state["tmp_inst"] + ".warm"
                    _write_warmstart(warm_path, suggestions, engaged)
                    extra = extra + ["-warmstart", warm_path]
                    llm_rec["suggested"] = ["%d->%d x%d" % (i, j, v)
                                            for i, j, v in suggestions]

        # ---- wave execution (mirrors simulator.decide, on `engaged`) -----
        info = {"objective": None, "solver_runtime": None, "wall_time": None,
                "solved": False, "warning": None}
        if not engaged:
            info["warning"] = ("all active targets delayed - wave not engaged, "
                               "all targets stay over")
            mem["last"] = {"llm": llm_rec, "engaged": engaged}
            return {}, info
        wave_runner.write_wave_instance(state["tmp_inst"], dyn, engaged,
                                        wave_idx=k)
        rc, output, wall = wave_runner.run_solver(
            state["tmp_inst"], state["tmp_sol"],
            delta=solver["delta"], timelimit=solver["timelimit"],
            threads=solver["threads"], python_exe=solver["python"],
            extra_args=extra or None)
        info["wall_time"] = wall
        if suggestions:
            llm_rec["mipstart_status"] = _mipstart_status(output)
        parsed = wave_runner.parse_wave_solution(state["tmp_inst"],
                                                 state["tmp_sol"], engaged)
        if parsed is None:
            info["warning"] = ("no solution file produced (rc=%s) - wave not "
                               "engaged, all targets stay over" % rc)
            mem["last"] = {"llm": llm_rec, "engaged": engaged}
            return {}, info
        info["objective"] = parsed["objective"]
        info["solver_runtime"] = parsed["runtime"]
        info["solved"] = True
        llm_rec["acceptance"] = _acceptance_stats(suggestions,
                                                  parsed["assignment"])
        llm_rec["timelimit_used"] = solver.get("timelimit")
        mem["last"] = {"llm": llm_rec, "engaged": engaged}
        return parsed["assignment"], info

    def on_wave_end(rec, state):
        """Post-settlement hook: update cumulative memory + module c call."""
        mem["cum_destroyed_value"] += rec["destroyed_value"]
        mem["cum_leak"] = rec["cumulative_leak"]
        last = mem.pop("last", None)
        if last:
            rec["llm"] = last["llm"]
        if "c" in ctx.modules:
            try:
                text = _call_post_solve(ctx, mem, rec)
                if text:
                    rec.setdefault("llm", {})["explanation"] = text
            except Exception as exc:  # noqa: BLE001
                rec.setdefault("llm", {})["post_error"] = str(exc)[:300]
        mem["prev_rec"] = rec

    policy.on_wave_end = on_wave_end
    return policy
