"""Single-wave execution: temp static instance, subprocess solve, solution parsing.

The original solver cplex/wta_cplex.py is NEVER imported or modified here; it is
invoked as a subprocess (branch-and-adjust approach), exactly like a user would.
Solution files are parsed by re-using validator.Solution (imported read-only).
"""

import os
import re
import subprocess
import sys

DWTA_DIR = os.path.dirname(os.path.abspath(__file__))      # .../dwta
PROJECT_ROOT = os.path.dirname(DWTA_DIR)                   # project root
CPLEX_DIR = os.path.join(PROJECT_ROOT, "cplex")            # validator.py lives here
if CPLEX_DIR not in sys.path:
    sys.path.insert(0, CPLEX_DIR)

from validator import Instance as _StaticInstance, Solution as _StaticSolution  # noqa: E402

DEFAULT_PYTHON = os.environ.get("DWTA_PYTHON", "/opt/anaconda3/envs/wta/bin/python")

# CPLEX community-edition problem-size markers (CPLEX Error 1016)
CPLEX_LIMIT_MARKERS = ("Error  1016", "Community Edition")

_RUNTIME_RE = re.compile(r"Total runtime\s*=\s*([0-9.]+)\s*secods")


class CplexLimitError(RuntimeError):
    """Raised when the subprocess reports CPLEX 1016 (community edition limits)."""

    def __init__(self, detail_line: str):
        super().__init__(detail_line)
        self.detail_line = detail_line


def write_wave_instance(path: str, dyn, target_ids, wave_idx=None):
    """Build the per-wave static sub-instance for the given original target ids.

    Local target indices 0..len(target_ids)-1 map back to target_ids.
    wave_idx: when given (current wave k), probabilities are the effective
    ones dyn.effective_p(i, j, k) (distance-scaled, capped); otherwise the
    base dyn.p is written (legacy behaviour, identical bytes for old files).
    """
    m, mu = dyn.m, dyn.mu
    n = len(target_ids)
    out = ["%d %d %d" % (m, n, mu)]
    for j in target_ids:
        out.append(str(dyn.w[j]))
    for i in range(m):
        for local_j, j in enumerate(target_ids):
            if wave_idx is None:
                pij = dyn.p[i, j]
            else:
                pij = dyn.effective_p(i, j, wave_idx)
            out.append("%d %d %.12f" % (i, local_j, pij))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return n


def run_solver(wave_inst: str, sol_path: str, delta: float, timelimit, threads: int,
               python_exe: str = DEFAULT_PYTHON, extra_args=None):
    """Invoke cplex/wta_cplex.py via subprocess; cwd is the project root.

    extra_args: optional list of additional CLI arguments appended verbatim
    (e.g. ["-warmstart", <path>] for the M3 LLM warm-start strategy).
    Returns (returncode, combined_output, wall_time_seconds).
    Raises CplexLimitError when community-edition limits are hit.
    """
    import time

    cmd = [python_exe, os.path.join("cplex", "wta_cplex.py"), os.path.abspath(wave_inst),
           "-delta", repr(float(delta))]
    if timelimit:
        cmd += ["-timelimit", str(int(timelimit))]
    cmd += ["-solution", os.path.abspath(sol_path), "-threads", str(int(threads))]
    if extra_args:
        cmd += [str(a) for a in extra_args]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    output = (proc.stdout or "") + (proc.stderr or "")

    for marker in CPLEX_LIMIT_MARKERS:
        if marker in output:
            detail = next((ln.strip() for ln in output.splitlines() if marker in ln), marker)
            raise CplexLimitError(detail)
    return proc.returncode, output, wall


def parse_wave_solution(wave_inst: str, sol_path: str, target_ids):
    """Parse a wave .sol file.

    Returns dict:
        assignment : {orig_j: {weapon_i: shots}}
        objective  : solver expected cost (float) or None
        runtime    : solver-reported runtime (float) or None
    Returns None if the solution file does not exist (no incumbent in time).
    """
    if not os.path.exists(sol_path):
        return None

    inst = _StaticInstance(wave_inst)
    sol = _StaticSolution(inst, sol_path)

    assignment = {}
    for i, local_targets in enumerate(sol.weapons):
        for local_j in local_targets:
            orig_j = target_ids[local_j]
            assignment.setdefault(orig_j, {})
            assignment[orig_j][i] = assignment[orig_j].get(i, 0) + 1

    runtime = None
    with open(sol_path, "r") as f:
        m = _RUNTIME_RE.search(f.read())
    if m:
        runtime = float(m.group(1))

    return {"assignment": assignment, "objective": sol.objective, "runtime": runtime}
