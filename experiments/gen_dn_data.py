"""CLI: generate DN-WTA v1 dataset instances (spec: 数据集规则.md).

Grid mode (default MARL comparison grid):

    python experiments/gen_dn_data.py --grid
    -> m in {3,5,10,20} x n in {10,20,50,100} x seeds {1,2} in data/

Single mode:

    python experiments/gen_dn_data.py --single --m 3 --n 10 --seed 7

Generation rules (all reproducible from --seed):
    k_arr   balanced split over K steps, shuffled (RandomState)
    w, p    sampled from a static instance pool (default data/50x100x1.txt,
            w ~ U{1..99}, p ~ U(0,1) as in the original static dataset);
            synthetic fallback with the same distributions if the pool is
            missing or too small
    r0      ~ U{r0_min..r0_max} km (default 5..15)
    d0_ij   = r0_j + b_i + eps_ij  with b_i ~ U{0..b_max}, eps_ij ~ U{0..eps_max}
            (guarantees d0_ij >= r0_j; b/eps are generation-only, not stored)
    mu      = max(1, min(K, ceil(ammo_ratio * n / m))), ammo_ratio default 0.3
            (total episode ammo per platform; roughly balances the ~19-25% of
            targets that can actually reach the boundary within K steps)
    K       10 for n <= 50, 15 for n = 100 (override with --K)

Verified invariants (self-check after each write):
    - re-parses cleanly through dwta.dn_instance.DNInstance (format + ranges)
    - exactly 1 + n + 2*m*n lines, d0_ij >= r0_j, k_arr in [0, K), p in (0,1)
"""

import argparse
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dwta.instance import parse_static  # noqa: E402
from dwta.dn_instance import DNInstance, write_dn  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_POOL = os.path.join(DATA_DIR, "50x100x1.txt")

GRID_M = (3, 5, 10, 20)
GRID_N = (10, 20, 50, 100)


def default_K(n: int) -> int:
    return 10 if n <= 50 else 15


def balanced_k_arr(n: int, K: int, rng) -> list:
    """Balanced arrival split over K steps, ids shuffled by rng."""
    base, rem = divmod(n, K)
    sizes = [base + (1 if k < rem else 0) for k in range(K)]
    order = rng.permutation(n).tolist()
    k_arr = [0] * n
    pos = 0
    for k, size in enumerate(sizes):
        for _ in range(size):
            k_arr[order[pos]] = k
            pos += 1
    return k_arr


def compute_mu(n: int, m: int, K: int, ammo_ratio: float) -> int:
    return max(1, min(K, int(math.ceil(ammo_ratio * n / m))))


def sample_wp(m: int, n: int, rng, pool_path: str):
    """Sample (w, p) from the static pool; synthetic fallback if unavailable.

    Returns (w, p, source_tag).
    """
    pool = None
    if pool_path and os.path.exists(pool_path):
        pool = parse_static(pool_path)
        if pool["m"] < m or pool["n"] < n:
            pool = None
    if pool is not None:
        rows = rng.choice(pool["m"], size=m, replace=False)
        cols = rng.choice(pool["n"], size=n, replace=False)
        w = [int(pool["w"][c]) for c in cols]
        p = {(i, j): float(pool["p"][rows[i], cols[j]])
             for i in range(m) for j in range(n)}
        tag = "pool %s (rows %d/%d, cols %d/%d)" % (
            os.path.basename(pool_path), m, pool["m"], n, pool["n"])
    else:
        w = rng.randint(1, 100, size=n).tolist()
        p = {(i, j): float(rng.uniform(0.0001, 0.9999))
             for i in range(m) for j in range(n)}
        tag = "synthetic (w~U{1..99}, p~U(0,1))"
    return w, p, tag


def make_instance(m, n, seed, args) -> str:
    K = args.K if args.K else default_K(n)
    mu = compute_mu(n, m, K, args.ammo_ratio)

    rs_split = np.random.RandomState(seed * 100 + 1)
    rs_r0 = np.random.RandomState(seed * 100 + 2)
    rs_b = np.random.RandomState(seed * 100 + 3)
    rs_eps = np.random.RandomState(seed * 100 + 4)
    rs_wp = np.random.RandomState(seed * 100 + 5)

    k_arr = balanced_k_arr(n, K, rs_split)
    r0 = rs_r0.randint(args.r0_min, args.r0_max + 1, size=n).tolist()
    b = rs_b.randint(0, args.b_max + 1, size=m)
    eps = rs_eps.randint(0, args.eps_max + 1, size=(m, n))
    d0 = {(i, j): int(r0[j] + b[i] + eps[i, j]) for i in range(m) for j in range(n)}

    w, p, wp_tag = sample_wp(m, n, rs_wp, args.pool)

    name = "dn_%dx%d_K%d_s%d.txt" % (m, n, K, seed)
    path = os.path.join(DATA_DIR, name)
    write_dn(path, m, n, K, mu, args.dt, args.delta_d, args.v_m, args.pcap,
             k_arr, w, r0, p, d0)
    return path, wp_tag


def self_check(path: str):
    """Re-parse and verify structural invariants; returns the instance."""
    dn = DNInstance(path)
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    assert n_lines == 1 + dn.n + 2 * dn.m * dn.n, \
        "line count %d != %d" % (n_lines, 1 + dn.n + 2 * dn.m * dn.n)
    for k in range(dn.K):
        cnt = sum(1 for x in dn.k_arr if x == k)
        base, rem = divmod(dn.n, dn.K)
        assert cnt in (base, base + 1), "unbalanced arrivals at step %d" % k
    return dn


def report_stats(dn: DNInstance, wp_tag: str):
    """Print dataset statistics relevant for experiment design."""
    thr = dn.threatened()
    thr_value = sum(dn.w[j] for j in thr)
    total_value = dn.total_value()
    per_step = [sum(1 for x in dn.k_arr if x == k) for k in range(dn.K)]
    # flight-time span over threatened targets at their arrival step
    hs = [dn.flight_steps(i, j, dn.k_arr[j]) for j in thr for i in dn.W]
    hs_late = [dn.flight_steps(i, j, dn.breakthrough_step(j) - 1)
               for j in thr for i in dn.W] if thr else []
    print("  m=%d n=%d K=%d mu=%d (total ammo %d) | dt=%s dd=%s v_m=%s pcap=%d%%"
          % (dn.m, dn.n, dn.K, dn.mu, dn.m * dn.mu,
             repr(dn.dt), repr(dn.delta_d), repr(dn.v_m), dn.pcap_pct))
    print("  w/p source: %s" % wp_tag)
    print("  arrivals per step: %s" % per_step)
    print("  threatened (breakthrough < K): %d/%d targets, value %d/%d (%.1f%%)"
          % (len(thr), dn.n, thr_value, total_value,
             100.0 * thr_value / total_value if total_value else 0.0))
    if thr:
        print("  ammo vs threat: %d shots for %d threatened targets"
              % (dn.m * dn.mu, len(thr)))
        print("  flight h at arrival: %d..%d steps | at last pre-leak step: %d..%d"
              % (min(hs), max(hs), min(hs_late), max(hs_late)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="DN-WTA v1 dataset generator")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--grid", action="store_true",
                      help="generate the MARL grid m{3,5,10,20} x n{10,20,50,100}")
    mode.add_argument("--single", action="store_true", help="generate one instance")
    ap.add_argument("--m", type=int, help="weapon count (single mode)")
    ap.add_argument("--n", type=int, help="target count (single mode)")
    ap.add_argument("--seed", type=int, default=1, help="seed (single mode / grid offset)")
    ap.add_argument("--grid-seeds", default="1,2",
                    help="comma-separated seeds per grid cell (default 1,2)")
    ap.add_argument("--K", type=int, default=0,
                    help="override time-steps (default: 10 for n<=50 else 15)")
    ap.add_argument("--ammo-ratio", type=float, default=0.3,
                    help="mu = max(1, min(K, ceil(ratio*n/m))) (default 0.3)")
    ap.add_argument("--dt", type=float, default=2.0, help="seconds per step")
    ap.add_argument("--delta-d", type=float, default=1.0, help="km closed per step")
    ap.add_argument("--v-m", type=float, default=1.5, help="interceptor speed km/s")
    ap.add_argument("--pcap", type=int, default=95, help="probability cap percent")
    ap.add_argument("--r0-min", type=int, default=5, help="min r0 km")
    ap.add_argument("--r0-max", type=int, default=15, help="max r0 km")
    ap.add_argument("--b-max", type=int, default=3,
                    help="max platform offset b_i km (generation only)")
    ap.add_argument("--eps-max", type=int, default=2,
                    help="max per-pair jitter eps_ij km (generation only)")
    ap.add_argument("--pool", default=DEFAULT_POOL,
                    help="static instance sampled for w/p (empty string disables)")
    args = ap.parse_args(argv)

    if args.single and (not args.m or not args.n):
        ap.error("--single requires --m and --n")
    if not (0 < args.r0_min <= args.r0_max):
        ap.error("--r0-min/--r0-max must satisfy 0 < r0-min <= r0-max")
    if not (1 <= args.pcap <= 100):
        ap.error("--pcap must be in [1, 100]")

    jobs = []
    if args.grid:
        seeds = [int(s) for s in args.grid_seeds.split(",") if s.strip()]
        for m in GRID_M:
            for n in GRID_N:
                for s in seeds:
                    jobs.append((m, n, args.seed + s - 1))
    else:
        jobs.append((args.m, args.n, args.seed))

    print("generating %d DN-WTA v1 instance(s) into %s" % (len(jobs), DATA_DIR))
    made = []
    for (m, n, seed) in jobs:
        path, wp_tag = make_instance(m, n, seed, args)
        dn = self_check(path)
        made.append(path)
        print("%s" % os.path.basename(path))
        report_stats(dn, wp_tag)
    print("done: %d file(s) written and verified" % len(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
