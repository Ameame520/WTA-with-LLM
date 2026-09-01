"""CLI: generate DN-WTA dataset instances (spec: DN-WTA_v3_数据集说明.md).

v1 (2026-08-25, file format) -> v2 (2026-08-26, params) -> v3 (2026-08-28).

DN-WTA v3 keeps the v1 file format, the v2 mechanics and the information
boundary untouched; only three parameter/distribution/protocol changes:

    w_trend = 1.5  (R1)               w = max(1, round(U{1..99} *
                                       (1 + 1.5 * k_arr / (K-1))))
                                       -> per-window mean w ~50 (k=0) to
                                       ~125 (k=9), range [1, 248]; the trend
                                       depends only on k_arr at generation
                                       time and is inferable from the public
                                       t at run time (spec §7 unchanged)
    mu = 6 per platform  (R1)         GLOBAL pool 18 (v2: 24): exhausts by
                                       decision step ~5-6 at full tempo,
                                       forcing cross-window ammo trade-offs
    r0 quota = (1, 1, 3)  (R2)        per wave: 1x 1-window, 1x 2-window,
                                       3x 3-window targets (v2: 2,2,1) ->
                                       60% of targets allow a re-shot loop;
                                       no-defense leak histogram
                                       0,1,2,5,...,5 + end settlement 12
    30-instance family  (R3)          s01-s02 test / s03-s26 train /
                                       s27-s30 val (fixed split, see
                                       data/dn-data-v3/MANIFEST.md)

Regression: `--w-trend 0 --mu 8 --quota 2,2,1` reproduces v2 instances
byte-identically (acceptance V8).

Usage:

    python experiments/gen_dn_data.py                  # v3 defaults, seeds 1,2
    python experiments/gen_dn_data.py --seeds 1,2,...,30 \
        --outdir data/dn-data-v3 --pad 2               # the v3 family (s01..s30)

All draws are reproducible from --seed (5 independent RandomState streams,
unchanged from v2: split / r0 / b / eps / w-p).
Verified invariants after each write: re-parses via DNInstance, exact line
count 1+n+2mn, k_arr balanced, d0 >= r0, p in (0,1).
"""

import argparse
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dwta.dn_instance import DNInstance, write_dn  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # default --outdir (v2 layout)

# per-wave r0 quota: how many of the 5 targets per wave get engagement
# windows of 1/2/3 steps (R2)
QUOTA_V2 = (2, 2, 1)   # v2: leak histogram 2,4,5,...,5 per window + end 9
QUOTA_V3 = (1, 1, 3)   # v3: leak histogram 0,1,2,5,...,5 per window + end 12


def balanced_k_arr(n: int, K: int, rng) -> list:
    """Balanced arrival split over K steps (n % K targets extra in early
    steps), target ids shuffled by rng."""
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


def quota_r0(k_arr, K, args, rng) -> list:
    """Per-wave quota sampling of r0 so no-defense breakthroughs stay uniform.

    Within EVERY wave the n/K targets get engagement windows
    (ceil(r0/delta_d)) with per-window counts from args.quota (default v3
    quota (1, 1, 3); v2 used (2, 2, 1)):
        q1 x r0 = delta_d               (1 window, 10 s warning)
        q2 x r0 ~ U{2..3}x delta_d      (2 windows, 20 s)
        q3 x r0 ~ U{3x delta_d..r0_max} (3 windows, 30 s)
    -> v3 leak histogram 0, 1, 2, 5, 5, ..., 5 per window + end settlement
    12 (derivation: DN-WTA_v3_数据集说明.md appendix A).
    """
    lo, hi = args.r0_min, args.r0_max
    bands = [
        (int(lo), int(args.delta_d)),                                  # 1 win
        (int(args.delta_d + 1), int(2 * args.delta_d)),                # 2 win
        (int(2 * args.delta_d + 1), int(3 * args.delta_d)),            # 3 win
    ]
    quota = [int(q) for q in args.quota]
    r0 = [0] * len(k_arr)
    for k in range(K):
        members = [j for j, a in enumerate(k_arr) if a == k]
        picks = []
        for (blo, bhi), q in zip(bands, quota):
            picks.extend(int(rng.randint(blo, bhi + 1)) for _ in range(q))
        rng.shuffle(picks)
        for j, r in zip(members, picks):
            r0[j] = r
    return r0


def make_instance(m, n, K, mu, args, seed) -> str:
    rs_split = np.random.RandomState(seed * 100 + 1)
    rs_r0 = np.random.RandomState(seed * 100 + 2)
    rs_b = np.random.RandomState(seed * 100 + 3)
    rs_eps = np.random.RandomState(seed * 100 + 4)
    rs_wp = np.random.RandomState(seed * 100 + 5)

    k_arr = balanced_k_arr(n, K, rs_split)
    r0 = quota_r0(k_arr, K, args, rs_r0)
    b = rs_b.randint(0, args.b_max + 1, size=m)
    eps = rs_eps.randint(0, args.eps_max + 1, size=(m, n))
    d0 = {(i, j): int(r0[j] + b[i] + eps[i, j])
          for i in range(m) for j in range(n)}

    # w ~ U{1..99} scaled by the arrival-window trend (v3 R1), then p ~ U(0,1).
    # rs_wp draw order is unchanged from v2 (w base first, then p);
    # --w-trend 0 makes the scaling the identity -> v2 bytes.
    base = rs_wp.randint(1, 100, size=n)
    k_np = np.asarray(k_arr, dtype=float)
    trend = 1.0 + args.w_trend * (k_np / max(K - 1, 1))   # k=0 -> 1.0
    w = np.maximum(1, np.rint(base * trend)).astype(int).tolist()
    p = {(i, j): float(rs_wp.uniform(0.0001, 0.9999))
         for i in range(m) for j in range(n)}

    if args.pad > 0:
        name = "dn_%dx%d_K%d_s%0*d.txt" % (m, n, K, args.pad, seed)
    else:
        name = "dn_%dx%d_K%d_s%d.txt" % (m, n, K, seed)
    path = os.path.join(args.outdir, name)
    write_dn(path, m, n, K, mu, args.dt, args.delta_d, args.v_m, args.pcap,
             k_arr, w, r0, p, d0)
    return path


def self_check(path: str, K: int) -> DNInstance:
    dn = DNInstance(path)
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    assert n_lines == 1 + dn.n + 2 * dn.m * dn.n, "line count mismatch"
    per = [sum(1 for x in dn.k_arr if x == k) for k in range(dn.K)]
    base, rem = divmod(dn.n, K)
    assert all(c in (base, base + 1) for c in per), "unbalanced arrivals"
    return dn


def report(dn: DNInstance):
    per_step = [sum(1 for x in dn.k_arr if x == k) for k in range(dn.K)]
    # no-defense breakthrough histogram; alive-inbound at t=K settled at K
    bt = [min(dn.breakthrough_step(j), dn.K) for j in dn.T]
    bt_hist = [sum(1 for x in bt if x == k) for k in range(dn.K + 1)]
    hs = [dn.flight_steps(i, j, dn.k_arr[j])
          for j in dn.T for i in dn.W]
    win = [math.ceil(r / dn.delta_d) for r in dn.r0]
    mean_w = [(sum(dn.w[j] for j in dn.T if dn.k_arr[j] == k) / per_step[k])
              if per_step[k] else 0.0 for k in range(dn.K)]
    print("  m=%d n=%d K=%d mu=%d(total %d, global) dt=%gs v_t=%.2fkm/s "
          "v_m=%.2fkm/s pcap=%d%%"
          % (dn.m, dn.n, dn.K, dn.mu, dn.m * dn.mu, dn.dt,
             dn.delta_d / dn.dt, dn.v_m, dn.pcap_pct))
    print("  arrivals per window : %s" % per_step)
    print("  per-window mean w   : %s" % ["%.1f" % x for x in mean_w])
    print("  eng. window (steps) : %s  (r0=%s km)"
          % (sorted(set(win)), sorted(dn.r0)))
    print("  leak step histogram : %s  (index K = end settlement, no defense)"
          % bt_hist)
    print("  no-defense leaks    : %d/%d by t=%gs"
          % (sum(bt_hist), dn.n, dn.K * dn.dt))
    print("  flight steps at det.: %d..%d" % (min(hs), max(hs)))
    print("  total value %d | %d shots vs %d targets: best-case defense "
          "<=%.1f%%" % (dn.total_value(), dn.m * dn.mu, dn.n,
                        100.0 * dn.m * dn.mu / dn.n))


def main(argv=None):
    ap = argparse.ArgumentParser(description="DN-WTA v3 dataset generator")
    ap.add_argument("--seeds", default="1,2", help="comma-separated seeds")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--mu", type=int, default=6,
                    help="per-platform ammo, global pool m*mu (v3 default 6; "
                         "v2 was 8)")
    ap.add_argument("--w-trend", type=float, default=1.5,
                    help="value trend slope s: w = max(1, round(U{1..99} * "
                         "(1 + s*k_arr/(K-1)))); 0 regresses to the v2 "
                         "distribution (default 1.5)")
    ap.add_argument("--quota", default=",".join(str(q) for q in QUOTA_V3),
                    help="per-wave r0 quota for 1/2/3-window targets "
                         "(v3 default 1,1,3; v2 was 2,2,1)")
    ap.add_argument("--outdir", default="data",
                    help="output directory for the instance files "
                         "(relative to cwd; default 'data')")
    ap.add_argument("--pad", type=int, default=0,
                    help="zero-pad the seed field in file names (v3 family "
                         "uses 2 -> s01..s30; 0 keeps the v2 name layout)")
    ap.add_argument("--dt", type=float, default=10.0, help="seconds per step")
    ap.add_argument("--delta-d", type=float, default=3.0,
                    help="km closed by target per step (=v_t*dt)")
    ap.add_argument("--v-m", type=float, default=1.0, help="interceptor km/s")
    ap.add_argument("--pcap", type=int, default=95)
    ap.add_argument("--r0-min", type=int, default=3)
    ap.add_argument("--r0-max", type=int, default=9)
    ap.add_argument("--b-max", type=int, default=3)
    ap.add_argument("--eps-max", type=int, default=2)
    args = ap.parse_args(argv)
    try:
        args.quota = tuple(int(x) for x in str(args.quota).split(","))
    except ValueError:
        ap.error("--quota must be three comma-separated ints, e.g. 1,1,3")
    if len(args.quota) != 3 or any(q < 0 for q in args.quota):
        ap.error("--quota must be three non-negative ints (1/2/3-window "
                 "counts per wave)")
    if not (0 < args.r0_min <= args.r0_max):
        ap.error("--r0-min/--r0-max must satisfy 0 < min <= max")
    if args.n % args.K != 0:
        ap.error("waves require n %% K == 0 (got n=%d, K=%d)" % (args.n, args.K))
    if args.n // args.K != 5:
        ap.error("quota sampling assumes 5 targets per wave (n/K == 5)")
    if sum(args.quota) != args.n // args.K:
        ap.error("--quota must sum to n/K == %d (got %d)"
                 % (args.n // args.K, sum(args.quota)))
    if args.r0_max < 3 * args.delta_d:
        ap.error("r0-max must be >= 3*delta-d for the 3-window quota band")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print("generating %d DN-WTA v3 instance(s) into %s "
          "(mu=%d, w_trend=%g, quota=%s, pad=%d)"
          % (len(seeds), args.outdir, args.mu, args.w_trend,
             ",".join(str(q) for q in args.quota), args.pad))
    for seed in seeds:
        path = make_instance(args.m, args.n, args.K, args.mu, args, seed)
        dn = self_check(path, args.K)
        print("%s" % os.path.basename(path))
        report(dn)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
