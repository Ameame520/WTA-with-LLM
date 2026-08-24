"""CLI: convert a static WTA instance into a dynamic (multi-wave) instance.

Usage:
    python experiments/gen_dynamic_data.py --input <static.txt> --waves K --seed S \
        --output <dyn.txt> [--shuffle/--no-shuffle]
    python experiments/gen_dynamic_data.py --all --waves K --seed S   # batch over data/

Split policy: balanced sizes (n//K or n//K+1 per wave). With shuffle enabled
(default) target ids are permuted with numpy RandomState(seed) before being
dealt to waves; with --no-shuffle targets keep their original order and are
split contiguously.

Distance extension (--dist): each target gets an initial distance
d0_j ~ Uniform{2..10} km appended as a 3rd column, and the header gains
two extra integers `L pcap%` (default 3 95). Writing format:

    line 1            : m n K mu L pcap%
    next n lines      : k w_j d0_j
    next m*n lines    : i j p_ij (unchanged)
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dwta.instance import parse_static, write_dynamic, DynInstance  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def balanced_sizes(n, K):
    base, rem = divmod(n, K)
    return [base + (1 if k < rem else 0) for k in range(K)]


def split_waves(n, K, shuffle=True, seed=0):
    """Return list wave_of_target (length n, values in [0, K))."""
    sizes = balanced_sizes(n, K)
    if shuffle:
        order = np.random.RandomState(seed).permutation(n).tolist()
    else:
        order = list(range(n))
    wave_of_target = [0] * n
    pos = 0
    for k, size in enumerate(sizes):
        for _ in range(size):
            wave_of_target[order[pos]] = k
            pos += 1
    assert pos == n
    return wave_of_target


def draw_d0(n, seed, d_min=2, d_max=10):
    """Initial distances d0_j ~ Uniform{d_min..d_max} km via RandomState(seed)."""
    return np.random.RandomState(seed).randint(d_min, d_max + 1, size=n).tolist()


def self_check(path, m, n, K, mu, wave_of_target, d0=None, d_min=2, d_max=10,
               L=None, pcap=None):
    """Re-parse the produced file and verify integrity."""
    dyn = DynInstance(path)
    assert dyn.m == m and dyn.n == n and dyn.K == K and dyn.mu == mu, \
        "header mismatch after write"
    assert dyn.wave == list(wave_of_target), "wave assignment mismatch after write"
    assert sorted(set(dyn.wave)) == sorted(set(wave_of_target)), "wave coverage broken"
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    assert n_lines == 1 + n + m * n, \
        "line count %d != %d" % (n_lines, 1 + n + m * n)
    for k in range(K):
        cnt = sum(1 for w in dyn.wave if w == k)
        assert cnt == sum(1 for w in wave_of_target if w == k)

    if d0 is not None:
        # header extension: extra fields must be exactly [L, pcap%]
        assert dyn.extra_header == [L, pcap], \
            "dist header mismatch: %r != [%d, %d]" % (dyn.extra_header, L, pcap)
        # every target line carries a third column d0_j within [d_min, d_max]
        with open(path) as f:
            lines = f.readlines()
        for j in range(n):
            parts = lines[1 + j].split()
            assert len(parts) == 3, "target line %d lacks d0 column: %r" % (j + 2, parts)
            dv = int(parts[2])
            assert d_min <= dv <= d_max, \
                "d0[%d]=%d outside [%d, %d]" % (j, dv, d_min, d_max)
            assert dv == d0[j], "d0 mismatch at target %d" % j


def print_d0_histogram(d0, d_min=2, d_max=10, width=40):
    counts = {}
    for v in d0:
        counts[v] = counts.get(v, 0) + 1
    peak = max(counts.values())
    print("  d0 distribution (km):")
    for v in range(d_min, d_max + 1):
        c = counts.get(v, 0)
        bar = "#" * int(round(c / float(peak) * width)) if peak else ""
        print("    %2d | %-8d %s" % (v, c, bar))


def convert(input_path, waves, seed, output_path, shuffle=True, dist=False,
            L=3, pcap=95, d0_min=2, d0_max=10):
    static = parse_static(input_path)
    m, n, mu = static["m"], static["n"], static["mu"]
    wave_of_target = split_waves(n, waves, shuffle=shuffle, seed=seed)
    d0 = draw_d0(n, seed, d0_min, d0_max) if dist else None
    write_dynamic(output_path, m, waves, mu, wave_of_target, static["w"], static["p"],
                  extra_header=[L, pcap] if dist else None, d0=d0)
    self_check(output_path, m, n, waves, mu, wave_of_target, d0=d0,
               d_min=d0_min, d_max=d0_max, L=L, pcap=pcap)
    counts = [wave_of_target.count(k) for k in range(waves)]
    print("converted %s -> %s" % (input_path, output_path))
    print("  m=%d n=%d K=%d mu=%d, targets per wave: %s" % (m, n, waves, mu, counts))
    if dist:
        print("  L=%d, pcap=%d%% (0.%02d), d0 ~ Uniform{%d..%d} (seed=%d)"
              % (L, pcap, pcap, d0_min, d0_max, seed))
        print_d0_histogram(d0, d0_min, d0_max)
    return output_path


def make_smoke_static(path, m=20, n=50, mu=1, seed=123):
    """Synthesise a small deterministic static instance (for --smoke demos)."""
    rng = np.random.RandomState(seed)
    weights = rng.randint(10, 100, size=n).tolist()
    p = {}
    for i in range(m):
        for j in range(n):
            p[i, j] = float(rng.uniform(0.05, 0.95))
    with open(path, "w") as f:
        f.write("%d %d %d\n" % (m, n, mu))
        for w in weights:
            f.write("%d\n" % int(w))
        for (i, j), v in sorted(p.items()):
            f.write("%d %d %.12f\n" % (i, j, v))
    return path


def dynamic_name(static_path, K, dist=False):
    base = os.path.basename(static_path)
    assert base.startswith("wta_") and base.endswith(".txt"), base
    return "dyn_" + base[:-4] + "_K%d%s.txt" % (K, "_dist" if dist else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Static -> dynamic WTA instance converter")
    ap.add_argument("--input", help="input static instance (wta_*.txt)")
    ap.add_argument("--waves", type=int, required=True, help="number of waves K")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random split")
    ap.add_argument("--output", help="output dynamic instance path")
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                    help="random balanced split (default); --no-shuffle keeps "
                         "original target order split contiguously")
    ap.add_argument("--dist", action="store_true",
                    help="add distance extension: d0_j ~ Uniform{d_min..d_max} km "
                         "3rd column per target, header extras 'L pcap%%'")
    ap.add_argument("--L", type=int, default=3,
                    help="fire-range threshold L km written to header (default 3)")
    ap.add_argument("--pcap", type=int, default=95,
                    help="max kill probability at distance 0 in percent (default 95)")
    ap.add_argument("--d0-min", type=int, default=2, help="min d0 km (default 2)")
    ap.add_argument("--d0-max", type=int, default=10, help="max d0 km (default 10)")
    ap.add_argument("--all", action="store_true",
                    help="convert every static instance in data/")
    args = ap.parse_args(argv)

    if args.waves < 1:
        ap.error("--waves must be >= 1")
    if args.dist and not (0 < args.d0_min <= args.d0_max):
        ap.error("--d0-min/--d0-max must satisfy 0 < d0-min <= d0-max")
    if args.dist and not (1 <= args.pcap <= 100):
        ap.error("--pcap must be a percent in [1, 100]")
    kw = dict(shuffle=args.shuffle, dist=args.dist, L=args.L, pcap=args.pcap,
              d0_min=args.d0_min, d0_max=args.d0_max)
    if args.all:
        statics = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.startswith("wta_") and f.endswith(".txt"))
        if not statics:
            print("no static instances found in data/", file=sys.stderr)
            return 1
        for f in statics:
            src = os.path.join(DATA_DIR, f)
            out = os.path.join(DATA_DIR, dynamic_name(f, args.waves, dist=args.dist))
            convert(src, args.waves, args.seed, out, **kw)
        print("batch conversion done: %d files" % len(statics))
        return 0

    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --all)")
    convert(args.input, args.waves, args.seed, args.output, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
