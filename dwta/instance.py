"""Dynamic instance parsing / writing.

Dynamic file format (backward-compatible extension of the static format):

    line 1            : m n K mu [reserved extra integers ...]
    next n lines      : k w_j [d0_j]   (arrival wave k in [0, K), weight,
                                        optional initial distance in km)
    next m*n lines    : i j p_ij       (same as static format)

The first line of a *static* file holds exactly `m n mu`; this parser
treats 3 tokens as an error and asks the user to run gen_dynamic_data.py.

Distance extension (gen_dynamic_data.py --dist): when the target lines
carry a third column d0_j, the header extras hold `L pcap%` and
killobs are scaled with the closing distance:

    d_j(k)     = max(1, d0_j - (k - k_arr(j)))          (1 km per wave)
    p_eff(i,j,k) = min(pcap, p_ij * d0_j / d_j(k))

Old files without the d0 column keep d0_j = 1 ("no closing"): distances
stay at 1 km and p_eff equals the base p_ij, so behaviour is identical
to the previous version.
"""

import os


class StaticInstanceError(ValueError):
    """Raised when a static (non-dynamic) instance file is given to the dynamic pipeline."""


def parse_static(path: str):
    """Parse an original static instance file.

    Returns dict with keys: m, n, mu, w (list of weights), p (dict {(i, j): prob}).
    """
    with open(path, "r") as f:
        lines = f.readlines()
    head = lines[0].split()
    if len(head) < 3:
        raise ValueError("Invalid static instance header: %r" % lines[0])
    m, n, mu = (int(v) for v in head[:3])
    weights = [int(lines[1 + j].split()[0]) for j in range(n)]
    p = {}
    expected = 1 + n + m * n
    if len(lines) < expected:
        raise ValueError("Static instance truncated: %d lines, expected %d" % (len(lines), expected))
    for line in lines[1 + n: expected]:
        parts = line.split()
        i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
        p[i, j] = v
    return {"m": m, "n": n, "mu": mu, "w": weights, "p": p}


class DynInstance:
    """Parsed dynamic instance."""

    def __init__(self, path: str):
        with open(path, "r") as f:
            lines = f.readlines()

        head = lines[0].split()
        if len(head) == 3:
            raise StaticInstanceError(
                "File %s is a *static* WTA instance (header 'm n mu' = %r).\n"
                "Convert it first, e.g.:\n"
                "  python experiments/gen_dynamic_data.py --input %s --waves <K> "
                "--seed <S> --output <dynamic_instance.txt>" % (path, head, path)
            )
        if len(head) < 4:
            raise ValueError("Invalid dynamic instance header: %r" % lines[0])

        # Forward compatibility: only the first 4 integers are used now;
        # extra header fields are preserved. With the distance extension
        # they carry [L, pcap%].
        self.m, self.n, self.K, mu = (int(v) for v in head[:4])
        self.extra_header = [int(v) for v in head[4:]]
        self.mu = mu

        expected = 1 + self.n + self.m * self.n
        if len(lines) < expected:
            raise ValueError("Dynamic instance truncated: %d lines, expected %d" % (len(lines), expected))

        self.wave = []
        self.w = []
        self.d0 = []
        has_dist = False
        for j in range(self.n):
            parts = lines[1 + j].split()
            k, wj = int(parts[0]), int(parts[1])
            if not (0 <= k < self.K):
                raise ValueError("Target %d has wave %d outside [0, %d)" % (j, k, self.K))
            self.wave.append(k)
            self.w.append(wj)
            if len(parts) >= 3:
                has_dist = True
                dj0 = int(parts[2])
                if dj0 < 1:
                    raise ValueError("Target %d has d0=%d, must be >= 1 km" % (j, dj0))
                self.d0.append(dj0)
            else:
                self.d0.append(1)  # legacy file: no closing behaviour

        # Distance-extension parameters: fire range L (waves), probability
        # cap pcap (fraction). Defaults 3 / 0.95 apply to legacy files,
        # where they are never consulted (effective_p returns base p).
        self.has_dist = has_dist
        if has_dist:
            if len(self.extra_header) >= 2:
                self.L, self.pcap = self.extra_header[0], self.extra_header[1] / 100.0
            else:  # d0 columns without header extras: documented defaults
                self.L, self.pcap = 3, 0.95
        else:
            self.L, self.pcap = 3, 0.95
        self.dist_step = 1  # km closed per wave
        self.d_min = 1      # floor distance (km)

        self.p = {}
        for line in lines[1 + self.n: expected]:
            parts = line.split()
            i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
            self.p[i, j] = v

        self.W = list(range(self.m))
        self.T = list(range(self.n))

    # ------------------------------------------------------------------

    def targets_in_wave(self, k: int):
        """Original ids of targets arriving at wave k (ascending order)."""
        return [j for j in self.T if self.wave[j] == k]

    def total_value(self) -> int:
        return sum(self.w)

    def dist(self, j: int, k: int) -> int:
        """Distance d_j(k) in km of target j at wave k.

        Targets close `dist_step` km per wave after arrival; never below
        `d_min`. Legacy files (no d0 column) stay at 1 km ("no closing").
        """
        if not self.has_dist:
            return self.d_min
        return max(self.d_min, self.d0[j] - (k - self.wave[j]) * self.dist_step)

    def effective_p(self, i: int, j: int, k: int) -> float:
        """Effective hit probability of weapon i on target j at wave k.

        p_eff = min(pcap, p_ij * d0_j / d_j(k)); at arrival (k = k_arr)
        the ratio is 1 so only the cap may bite. Legacy files (no d0
        column) return the base p_ij unchanged.
        """
        if not self.has_dist:
            return self.p[i, j]
        return min(self.pcap, self.p[i, j] * self.d0[j] / float(self.dist(j, k)))


def write_dynamic(path: str, m: int, K: int, mu: int, wave_of_target, weights, p,
                  extra_header=None, d0=None):
    """Write a dynamic instance file.

    wave_of_target: list of length n with arrival wave per target.
    weights: list of length n. p: dict {(i, j): prob}.
    d0: optional list of length n with initial distances (km); when given,
       a third column d0_j is appended to each target line and `extra_header`
       is expected to carry [L, pcap%].
    """
    n = len(weights)
    head = [str(m), str(n), str(K), str(mu)]
    if extra_header:
        head += [str(v) for v in extra_header]
    out = [" ".join(head)]
    for j in range(n):
        line = "%d %d" % (wave_of_target[j], weights[j])
        if d0 is not None:
            line += " %d" % d0[j]
        out.append(line)
    for i in range(m):
        for j in range(n):
            out.append("%d %d %.12f" % (i, j, p[i, j]))
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
