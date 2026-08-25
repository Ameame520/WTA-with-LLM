"""DN-WTA v1 instance parsing / writing (spec: 数据集规则.md).

File layout (pure .txt, exactly 1 + n + 2*m*n lines):

    line 1          : m n K mu dt delta_d v_m pcap
    next n lines    : k_arr w_j r0_j          (per target)
    next m*n lines  : i j p_ij                (base kill probability)
    next m*n lines  : i j d0_ij               (platform-target distance at detection)

Header fields:
    m        weapon platforms (= number of MARL agents)
    n        total targets
    K        decision time-steps
    mu       initial total ammo per platform for the WHOLE episode
    dt       real seconds per time-step
    delta_d  km closed by each target per time-step
    v_m      interceptor flight speed (km/s)
    pcap     kill-probability cap in percent (e.g. 95 -> 0.95)

All dynamics are derivable from the file (nothing extra is stored):

    r_j(t)      = max(0, r0_j - delta_d*(t - k_arr_j))          boundary distance
    d_ij(t)     = max(0, d0_ij - delta_d*(t - k_arr_j))         plat-target distance
    p_eff_ij(t) = min(pcap, p_ij * d0_ij / d_ij(t))              d_ij(t) > 0
    tau_ij(t)   = d_ij(t) / v_m                                  flight time (s)
    h_ij(t)     = max(1, ceil(tau_ij(t)/dt))                     flight time-steps
    t_hit       = t + h_ij(t)                                    settlement step
    breakthrough when r_j(t) == 0 and the target is still alive.

This module is a pure data layer (standard library only): it parses, validates
and derives distances/probabilities/flight times. It deliberately contains no
simulation loop and no algorithm - the shared multi-agent environment and the
policies are built on top of it in later stages.
"""

import math
import os


class DNFormatError(ValueError):
    """Raised when a file does not conform to the DN-WTA v1 layout."""


def _int_token(tok, name):
    try:
        return int(tok)
    except ValueError:
        raise DNFormatError("%s must be an integer, got %r" % (name, tok))


def _pos_float_token(tok, name):
    try:
        v = float(tok)
    except ValueError:
        raise DNFormatError("%s must be a float, got %r" % (name, tok))
    if not math.isfinite(v) or v <= 0.0:
        raise DNFormatError("%s must be positive and finite, got %r" % (name, tok))
    return v


def _ceil_int(x, eps=1e-9):
    """ceil(x) robust to floating point noise (6.0/2.0 -> 3, not 4)."""
    return int(math.ceil(x - eps))


class DNInstance:
    """Parsed and validated DN-WTA v1 instance."""

    def __init__(self, path: str):
        with open(path, "r") as f:
            lines = f.readlines()

        head = lines[0].split()
        if len(head) != 8:
            raise DNFormatError(
                "DN-WTA v1 header must have 8 fields "
                "'m n K mu dt delta_d v_m pcap', got %d: %r"
                % (len(head), lines[0]))
        self.m = _int_token(head[0], "m")
        self.n = _int_token(head[1], "n")
        self.K = _int_token(head[2], "K")
        self.mu = _int_token(head[3], "mu")
        self.dt = _pos_float_token(head[4], "dt")
        self.delta_d = _pos_float_token(head[5], "delta_d")
        self.v_m = _pos_float_token(head[6], "v_m")
        self.pcap_pct = _int_token(head[7], "pcap")
        if self.m < 1 or self.n < 1:
            raise DNFormatError("need m >= 1 and n >= 1, got m=%d n=%d" % (self.m, self.n))
        if self.K < 1:
            raise DNFormatError("K must be >= 1, got %d" % self.K)
        if self.mu < 1:
            raise DNFormatError("mu must be >= 1, got %d" % self.mu)
        if not (1 <= self.pcap_pct <= 100):
            raise DNFormatError("pcap must be a percent in [1, 100], got %d" % self.pcap_pct)
        self.pcap = self.pcap_pct / 100.0

        expected = 1 + self.n + 2 * self.m * self.n
        if len(lines) < expected:
            raise DNFormatError("file truncated: %d lines, expected %d"
                                % (len(lines), expected))
        extra = [ln for ln in lines[expected:] if ln.strip()]
        if extra:
            raise DNFormatError("file has %d extra non-empty line(s) after the "
                                "distance block" % len(extra))

        # --- target block: k_arr w_j r0_j -------------------------------
        self.k_arr = []
        self.w = []
        self.r0 = []
        for j in range(self.n):
            parts = lines[1 + j].split()
            if len(parts) != 3:
                raise DNFormatError("target line %d must be 'k_arr w_j r0_j', got %r"
                                    % (2 + j, lines[1 + j]))
            k = _int_token(parts[0], "k_arr")
            wj = _int_token(parts[1], "w_j")
            r0 = _int_token(parts[2], "r0_j")
            if not (0 <= k < self.K):
                raise DNFormatError("target %d k_arr=%d outside [0, %d)"
                                    % (j, k, self.K))
            if wj < 1:
                raise DNFormatError("target %d weight %d must be >= 1" % (j, wj))
            if r0 < 1:
                raise DNFormatError("target %d r0=%d must be >= 1 km" % (j, r0))
            self.k_arr.append(k)
            self.w.append(wj)
            self.r0.append(r0)

        # --- probability block: i j p_ij --------------------------------
        self.p = {}
        pblock = lines[1 + self.n: 1 + self.n + self.m * self.n]
        for ln in pblock:
            parts = ln.split()
            i = _int_token(parts[0], "i")
            j = _int_token(parts[1], "j")
            v = float(parts[2])
            if not (0.0 < v < 1.0):
                raise DNFormatError("p_ij must be in (0, 1), got %r (line %r)"
                                    % (parts[2], ln))
            self.p[i, j] = v

        # --- distance block: i j d0_ij ----------------------------------
        self.d0 = {}
        dblock = lines[1 + self.n + self.m * self.n: expected]
        for ln in dblock:
            parts = ln.split()
            i = _int_token(parts[0], "i")
            j = _int_token(parts[1], "j")
            v = _int_token(parts[2], "d0_ij")
            if v < self.r0[j]:
                raise DNFormatError(
                    "d0[%d, %d]=%d violates the generation constraint d0_ij >= r0_j=%d"
                    % (i, j, v, self.r0[j]))
            self.d0[i, j] = v

        # index completeness of the two m*n blocks
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in self.p:
                    raise DNFormatError("probability block missing entry (%d, %d)" % (i, j))
                if (i, j) not in self.d0:
                    raise DNFormatError("distance block missing entry (%d, %d)" % (i, j))

        self.W = list(range(self.m))
        self.T = list(range(self.n))

    # ------------------------------------------------------------------
    # derivable dynamics (pure functions of the dataset)
    # ------------------------------------------------------------------

    def targets_arriving(self, t: int):
        """Original ids of targets first detected at time-step t (ascending)."""
        return [j for j in self.T if self.k_arr[j] == t]

    def total_value(self) -> int:
        return sum(self.w)

    def age(self, j: int, t: int) -> int:
        """Time-steps since target j was detected (0 at arrival)."""
        return t - self.k_arr[j]

    def r(self, j: int, t: int) -> float:
        """Distance of target j to the defense boundary at step t (km)."""
        return max(0.0, self.r0[j] - self.delta_d * (t - self.k_arr[j]))

    def dist(self, i: int, j: int, t: int) -> float:
        """Distance between platform i and target j at step t (km)."""
        return max(0.0, self.d0[i, j] - self.delta_d * (t - self.k_arr[j]))

    def p_eff(self, i: int, j: int, t: int) -> float:
        """Effective kill probability of platform i on target j at step t.

        p_eff = min(pcap, p_ij * d0_ij / d_ij(t)). Guarded for d_ij(t) = 0
        (target at the boundary): returns pcap; engagements only make sense
        while the target is not through, so this branch is defensive.
        """
        d = self.dist(i, j, t)
        if d <= 0.0:
            return self.pcap
        return min(self.pcap, self.p[i, j] * self.d0[i, j] / d)

    def flight_time(self, i: int, j: int, t: int) -> float:
        """Interceptor flight time in seconds for a shot fired at step t."""
        return self.dist(i, j, t) / self.v_m

    def flight_steps(self, i: int, j: int, t: int) -> int:
        """Flight time in whole time-steps: h = max(1, ceil(tau/dt))."""
        return max(1, _ceil_int(self.flight_time(i, j, t) / self.dt))

    def t_hit(self, i: int, j: int, t: int) -> int:
        """Time-step at which a missile fired at step t settles."""
        return t + self.flight_steps(i, j, t)

    def breakthrough_step(self, j: int) -> int:
        """First step t at which r_j(t) = 0 (target leaks if still alive)."""
        return self.k_arr[j] + _ceil_int(self.r0[j] / self.delta_d)

    def threatened(self) -> list:
        """Targets whose r hits 0 at some step in [0, K) (leak possible)."""
        return [j for j in self.T if self.breakthrough_step(j) < self.K]


def write_dn(path: str, m: int, n: int, K: int, mu: int, dt: float, delta_d: float,
             v_m: float, pcap_pct: int, k_arr, w, r0, p, d0):
    """Write a DN-WTA v1 instance file.

    k_arr/w/r0: lists of length n; p/d0: dicts {(i, j): value} covering all
    m*n pairs (p floats in (0,1), d0 ints in km with d0 >= r0).
    """
    head = ["%d %d %d %d %s %s %s %d"
            % (m, n, K, mu, repr(float(dt)), repr(float(delta_d)),
               repr(float(v_m)), pcap_pct)]
    tgt = ["%d %d %d" % (k_arr[j], w[j], r0[j]) for j in range(n)]
    prob = ["%d %d %.12f" % (i, j, p[i, j]) for i in range(m) for j in range(n)]
    dist = ["%d %d %d" % (i, j, d0[i, j]) for i in range(m) for j in range(n)]
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(head + tgt + prob + dist) + "\n")
    return path
