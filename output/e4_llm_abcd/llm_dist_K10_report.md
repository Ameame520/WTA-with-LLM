# DWTA dynamic simulation report

- instance: `data/dyn_wta_50x100x3_K10_dist.txt`
- m (weapons): 50, n (targets): 100, K (waves): 10, mu: 3, total value: 5521
- seeds: 3 (seed base 42), delta: 0.001, timelimit: 60, threads: 1, branching: cplex
- result hash: `b4de427c505b5c7c30f8239b95410c34d918c9c4db4b0aa78dadeb2e025e21ad`
- generated at: 2026-08-21 01:39:56

## Per-wave averages (over 3 MC runs)

| wave | avg new | avg stay | avg targets | avg dist (km) | avg best p | age mix | avg expected cost | avg destroyed value | bt | avg bt leak | avg cumulative leak | avg wall time (s) | solved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.00 | 0.00 | 10.00 | 5.90 | 0.9500 | 10.00/0.00/0.00 | 0.0035 | 377.0 | 0.00 | 0.0 | 30.0 | 60.46 | 3/3 |
| 1 | 10.00 | 1.33 | 11.33 | 6.43 | 0.9445 | 10.00/1.33/0.00 | 0.0262 | 477.3 | 0.00 | 0.0 | 45.7 | 60.45 | 3/3 |
| 2 | 10.00 | 1.00 | 11.00 | 6.02 | 0.9491 | 10.00/1.00/0.00 | 0.0003 | 476.7 | 0.00 | 0.0 | 105.0 | 60.39 | 3/3 |
| 3 | 10.00 | 2.67 | 12.67 | 6.47 | 0.9500 | 10.00/2.00/0.67 | 0.0065 | 601.3 | 0.00 | 0.0 | 68.7 | 60.45 | 3/3 |
| 4 | 10.00 | 2.33 | 12.33 | 5.69 | 0.9477 | 10.00/1.33/1.00 | 0.0111 | 720.7 | 0.00 | 0.0 | 0.0 | 60.44 | 3/3 |
| 5 | 10.00 | 0.00 | 10.00 | 6.90 | 0.9460 | 10.00/0.00/0.00 | 0.0016 | 524.3 | 0.00 | 0.0 | 93.7 | 60.35 | 3/3 |
| 6 | 10.00 | 2.33 | 12.33 | 6.00 | 0.9500 | 10.00/2.33/0.00 | 0.0100 | 560.0 | 0.00 | 0.0 | 24.7 | 60.39 | 3/3 |
| 7 | 10.00 | 2.00 | 12.00 | 6.68 | 0.9500 | 10.00/2.00/0.00 | 0.0068 | 488.3 | 0.00 | 0.0 | 17.3 | 60.40 | 3/3 |
| 8 | 10.00 | 2.67 | 12.67 | 7.24 | 0.9500 | 10.00/2.00/0.67 | 0.0295 | 672.0 | 0.00 | 0.0 | 7.3 | 60.44 | 3/3 |
| 9 | 10.00 | 1.33 | 11.33 | 5.61 | 0.9497 | 10.00/0.67/0.67 | 0.0125 | 623.3 | 0.00 | 0.0 | 0.0 | 60.38 | 3/3 |

Age mix column: mean number of active targets per age (age 0 = arriving wave, 1, 2, ...; distance instances cap at L-1).

## Monte-Carlo summary

| seed | leak value | leak rate | breakthrough count | breakthrough leak | solver runtime (s) |
|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 0.000000 | 0 | 0 | 600.00 |
| 43 | 0 | 0.000000 | 0 | 0 | 600.00 |
| 44 | 0 | 0.000000 | 0 | 0 | 600.00 |

| metric | value |
|---|---:|
| leak rate (mean +- std) | 0.000000 +- 0.000000 |
| leak rate min / max | 0.000000 / 0.000000 |
| mean leak value | 0.0 of 5521 |
| mean breakthrough count (total) | 0.00 |
| mean breakthrough leak (total) | 0.0 |
| avg total solver runtime (s) | 600.00 |
| avg total wall time (s) | 604.14 |

Field notes: `expected cost` = solver's optimised expected surviving value for the wave's target set (effective, distance-scaled probabilities); `avg dist` / `avg best p` = mean over active targets of the current distance d_j(k) and best single-weapon effective hit probability; `cumulative leak` = weight of targets still alive after the wave plus all breakthrough value so far; `bt` / `bt leak` = targets that broke through in that wave (alive after their L-th wave) and their leaked weight; final leak value is the leak after the last wave. Legacy instances without the d0 column: distance stays 1 km, no breakthrough occurs.
