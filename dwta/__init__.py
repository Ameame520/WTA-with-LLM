"""DWTA - dynamic multi-wave weapon-target assignment simulation.

Wraps the original static branch-and-adjust solver (cplex/wta_cplex.py) via
subprocess and simulates K attack waves with probabilistic (Bernoulli)
damage settlement and target carry-over between waves.
"""

__version__ = "0.1.0"
