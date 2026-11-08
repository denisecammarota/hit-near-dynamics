"""Stochastic SIR final sizes near the herd immunity threshold.

Exact simulation of the stochastic SIR model (Gillespie direct method, written as
the embedded jump chain), vectorised across replicates. Each replicate runs
until I = 0, so no time step and no maximum simulation time are needed.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.special import lambertw

# ----------------------------------------------------------------------------
# Parameters (beta = 1, gamma = 0.1 per day -> R0 = 10)
# ----------------------------------------------------------------------------
BETA = 1.0
GAMMA = 0.1
R0 = BETA / GAMMA


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate_final_sizes(N, eps, reps=10_000, I0=1, seed=123):
    """Return (cases, duration) arrays of length `reps`.

    Initial condition: S(0) = N (1/R0 + eps), I(0) = I0, the rest recovered.
    cases    = total number infected, including the I0 initial cases.
    duration = time until I = 0, in days.
    """
    rng = np.random.default_rng(seed)
    S0 = int(round(N * (1 / R0 + eps)))
    S = np.full(reps, S0)
    I = np.full(reps, I0)
    T = np.zeros(reps)
    active = np.arange(reps)

    while active.size:
        s, i = S[active], I[active]
        inf_rate = BETA * s / N                       # infection rate per infected
        tot_rate = inf_rate + GAMMA                   # total event rate per infected
        is_inf = rng.random(active.size) * tot_rate < inf_rate
        T[active] += rng.exponential(1 / (tot_rate * i))
        S[active] = s - is_inf
        I[active] = i + np.where(is_inf, 1, -1)
        active = active[I[active] > 0]

    return (S0 - S) + I0, T


# ----------------------------------------------------------------------------
# Analytical results
# ----------------------------------------------------------------------------
def R_eff(eps):
    """Effective reproduction number, R_eff = R0 S(0)/N = 1 + R0 eps."""
    return 1 + R0 * eps


def p_small_theory(eps, I0=1):
    """Probability of a small outbreak from the branching process, (1/R_eff)^I0."""
    R = R_eff(np.asarray(eps, float))
    return np.where(R > 1, (1 / np.maximum(R, 1e-12)) ** I0, 1.0)


def final_size_fraction(eps):
    """Deterministic final size, as a proportion of N, for I(0)/N -> 0.

    C = S(0)/N * z, with z the positive solution of 1 - z = exp(-R_eff z),
    written with the principal branch of the Lambert W function.
    """
    R = R_eff(eps)
    if R <= 1:
        return 0.0
    z = 1 + np.real(lambertw(-R * np.exp(-R))) / R
    return (1 / R0 + eps) * z


# ----------------------------------------------------------------------------
# Classification and summaries
# ----------------------------------------------------------------------------
def threshold_cases(N, eps):
    """Small/big threshold C_s, in cases: half the deterministic final size.
    For eps <= 0 it is infinite, so that every outbreak is small."""
    C = final_size_fraction(eps)
    return np.inf if C == 0 else 0.5 * C * N


def summarise(cases, N, eps):
    """p_s, p_b, its standard error and the conditional mean sizes for one (N, eps)."""
    small = cases < threshold_cases(N, eps)
    p_s = small.mean()
    return dict(N=N, eps=eps, threshold=threshold_cases(N, eps),
                p_s=p_s, p_b=1 - p_s, se_p=np.sqrt(p_s * (1 - p_s) / cases.size),
                mean_small_cases=cases[small].mean() if small.any() else np.nan,
                mean_big_cases=cases[~small].mean() if (~small).any() else np.nan)


# ----------------------------------------------------------------------------
# Loading the simulations
# ----------------------------------------------------------------------------
def load_grid(out="results"):
    """Dict {(N, eps): (cases, duration)} from the files written by run_grid.py."""
    grid = {}
    for f in glob.glob(os.path.join(out, "sizes_N*_eps*.npz")):
        N, eps = re.search(r"N(\d+)_eps([+-][\d.]+)\.npz", f).groups()
        d = np.load(f)
        grid[(int(N), round(float(eps), 3))] = (d["cases"], d["duration"])
    return grid


def get(grid, N, eps):
    """(cases, duration) for one (N, eps)."""
    return grid[(N, round(eps, 3))]


def ps_table(grid):
    """DataFrame with the output of summarise for every (N, eps)."""
    rows = [summarise(c, N, eps) for (N, eps), (c, _) in grid.items()]
    return pd.DataFrame(rows).sort_values(["N", "eps"]).reset_index(drop=True)


def ps_at(table, N, eps, floor=1e-12):
    """p_s at population N, by linear interpolation in log N between simulated N."""
    t = table[np.isclose(table.eps, eps)].sort_values("N")
    p = np.interp(np.log(N), np.log(t.N), t.p_s)
    return float(np.clip(p, floor, 1 - floor))