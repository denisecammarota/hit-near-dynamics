"""Run the full (N, eps) grid with one introduction and save final sizes.

Usage (from this folder):  python run_grid.py

Output: results/sizes_N{N}_eps{eps}.npz with arrays `cases` and `duration`.
Existing files are skipped, so the script can be interrupted and resumed.
"""
import os
import time
import numpy as np
from sir_sim import simulate_final_sizes, R0

N_VEC = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
# -0.05, -0.04, -0.03 and 0.1 keep the values used in the paper figures;
# -0.02 to 0.055 in steps of 0.005 is the fine grid near the HIT
# (immunity ~95.3% to ~87.8% with R0 = 15) used for the data comparison.
EPS_VEC = ([-0.05, -0.04, -0.03]
           + [float(e) for e in np.round(np.arange(-0.02, 0.0551, 0.005), 3)]
           + [0.1])
REPS = 10_000
OUT = "results"


def fname(N, eps):
    return os.path.join(OUT, f"sizes_N{N}_eps{eps:+.3f}.npz")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    for N in N_VEC:
        for eps in EPS_VEC:
            f = fname(N, eps)
            if os.path.exists(f):
                continue
            t = time.time()
            cases, dur = simulate_final_sizes(N, eps, reps=REPS, I0=1, seed=123)
            np.savez_compressed(f, cases=cases.astype(np.int32),
                                duration=dur.astype(np.float32))
            print(f"N={N:>7} eps={eps:+.3f} immunity={1-1/R0-eps:.4f} "
                  f"{time.time()-t:6.1f}s", flush=True)
    print(f"total {time.time()-t_all:.0f}s")
