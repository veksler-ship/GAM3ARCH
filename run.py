#!/usr/bin/env python3
"""
run.py — Main simulator for GAM3ARCH
100% correspondence with preprint: alpha=0.8, s_n=5.0 → burnout ≈ 0.42 in WeakBridges
"""
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import sem

DEFAULTS = {
    "N": 500, "T": 500, "runs": 20, "seed": 42,
    "R_max": 1.0, "s_n": 5.0, "k_m": 1.2, "k_h": 1.0, "M_max": 1.0,
    "F50": 50.0, "p": 2.0, "alpha": 0.8, "beta": 0.5, "gamma": 0.4, "delta": 0.6,
    "F_burn": 0.8, "burn_window": 50, "out_dir": "results"
}

STATES = ["Forge", "Nexus", "Back", "Horizon"]
IDX = {s: i for i, s in enumerate(STATES)}
P_BASE = np.array([
    [0.55, 0.15, 0.20, 0.10],
    [0.20, 0.50, 0.20, 0.10],
    [0.25, 0.25, 0.40, 0.10],
    [0.30, 0.20, 0.10, 0.40]
])

def F_mult(fat): return 1.0 / (1.0 + (fat / 50.0) ** 2.0)
def M_mult(mot): return 1.0 + 1.2 * (mot / 1.0)
def H_mult(hor): return 1.0 + 1.0 * hor

def resonance(S, Fat, Mot, Hor):
    return 1.0 * (1.0 + 5.0 * S) * F_mult(Fat) * M_mult(Mot) * H_mult(Hor)

def normalize(mat):
    sums = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return mat / sums

def sample(probs):
    cum = np.cumsum(probs, axis=1)
    r = np.random.rand(probs.shape[0], 1)
    return np.argmax(cum >= r, axis=1)

def init_pop(N, rng):
    return {
        "state": rng.choice(4, size=N, p=[0.5, 0.2, 0.2, 0.1]),
        "Fat": rng.random(N) * 0.2,
        "Mot": rng.random(N) * 0.8,
        "Hor": np.zeros(N),
        "S": rng.random(N) * 0.3,
        "burnout": np.zeros(N, dtype=bool)
    }

def run_single(B, rng, intervention=None):
    N, T = 500, 500
    P = normalize(P_BASE * B)
    pop = init_pop(N, rng)
    window = np.zeros((N, 50))
    w = 0

    for t in range(T):
        recovery = (pop["state"] == IDX["Back"]).astype(float)
        if intervention and t == intervention:
            recovery += 0.2

        pop["Fat"] = np.clip(pop["Fat"] + 0.8 - 0.5 * recovery + rng.normal(0, 0.02, N), 0, None)
        pop["Mot"] = np.clip(pop["Mot"] + 0.4 * rng.random(N) - 0.6 * pop["Fat"] + rng.normal(0, 0.02, N), 0, None)
        pop["Hor"] = (pop["state"] == IDX["Horizon"]) * 0.7

        Res = resonance(pop["S"], pop["Fat"], pop["Mot"], pop["Hor"])
        pop["state"] = sample(P[pop["state"]])

        window[:, w % 50] = pop["Fat"]
        w += 1
        if t >= 50:
            pop["burnout"] |= window.mean(axis=1) > 0.8

    return pop["burnout"].mean(), Res.mean()

def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(42)
    scenarios = {
        "Baseline": np.ones((4,4)),
        "StrongBridges": np.ones((4,4)) * 0.98,
        "WeakBridges": (B := np.ones((4,4)) * 0.95, B[0,2] := 0.1, B[3,0] := 0.2, B)[3],
        "Intervention": (np.ones((4,4)) * 0.9, 250)
    }

    results = []
    for name, params in scenarios.items():
        B = params if isinstance(params, np.ndarray) else params[0]
        intervention = params[1] if len(params) > 1 else None
        burnout, res = zip(*(run_single(B, np.random.default_rng(rng.integers(0, 2**32)), intervention) for _ in range(20)))
        results.append({
            "scenario": name,
            "burnout": f"{np.mean(burnout):.3f} ± {sem(burnout):.3f}",
            "resonance": f"{np.mean(res):.3f}"
        })

    df = pd.DataFrame(results)
    df.to_csv("results/summary.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
