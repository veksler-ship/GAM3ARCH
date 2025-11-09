#!/usr/bin/env python3
"""GAM3ARCH v3.1 - Simulator (paper reproduction version)
This script implements the agent-based simulation used in the GAM3ARCH preprint (Nov 2025).
Parameters are set to reproduce results reported in the paper (alpha=0.8, s_n=5.0, F_burn=0.8, burn_window=50).
"""
__version__ = "3.1.0"
__author__ = "A. Skrobov"

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from scipy.stats import sem
from typing import Dict, Any

# Paper defaults (reproducibility)
DEFAULTS = {
    "N_agents": 500,
    "T_steps": 500,
    "runs": 20,
    "seed": 42,
    "R_max": 1.0,
    "s_n": 5.0,
    "k_m": 1.2,
    "k_h": 1.0,
    "M_max": 1.0,
    "F50": 50.0,
    "p": 2.0,
    "alpha": 0.8,
    "beta": 0.5,
    "gamma": 0.4,
    "delta": 0.6,
    "F_burn": 0.8,
    "burn_window": 50,
    "B_default": 1.0,
    "out_dir": "results"
}

STATE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]
STATE_IDX = {name: i for i, name in enumerate(STATE_NAMES)}

P_BASE = np.array([
    [0.55, 0.15, 0.20, 0.10],
    [0.20, 0.50, 0.20, 0.10],
    [0.25, 0.25, 0.40, 0.10],
    [0.30, 0.20, 0.10, 0.40]
], dtype=float)

def F_mult(fat, F50, p): return 1.0 / (1.0 + np.power(fat / F50, p))
def M_mult(mot, k_m, M_max): return 1.0 + k_m * (mot / M_max)
def H_mult(hor, k_h): return 1.0 + k_h * hor

def compute_resonance(R_max, s_n, S, Fat, Mot, Hor, F50, p, k_m, M_max, k_h):
    return R_max * (1.0 + s_n * S) * F_mult(Fat, F50, p) * M_mult(Mot, k_m, M_max) * H_mult(Hor, k_h)

def normalize_rows(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums

def sample_states_vectorized(probs):
    cumsum = np.cumsum(probs, axis=1)
    r = np.random.rand(probs.shape[0], 1)
    return np.argmax(cumsum >= r, axis=1)

@dataclass
class SimulationConfig:
    N: int = DEFAULTS["N_agents"]
    T: int = DEFAULTS["T_steps"]
    runs: int = DEFAULTS["runs"]
    seed: int = DEFAULTS["seed"]
    R_max: float = DEFAULTS["R_max"]
    s_n: float = DEFAULTS["s_n"]
    k_m: float = DEFAULTS["k_m"]
    k_h: float = DEFAULTS["k_h"]
    M_max: float = DEFAULTS["M_max"]
    F50: float = DEFAULTS["F50"]
    p: float = DEFAULTS["p"]
    alpha: float = DEFAULTS["alpha"]
    beta: float = DEFAULTS["beta"]
    gamma: float = DEFAULTS["gamma"]
    delta: float = DEFAULTS["delta"]
    F_burn: float = DEFAULTS["F_burn"]
    burn_window: int = DEFAULTS["burn_window"]
    B_default: float = DEFAULTS["B_default"]
    out_dir: str = DEFAULTS["out_dir"]

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

class GAM3ARCHSim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def _init_pop(self, rng):
        N = self.cfg.N
        return dict(
            state=rng.choice(4, size=N, p=[0.5, 0.2, 0.2, 0.1]),
            Fat=rng.random(N) * 0.2,
            Mot=rng.random(N) * 0.8,
            Hor=np.zeros(N),
            S=rng.random(N) * 0.3,
            burnout=np.zeros(N, dtype=bool)
        )

    def _compute_Pprime(self, B_matrix):
        return normalize_rows(P_BASE * B_matrix)

    def run_single(self, B_matrix, rng, intervention_at=None, recovery_boost=0.0):
        cfg, N, T = self.cfg, self.cfg.N, self.cfg.T
        P_prime = self._compute_Pprime(B_matrix)
        pop = self._init_pop(rng)
        history = {k: [] for k in ["step", "burnout_incidence", "mean_resonance", "mean_fatigue", "mean_motivation"]}
        fatigue_windows = np.zeros((N, cfg.burn_window))
        window_idx = 0

        for t in range(T):
            in_back = (pop["state"] == STATE_IDX["Back"])
            recovery = in_back.astype(float)
            if intervention_at == t:
                recovery += recovery_boost

            pop["Fat"] = np.clip(pop["Fat"] + cfg.alpha - cfg.beta * recovery + rng.normal(0, 0.02, N), 0, None)
            pop["Mot"] = np.clip(pop["Mot"] + cfg.gamma * rng.random(N) - cfg.delta * pop["Fat"] + rng.normal(0, 0.02, N), 0, None)
            pop["Hor"] = (pop["state"] == STATE_IDX["Horizon"]) * 0.7

            Res = compute_resonance(cfg.R_max, cfg.s_n, pop["S"], pop["Fat"], pop["Mot"], pop["Hor"],
                                    cfg.F50, cfg.p, cfg.k_m, cfg.M_max, cfg.k_h)

            probs = P_prime[pop["state"]]
            pop["state"] = sample_states_vectorized(probs)

            fatigue_windows[:, window_idx % cfg.burn_window] = pop["Fat"]
            window_idx += 1
            if t >= cfg.burn_window:
                burned = fatigue_windows.mean(axis=1) > cfg.F_burn
                pop["burnout"] |= burned

            for k, v in zip(history.keys(), [t, pop["burnout"].mean(), Res.mean(), pop["Fat"].mean(), pop["Mot"].mean()]):
                history[k].append(v)

        return {
            "burnout_abs": float(pop["burnout"].mean()),
            "mean_resonance_end": float(Res.mean()),
            "history": history
        }

    def run_experiment(self, scenarios, save_prefix="exp"):
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        results = []
        for name, params in scenarios.items():
            print(f"[RUN] {name}")
            rng = np.random.default_rng(self.cfg.seed)
            per_run = []
            for r in range(self.cfg.runs):
                run_rng = np.random.default_rng(rng.integers(0, 2**32))
                metrics = self.run_single(
                    params["B"], run_rng,
                    params.get("intervention_at"),
                    params.get("recovery_boost", 0.0)
                )
                per_run.append(metrics)

            burnout = [m["burnout_abs"] for m in per_run]
            resonance = [m["mean_resonance_end"] for m in per_run]
            results.append({
                "scenario": name,
                "burnout_mean": np.mean(burnout),
                "burnout_sem": sem(burnout),
                "resonance_mean": np.mean(resonance),
                "resonance_sem": sem(resonance),
            })

            pd.DataFrame([{"run": i, "burnout": burnout[i], "resonance": resonance[i]} for i in range(len(burnout))]) \
                .to_csv(f"{self.cfg.out_dir}/{save_prefix}_{name}_runs.csv", index=False)

        df = pd.DataFrame(results)
        df.to_csv(f"{self.cfg.out_dir}/{save_prefix}_summary.csv", index=False)
        self._plot(df, save_prefix)
        return df

    def _plot(self, df, prefix):
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].bar(df["scenario"], df["burnout_mean"], yerr=df["burnout_sem"], capsize=5, color='coral')
            ax[0].set_title("Burnout Incidence")
            ax[0].set_ylabel("Mean ± SEM")
            ax[1].errorbar(df["burnout_mean"], df["resonance_mean"],
                           xerr=df["burnout_sem"], yerr=df["resonance_sem"], fmt='o', capsize=5)
            ax[1].set_xlabel("Burnout"); ax[1].set_ylabel("Resonance"); ax[1].set_title("Resonance vs Burnout")
            plt.tight_layout()
            plt.savefig(f"{self.cfg.out_dir}/{prefix}_summary.png", dpi=200)
            plt.close()
            print(f"[INFO] Plot saved: {prefix}_summary.png")
        except ImportError:
            print("[WARN] Matplotlib not found. Skipping plots.")

def build_B(strength): return np.full_like(P_BASE, strength)

def main():
    parser = argparse.ArgumentParser(description="GAM3ARCH v3.1 — Ethical Retention Simulator")
    parser.add_argument("--fast", action="store_true", help="Debug mode")
    args = parser.parse_args()

    cfg = SimulationConfig()
    if args.fast:
        cfg.N = 100; cfg.T = 100; cfg.runs = 2

    sim = GAM3ARCHSim(cfg)

    # === Scenarios from the paper ===
    scenarios = {
        "Baseline": {"B": build_B(1.0)},
        "StrongBridges": {"B": build_B(0.98)},
        "WeakBridges": {"B": (B := np.ones_like(P_BASE) * 0.95, B[0,2] := 0.1, B[3,0] := 0.2, B)[3]},
        "Intervention": {"B": build_B(0.9), "intervention_at": cfg.T // 2, "recovery_boost": 0.2}
    }

    prefix = f"gam3arch_v3_{time.strftime('%Y%m%d_%H%M%S')}"
    df = sim.run_experiment(scenarios, prefix)

    with open(f"{cfg.out_dir}/{prefix}_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n=== RESULTS (paper reproduction) ===")
    print(df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
