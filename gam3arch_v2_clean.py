# GAM3ARCH v2 — Ethical Game Cycle and Player Burnout Model (Clean Edition)
# Author: GAM3ARCH Initiative (2025)
# License: MIT
# ------------------------------------------------------------
# This version removes cultural/authority bias factors and focuses
# on systemic, player-centered ethical cycles (Forge / Nexus / Back / Horizon).

import random
import pandas as pd

STATES = ["Forge", "Nexus", "Back", "Horizon"]
N_PLAYERS = 500
N_STEPS = 100

BASE_TRANSITIONS = {
    "Forge": {"Nexus": 0.40, "Back": 0.20, "Forge": 0.25, "Horizon": 0.15},
    "Nexus": {"Forge": 0.20, "Back": 0.10, "Nexus": 0.50, "Horizon": 0.20},
    "Back": {"Forge": 0.25, "Nexus": 0.35, "Back": 0.30, "Horizon": 0.10},
    "Horizon": {"Forge": 0.60, "Nexus": 0.10, "Back": 0.10, "Horizon": 0.20},
}

def scenario_definitions():
    scenarios = {}
    scenarios["Baseline"] = {}
    scenarios["StrongBridges"] = {
        "base_transitions": {
            "Forge": {"Nexus": 0.55, "Back": 0.25, "Forge": 0.10, "Horizon": 0.10},
            "Nexus": {"Forge": 0.20, "Back": 0.10, "Nexus": 0.50, "Horizon": 0.20},
            "Back": {"Forge": 0.25, "Nexus": 0.35, "Back": 0.30, "Horizon": 0.10},
            "Horizon": {"Forge": 0.60, "Nexus": 0.10, "Back": 0.10, "Horizon": 0.20},
        }
    }
    scenarios["WeakBridges"] = {
        "base_transitions": {
            "Forge": {"Nexus": 0.10, "Back": 0.05, "Forge": 0.70, "Horizon": 0.15},
            "Nexus": {"Forge": 0.20, "Back": 0.10, "Nexus": 0.50, "Horizon": 0.20},
            "Back": {"Forge": 0.25, "Nexus": 0.35, "Back": 0.30, "Horizon": 0.10},
            "Horizon": {"Forge": 0.60, "Nexus": 0.10, "Back": 0.10, "Horizon": 0.20},
        }
    }
    scenarios["Intervention"] = {"fomo_event_prob": 0.02, "fomo_delta": 0.12}
    return scenarios

def simulate(scenario, steps=N_STEPS, players=N_PLAYERS):
    transitions = scenario.get("base_transitions", BASE_TRANSITIONS)
    fomo_prob = scenario.get("fomo_event_prob", 0.05)
    fomo_delta = scenario.get("fomo_delta", 0.25)
    data = []
    states = [random.choice(STATES) for _ in range(players)]
    for step in range(steps):
        for i, state in enumerate(states):
            current = transitions[state].copy()
            if random.random() < fomo_prob:
                current["Forge"] = min(1.0, current["Forge"] + fomo_delta)
            next_state = random.choices(list(current.keys()), weights=current.values())[0]
            states[i] = next_state
            data.append({"step": step, "player": i, "state": next_state})
    return pd.DataFrame(data)

def run_all():
    scenarios = scenario_definitions()
    results = {}
    for name, scenario in scenarios.items():
        df = simulate(scenario)
        results[name] = df
        df.to_csv(f"results_{name}.csv", index=False)
    return results

if __name__ == "__main__":
    print("Running GAM3ARCH v2 clean simulation...")
    data = run_all()
    print("Done. CSV files saved to current directory.")
