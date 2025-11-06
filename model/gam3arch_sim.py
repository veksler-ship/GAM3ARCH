import random

def run_simulation(steps=500):
    zones = ["Forge", "Nexus", "Back", "Horizon"]
    transitions = {
        "Forge": {"Nexus": 0.3, "Back": 0.1},
        "Nexus": {"Back": 0.6},
        "Back": {"Horizon": 0.8},
        "Horizon": {"Forge": 0.6}
    }

    fatigue = 0.0
    current = "Forge"
    history = []

    for step in range(steps):
        fatigue += 0.02 if current == "Forge" else -0.01
        fatigue = max(0, min(1, fatigue))

        if fatigue > 0.8:
            current = "Back"
        else:
            next_zones = transitions.get(current, {})
            if next_zones:
                current = random.choices(list(next_zones.keys()), weights=next_zones.values())[0]

        history.append((current, fatigue))

    burnout_steps = sum(1 for z, f in history if f > 0.7)
    print(f"Шагов в состоянии выгорания: {burnout_steps}/{steps}")
    return history
