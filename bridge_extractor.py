#!/usr/bin/env python3
"""Bridge extractor: infer bridge matrix from telemetry CSV.
CSV format: player_id,timestamp,zone  (timestamp = unix seconds)
"""
import pandas as pd
import numpy as np
import json
import argparse
import os

STATE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]
STATE_IDX = {name: i for i, name in enumerate(STATE_NAMES)}

def extract_bridges(df, T0=60.0):
    df = df.sort_values(['player_id', 'timestamp'])
    transitions = {}
    for _, group in df.groupby('player_id'):
        zones = group['zone'].values
        times = group['timestamp'].values
        for i in range(len(zones)-1):
            if zones[i] != zones[i+1]:
                dt_min = (times[i+1] - times[i]) / 60.0
                key = (zones[i], zones[i+1])
                transitions.setdefault(key, []).append(dt_min)

    B = np.ones((4,4))
    for (f, t), times in transitions.items():
        if f in STATE_IDX and t in STATE_IDX:
            i, j = STATE_IDX[f], STATE_IDX[t]
            T_med = np.median(times)
            B[i,j] = 1.0 / (1.0 + T_med / T0)

    return B

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to telemetry CSV")
    parser.add_argument("--T0", type=float, default=60.0, help="Healthy interval (minutes)")
    parser.add_argument("--out", default="B_matrix.json")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    B = extract_bridges(df, args.T0)
    with open(args.out, "w") as f:
        json.dump(B.tolist(), f, indent=2)

    print("Bridge matrix:")
    print(pd.DataFrame(B, index=STATE_NAMES, columns=STATE_NAMES).round(3))
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
