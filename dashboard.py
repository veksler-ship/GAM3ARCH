import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from gam3arch_v3 import GAM3ARCHSim, SimulationConfig, build_B, P_BASE

st.set_page_config(page_title="GAM3ARCH", layout="wide")
st.title("GAM3ARCH — Ethical Retention Framework (paper reproduction)")
tabs = st.tabs(["Bridge Extractor", "Simulation", "Results"])
STATE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]

with tabs[0]:
    st.header("Extract B matrix from telemetry")
    uploaded = st.file_uploader("CSV with columns: player_id,timestamp,zone", type=["csv"])
    T0 = st.slider("T0 (healthy interval, minutes)", 10, 480, 60)
    if uploaded is not None and st.button("Analyze"):
        df = pd.read_csv(uploaded)
        from bridge_extractor import extract_bridges
        B = extract_bridges(df, T0)
        st.success("Done")
        st.dataframe(pd.DataFrame(B, index=STATE_NAMES, columns=STATE_NAMES).round(3))
        st.download_button("Download B_matrix.json", json.dumps(B.tolist()), "B_matrix.json")

with tabs[1]:
    st.header("Run simulation (paper presets)")
    scenario = st.selectbox("Scenario", ["Baseline", "WeakBridges", "Intervention", "StrongBridges"])
    runs = st.slider("Monte Carlo runs", 1, 50, 10)
    if st.button("Run"):
        cfg = SimulationConfig()
        cfg.runs = runs
        if scenario == "WeakBridges":
            B = np.ones_like(P_BASE) * 0.95
            B[0,2] = 0.1; B[3,0] = 0.2
            intervention_at = None; recovery_boost = 0.0
        elif scenario == "Intervention":
            B = build_B(0.9)
            intervention_at = cfg.T // 2; recovery_boost = 0.2
        elif scenario == "StrongBridges":
            B = build_B(0.98)
            intervention_at = None; recovery_boost = 0.0
        else:
            B = build_B(1.0); intervention_at = None; recovery_boost = 0.0

        sim = GAM3ARCHSim(cfg)
        scen = {scenario: {"B": B, "intervention_at": intervention_at, "recovery_boost": recovery_boost}}
        df = sim.run_experiment(scen, "dashboard")
        st.success("Simulation complete")
        st.dataframe(df.round(4))

with tabs[2]:
    st.header("Latest results")
    if os.path.exists("results"):
        files = [f for f in os.listdir("results") if f.endswith("_summary.csv")]
        if files:
            latest = sorted(files)[-1]
            df = pd.read_csv(os.path.join("results", latest))
            st.dataframe(df.round(4))
            st.bar_chart(df.set_index("scenario")["burnout_mean"])
