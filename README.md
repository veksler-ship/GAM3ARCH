# GAM3ARCH — Ethical Retention Framework (v3.1, paper reproduction)

This repository contains the code used to reproduce the simulations reported in the GAM3ARCH preprint (Nov 2025).
The "paper" preset (default constants in gam3arch_v3.py) reproduces the numbers presented in the publication.

## Quick start

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run simulation:
   ```
   python gam3arch_v3.py
   ```

3. Extract bridge matrix from sample telemetry:
   ```
   python bridge_extractor.py examples/sample_telemetry.csv
   ```

4. Run dashboard:
   ```
   streamlit run dashboard.py
   ```

Results (CSV + PNG) are written to the `results/` directory.
