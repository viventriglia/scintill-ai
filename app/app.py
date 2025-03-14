from pathlib import Path

import pandas as pd
import taipy.gui as tpg

from app.plot import plot_pis
from app import DATA_PATH

df_aci = (
    pd.read_parquet(Path(DATA_PATH, "eval_aci.parquet"), engine="pyarrow")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)
df_aci["timestamp"] = pd.to_datetime(df_aci["timestamp"], utc=True)

df_cqr_ct = (
    pd.read_parquet(Path(DATA_PATH, "eval_cqr_ct.parquet"), engine="pyarrow")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)
df_cqr_ct["timestamp"] = pd.to_datetime(df_cqr_ct["timestamp"], utc=True)

df_aci_5_ahead = (
    pd.read_parquet(Path(DATA_PATH, "eval_aci_5_ahead.parquet"), engine="pyarrow")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)
df_aci_5_ahead["timestamp"] = pd.to_datetime(df_aci_5_ahead["timestamp"], utc=True)

df_cqr_ct_5_ahead = (
    pd.read_parquet(Path(DATA_PATH, "eval_cqr_ct_5_ahead.parquet"), engine="pyarrow")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)
df_cqr_ct_5_ahead["timestamp"] = pd.to_datetime(
    df_cqr_ct_5_ahead["timestamp"], utc=True
)

fig_aci = plot_pis(df_aci)
fig_cqr_ct = plot_pis(df_cqr_ct)
fig_aci_5_ahead = plot_pis(df_aci_5_ahead)
fig_cqr_ct_5_ahead = plot_pis(df_cqr_ct_5_ahead)

page = """
# ✨ Scintill-AI

## ⏱️ 1 minute ahead

### CatBoost + Adaptive Conformal Inference (ACI)
<|chart|figure={fig_aci}|>

### CatBoost + Conformalised Quantile Regression (CQR)
<|chart|figure={fig_cqr_ct}|>

## ⏱️ 5 minutes ahead

### CatBoost + Adaptive Conformal Inference (ACI)
<|chart|figure={fig_aci_5_ahead}|>

### CatBoost + Conformalised Quantile Regression (CQR)
<|chart|figure={fig_cqr_ct_5_ahead}|>
"""

if __name__ == "__main__":
    tpg.Gui(page).run(
        title="Scintill-AI",
        dark_mode=True,
        use_reloader=True,
        watermark="",
    )
