from pathlib import Path

import taipy.gui as tpg

from app.plot import plot_pis
from app.io import read_model_output
from app import DATA_PATH

df_aci = read_model_output(Path(DATA_PATH, "eval_aci.parquet"))
df_cqr_ct = read_model_output(Path(DATA_PATH, "eval_cqr_ct.parquet"))
df_aci_5_ahead = read_model_output(Path(DATA_PATH, "eval_aci_5_ahead.parquet"))
df_cqr_ct_5_ahead = read_model_output(Path(DATA_PATH, "eval_cqr_ct_5_ahead.parquet"))

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
