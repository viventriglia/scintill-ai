from pathlib import Path
import taipy.gui as tpg

from app.plot import plot_pis
from app.io import read_model_output, aggregated_evaluation
from app import DATA_PATH

df_aci_1 = read_model_output(Path(DATA_PATH, "eval_aci.parquet"))
df_cqr_1 = read_model_output(Path(DATA_PATH, "eval_cqr_ct.parquet"))
df_aci_5 = read_model_output(Path(DATA_PATH, "eval_aci_5_ahead.parquet"))
df_cqr_5 = read_model_output(Path(DATA_PATH, "eval_cqr_ct_5_ahead.parquet"))

df_aci_1_wc, df_aci_1_yc = aggregated_evaluation(df_aci_1)
df_cqr_1_wc, df_cqr_1_yc = aggregated_evaluation(df_cqr_1)
df_aci_5_wc, df_aci_5_yc = aggregated_evaluation(df_aci_5)
df_cqr_5_wc, df_cqr_5_yc = aggregated_evaluation(df_cqr_5)

fig_aci_1 = plot_pis(df_aci_1)
fig_cqr_1 = plot_pis(df_cqr_1)
fig_aci_5 = plot_pis(df_aci_5)
fig_cqr_5 = plot_pis(df_cqr_5)

chart_1min = """
## CatBoost + Adaptive Conformal Inference (ACI)
<|chart|figure={fig_aci_1}|>

## CatBoost + Conformalised Quantile Regression (CQR)
<|chart|figure={fig_cqr_1}|>
"""

chart_5min = """
## CatBoost + Adaptive Conformal Inference (ACI)
<|chart|figure={fig_aci_5}|>

## CatBoost + Conformalised Quantile Regression (CQR)
<|chart|figure={fig_cqr_5}|>
"""

metric_1min = """
## CatBoost + Adaptive Conformal Inference (ACI)

<|Aggregated by PI width|expandable|expanded=False|
<|{df_aci_1_wc}|table|show_all|width=100%|>
|>

<|Aggregated by <S4>|expandable|expanded=False|
<|{df_aci_1_yc}|table|show_all|width=100%|>
|>

## CatBoost + Conformalised Quantile Regression (CQR)

<|Aggregated by PI width|expandable|expanded=False|
<|{df_cqr_1_wc}|table|show_all|width=100%|>
|>

<|Aggregated by <S4>|expandable|expanded=False|
<|{df_cqr_1_yc}|table|show_all|width=100%|>
|>
"""

metric_5min = """
## CatBoost + Adaptive Conformal Inference (ACI)

<|Aggregated by PI width|expandable|expanded=False|
<|{df_aci_5_wc}|table|show_all|width=100%|>
|>

<|Aggregated by <S4>|expandable|expanded=False|
<|{df_aci_5_yc}|table|show_all|width=100%|>
|>

## CatBoost + Conformalised Quantile Regression (CQR)

<|Aggregated by PI width|expandable|expanded=False|
<|{df_cqr_5_wc}|table|show_all|width=100%|>
|>

<|Aggregated by <S4>|expandable|expanded=False|
<|{df_cqr_5_yc}|table|show_all|width=100%|>
|>
"""

pages = {
    "/": """<|navbar|lov={[("/1-min-ahead-chart", "1-min ahead 📈"), ("/1-min-ahead-metric", "1-min ahead 🔎"), ("/5-min-ahead-chart", "5-min ahead 📈"), ("/5-min-ahead-metric", "5-min ahead 🔎")]}|>""",
    "1-min-ahead-chart": chart_1min,
    "1-min-ahead-metric": metric_1min,
    "5-min-ahead-chart": chart_5min,
    "5-min-ahead-metric": metric_5min,
}

if __name__ == "__main__":
    gui = tpg.Gui(pages=pages)
    gui.run(
        title="Scintill-AI",
        dark_mode=True,
        use_reloader=True,
        watermark="",
    )
