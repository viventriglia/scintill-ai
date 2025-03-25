from pathlib import Path

import pandas as pd

N_CUT = 9


def read_model_output(path: Path) -> pd.DataFrame:
    df = (
        pd.read_parquet(path, engine="pyarrow")
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["y_pred_hig"], df["y_pred_low"] = df["y_pred_hig"].clip(0), df[
        "y_pred_low"
    ].clip(0)
    return df


def aggregated_evaluation(
    df: pd.DataFrame, n_cut: int = N_CUT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ = df.copy()
    df_["width"] = df_["y_pred_hig"] - df_["y_pred_low"]
    df_["width_category"] = pd.cut(df_["width"], bins=n_cut)
    df_["y_test_category"] = pd.cut(df_["y_test"], bins=n_cut)

    # Aggregate by width size
    df_agg_wc = df_.groupby("width_category", observed=False, as_index=False).agg(
        n_samples=("width", "count"),
        n_covered=("is_covered", "sum"),
        mean_y_test=("y_test", "mean"),
    )
    df_agg_wc["perc_covered"] = (
        100 * df_agg_wc["n_covered"].div(df_agg_wc["n_samples"])
    ).round(1)
    df_agg_wc["width_category"] = df_agg_wc["width_category"].astype(str)

    # Aggregate by <S4>
    df_agg_yc = df_.groupby("y_test_category", observed=False, as_index=False).agg(
        n_samples=("width", "count"),
        n_covered=("is_covered", "sum"),
        mean_width=("width", "mean"),
    )
    df_agg_yc["perc_covered"] = (
        100 * df_agg_yc["n_covered"].div(df_agg_yc["n_samples"])
    ).round(1)
    df_agg_yc["y_test_category"] = df_agg_yc["y_test_category"].astype(str)

    return df_agg_wc, df_agg_yc
