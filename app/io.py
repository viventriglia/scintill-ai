from pathlib import Path

import pandas as pd


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
