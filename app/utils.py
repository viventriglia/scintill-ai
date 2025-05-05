from typing import Union
import pandas as pd

from app import SCINT_THR


def evaluate_metrics(df_eval: pd.DataFrame) -> dict[str, Union[float, None]]:
    mask_nonzero = df_eval["y_test"] != 0

    # Marginal coverage
    marg_cov = df_eval.loc[mask_nonzero, "is_covered"].sum() / mask_nonzero.sum()

    # Conditional coverage
    mask_thr = df_eval["y_test"].gt(SCINT_THR) & mask_nonzero
    mask_cond = mask_thr & (df_eval["y_test"] <= df_eval["y_pred_hig"])
    cond_cov = (
        df_eval[mask_cond].shape[0] / mask_thr.sum() if mask_thr.sum() > 0 else None
    )

    # Mean prediction interval width
    mean_width = (
        df_eval.loc[mask_nonzero, "y_pred_hig"]
        - df_eval.loc[mask_nonzero, "y_pred_low"]
    ).mean()

    return {
        "marg_cov": marg_cov,
        "cond_cov": cond_cov,
        "mean_width": mean_width,
    }
