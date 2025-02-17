import numpy as np
import pandas as pd
from mapie.regression import MapieTimeSeriesRegressor
from mapie.metrics import (
    regression_coverage_score,
    coverage_width_based,
    regression_mean_width_score,
)

from scintill_ai import ALPHAS, GAP


def enbpi_ts_regressor_predict(
    model,
    cv,
    train_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    alpha_list: list[float] = ALPHAS,
) -> dict:

    results = []
    for alpha in alpha_list:
        ts_regressor = MapieTimeSeriesRegressor(
            model,
            method="enbpi",
            cv=cv,
            agg_function="mean",
            n_jobs=-1,
        )

        ts_regressor.fit(train_data[0], train_data[1])
        y_pred, y_pis = ts_regressor.predict(
            test_data[0],
            alpha=alpha,
            ensemble=True,
        )

        results.append(
            {
                "y_pred": y_pred,
                "y_pis": y_pis,
                "coverage": regression_coverage_score(
                    test_data[1], y_pis[:, 0, 0], y_pis[:, 1, 0]
                ),
                "mean_width": regression_mean_width_score(
                    y_pis[:, 1, 0], y_pis[:, 0, 0]
                ),
                "cwc": coverage_width_based(
                    test_data[1], y_pis[:, 0, 0], y_pis[:, 1, 0], eta=5, alpha=alpha
                ),
                "alpha": alpha,
            }
        )

    return results


def aci_ts_regressor_predict(
    model,
    cv,
    gamma: float,
    train_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    alpha_list: list[float] = ALPHAS,
    gap: int = GAP,
) -> dict:

    results = []
    for alpha in alpha_list:
        ts_regressor = MapieTimeSeriesRegressor(
            model,
            method="aci",
            cv=cv,
            agg_function="mean",
            n_jobs=-1,
        )

        ts_regressor.fit(train_data[0].to_numpy(), train_data[1].to_numpy())

        y_pred = np.zeros(test_data[1].shape)
        y_pis = np.zeros((test_data[1].shape[0], 2, 1))
        y_pred[:gap], y_pis[:gap, :, :] = ts_regressor.predict(
            test_data[0].iloc[:gap, :].to_numpy(),
            alpha=alpha,
            ensemble=True,
            allow_infinite_bounds=True,
        )

        for step in range(gap, test_data[0].shape[0], gap):
            ts_regressor.adapt_conformal_inference(
                test_data[0].iloc[(step - gap) : step, :].to_numpy(),
                test_data[1].iloc[(step - gap) : step].to_numpy(),
                gamma=gamma,
            )

            y_pred[step : step + gap], y_pis[step : step + gap, :, :] = (
                ts_regressor.predict(
                    test_data[0].iloc[step : step + gap, :].to_numpy(),
                    alpha=alpha,
                    ensemble=True,
                    allow_infinite_bounds=True,
                )
            )
            y_pis[step : step + gap, :, :] = np.clip(
                a=y_pis[step : step + gap, :, :],
                a_min=0,
                a_max=1,
            )

        results.append(
            {
                "y_pred": y_pred,
                "y_pis": y_pis,
                "coverage": regression_coverage_score(
                    test_data[1].values, y_pis[:, 0, 0], y_pis[:, 1, 0]
                ),
                "mean_width": regression_mean_width_score(
                    y_pis[:, 1, 0], y_pis[:, 0, 0]
                ),
                "cwc": coverage_width_based(
                    test_data[1], y_pis[:, 0, 0], y_pis[:, 1, 0], eta=5, alpha=alpha
                ),
                "alpha": alpha,
            }
        )

    return results
