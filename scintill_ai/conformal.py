from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import RegressorMixin
from mapie.regression import MapieRegressor, MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from mapie.metrics import (
    regression_coverage_score,
    coverage_width_based,
    regression_mean_width_score,
)

from scintill_ai import ALPHAS, FCST_HORIZON, GAMMA


def enbpi_ts_regressor_predict(
    model: RegressorMixin,
    cv: BlockBootstrap,
    train_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    alpha_list: list[float] = ALPHAS,
) -> dict:

    ts_regressor = fit_mapie_regressor(
        model=model, method="enbpi", cv=cv, train_data=train_data
    )

    results = []
    for alpha in alpha_list:
        y_pred, y_pis = ts_regressor.predict(
            test_data[0].to_numpy(),
            alpha=alpha,
            ensemble=True,
        )

        results.append(
            {
                "y_pred": y_pred,
                "y_pis": y_pis,
                **evaluate_metrics(test_data, y_pis, alpha),
                "alpha": alpha,
            }
        )

    return results


def aci_ts_regressor_predict(
    model: RegressorMixin,
    cv: BlockBootstrap,
    train_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    update_calibration: bool = False,
    alpha_list: list[float] = ALPHAS,
    forecast_horizon: int = FCST_HORIZON,
    gamma: float = GAMMA,
) -> dict:

    ts_regressor = fit_mapie_regressor(
        model=model, method="aci", cv=cv, train_data=train_data
    )

    results = []
    for alpha in alpha_list:
        y_pred = np.zeros(test_data[1].shape)
        y_pis = np.zeros((test_data[1].shape[0], 2, 1))
        y_pred[:forecast_horizon], y_pis[:forecast_horizon, :, :] = (
            ts_regressor.predict(
                test_data[0].iloc[:forecast_horizon, :].to_numpy(),
                alpha=alpha,
                ensemble=True,
                allow_infinite_bounds=True,
            )
        )

        for step in range(forecast_horizon, test_data[0].shape[0], forecast_horizon):
            if update_calibration:
                ts_regressor.partial_fit(
                    test_data[0].iloc[(step - forecast_horizon) : step, :].to_numpy(),
                    test_data[1].iloc[(step - forecast_horizon) : step].to_numpy(),
                )
            ts_regressor.adapt_conformal_inference(
                test_data[0].iloc[(step - forecast_horizon) : step, :].to_numpy(),
                test_data[1].iloc[(step - forecast_horizon) : step].to_numpy(),
                gamma=gamma,
            )

            (
                y_pred[step : step + forecast_horizon],
                y_pis[step : step + forecast_horizon, :, :],
            ) = ts_regressor.predict(
                test_data[0].iloc[step : step + forecast_horizon, :].to_numpy(),
                alpha=alpha,
                ensemble=True,
                allow_infinite_bounds=True,
            )
            y_pis[step : step + forecast_horizon, :, :] = np.clip(
                a=y_pis[step : step + forecast_horizon, :, :],
                a_min=0,
                a_max=1.2,
            )

        results.append(
            {
                "y_pred": y_pred,
                "y_pis": y_pis,
                **evaluate_metrics(test_data, y_pis, alpha),
                "alpha": alpha,
            }
        )

    return results


def evaluate_metrics(
    test_data: tuple[pd.DataFrame, pd.Series], y_pis: NDArray[np.float64], alpha: float
):
    return {
        "coverage": regression_coverage_score(
            test_data[1].values, y_pis[:, 0, 0], y_pis[:, 1, 0]
        ),
        "mean_width": regression_mean_width_score(y_pis[:, 1, 0], y_pis[:, 0, 0]),
        "cwc": coverage_width_based(
            test_data[1], y_pis[:, 0, 0], y_pis[:, 1, 0], eta=5, alpha=alpha
        ),
    }


def fit_mapie_regressor(
    model: RegressorMixin,
    method: Literal["enbpi", "aci"],
    cv: BlockBootstrap,
    train_data: tuple[pd.DataFrame, pd.Series],
) -> MapieRegressor:
    return MapieTimeSeriesRegressor(
        model,
        method=method,
        cv=cv,
        agg_function="mean",
        n_jobs=-1,
    ).fit(train_data[0].to_numpy(), train_data[1].to_numpy())
