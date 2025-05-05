import numpy as np
import pandas as pd


def rrmse(y_true, y_pred, y_train, unconstrained: bool = True):
    rmse_test = rmse(y_true, y_pred, unconstrained=unconstrained)
    mean_train = np.mean(y_train)
    return rmse_test / mean_train


def rmse(y_true, y_pred, unconstrained: bool = True):
    if unconstrained:
        mask = y_true != 0.0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def crps(y_pred_quantiles: pd.DataFrame, y_true: pd.Series) -> float:
    """
    Compute the mean CRPS between predicted quantiles and true values.

    Parameters:
        y_pred_quantiles (pd.DataFrame): rows = samples, columns = quantiles from 0.0 to 1.0
        y_true (pd.Series or np.ndarray): true target values, same length as y_pred_quantiles

    Returns:
        float: mean CRPS
    """
    quantiles_sorted = sorted(y_pred_quantiles.columns)
    crps_total = 0.0

    for i in range(len(y_true)):
        y = y_true.iloc[i]
        pred_values = y_pred_quantiles.iloc[i][quantiles_sorted].values.tolist()

        for j in range(len(pred_values) - 1):
            if pred_values[j] <= y <= pred_values[j + 1]:
                x1, x2 = pred_values[j], pred_values[j + 1]
                q1, q2 = quantiles_sorted[j], quantiles_sorted[j + 1]
                if x2 > x1:
                    q_at_y = q1 + (q2 - q1) * (y - x1) / (x2 - x1)
                else:
                    q_at_y = q1
                break
        else:
            q_at_y = 0.0 if y < pred_values[0] else 1.0

        lhs_x, lhs_q = [], []
        for x, q in zip(pred_values, quantiles_sorted):
            if x < y:
                lhs_x.append(x)
                lhs_q.append(q)
            else:
                break
        lhs_x.append(y)
        lhs_q.append(q_at_y)

        rhs_x, rhs_q = [y], [q_at_y]
        for x, q in zip(pred_values, quantiles_sorted):
            if x >= y:
                rhs_x.append(x)
                rhs_q.append(q)

        lhs = sum(
            0.5 * (lhs_x[k + 1] - lhs_x[k]) * (lhs_q[k] ** 2 + lhs_q[k + 1] ** 2)
            for k in range(len(lhs_x) - 1)
        )
        rhs = sum(
            0.5
            * (rhs_x[k + 1] - rhs_x[k])
            * ((1 - rhs_q[k]) ** 2 + (1 - rhs_q[k + 1]) ** 2)
            for k in range(len(rhs_x) - 1)
        )

        crps_total += lhs + rhs

    return crps_total / len(y_true)
