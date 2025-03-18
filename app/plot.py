import plotly.graph_objects as go
import pandas as pd

from app import SCINT_THR


def plot_pis(df_eval: pd.DataFrame) -> go.Figure:
    out_of_pi = df_eval[df_eval["is_covered"].eq(False)]
    marg_cov = df_eval["is_covered"].sum() / df_eval.shape[0]
    cond_cov = (
        df_eval[
            (df_eval["y_test"].gt(SCINT_THR))
            & (df_eval["y_test"].le(df_eval["y_pred_hig"]))
        ].shape[0]
        / df_eval[(df_eval["y_test"].gt(SCINT_THR))].shape[0]
    )
    mean_width = (df_eval["y_pred_hig"] - df_eval["y_pred_low"]).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_eval["timestamp"],
            y=df_eval["y_test"],
            mode="lines",
            name="<S4> (actual)",
            line=dict(color="cornsilk"),
            hovertemplate="%{y:.2f}<extra>Actual</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_eval["timestamp"],
            y=df_eval["y_pred_hig"],
            fill=None,
            mode="lines",
            line=dict(color="deepskyblue"),
            hovertemplate="%{y:.2f}<extra>PI ↑</extra>",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_eval["timestamp"],
            y=df_eval["y_pred_low"],
            fill="tonexty",
            mode="lines",
            line=dict(color="deepskyblue"),
            opacity=0.2,
            hovertemplate="%{y:.2f}<extra>PI ↓</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=out_of_pi["timestamp"],
            y=out_of_pi["y_test"],
            mode="markers",
            marker=dict(color="red"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"Marginal coverage: <b>{marg_cov:.1%}</b> | Conditional coverage (above {SCINT_THR}): <b>{cond_cov:.1%}</b> | Mean interval width: <b>{mean_width:.2f}</b>",
        xaxis_title="",
        yaxis_title="<S4>",
        template="plotly_dark",
        hovermode="x",
        height=550,
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
            visible=False,
        ),
    )

    return fig
