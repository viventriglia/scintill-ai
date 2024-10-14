from typing import Literal

import pandas as pd
import numpy as np
import pvlib

from . import LATITUDE, LONGITUDE, ALTITUDE


def get_solar_position(
    time: pd.DatetimeIndex,
    columns: Literal[
        "apparent_zenith",
        "zenith",
        "apparent_elevation",
        "elevation",
        "azimuth",
        "equation_of_time",
    ] = ["zenith"],
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    altitude: float = ALTITUDE,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper for solar position data

    Parameters
    ----------
    time : pd.DatetimeIndex
        Must be localized or UTC will be assumed
    columns : list[str], optional
        Solar position attributes to return, by default ["zenith"]
    latitude : float, optional
        Latitude in decimal degrees; positive north of equator, negative to south
    longitude : float, optional
        Longitude in decimal degrees; positive east of prime meridian, negative to west
    altitude : float, optional
        Altitude in metres
    **kwargs
        Other keywords to be passed to the underlying solar position function

    Returns
    -------
    pd.DataFrame
    """
    return pvlib.solarposition.get_solarposition(
        time=time,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        **kwargs,
    )[columns]


def preprocess_S4_data(
    df: pd.DataFrame,
    elevation_threshold: float,
    lower_S4_threshold: float,
    higher_S4_threshold: float,
) -> pd.DataFrame:
    df["s4_denoised"] = denoise_S4(df["s4"], df["s4_correction"])

    df_high_elev = filter_higher_elevs(df, elevation_threshold)

    df_high_elev["is_mild_scint"] = np.where(
        df_high_elev["s4_denoised"].ge(lower_S4_threshold),
        True,
        False,
    )

    df_high_elev["is_strong_scint"] = np.where(
        df_high_elev["s4_denoised"].ge(higher_S4_threshold),
        True,
        False,
    )

    df_agg = df_high_elev.groupby(
        ["time_utc"],
        as_index=False,
    ).agg(
        n_sat=("svid", "nunique"),
        n_sat_mild_scint=("is_mild_scint", "sum"),
        n_sat_strong_scint=("is_strong_scint", "sum"),
        s4_max=("s4_denoised", "max"),
        s4_mean=("s4_denoised", "mean"),
    )

    df_agg["perc_mild_scint"] = np.round(
        df_agg["n_sat_mild_scint"].div(df_agg["n_sat"]), 3
    )
    df_agg["perc_strong_scint"] = np.round(
        df_agg["n_sat_strong_scint"].div(df_agg["n_sat"]), 3
    )

    return df_agg


def denoise_S4(s4_series: pd.Series, s4_corr_series: pd.Series) -> np.ndarray:
    """
    _summary_

    Parameters
    ----------
    s4_series : pd.Series
        _description_
    s4_corr_series : pd.Series
        _description_

    Returns
    -------
    np.ndarray
    """
    return np.round(
        np.emath.sqrt(s4_series.pow(2.0) - s4_corr_series.pow(2.0)).real,
        3,
    )


def filter_higher_elevs(df: pd.DataFrame, elevation_threshold: float) -> pd.DataFrame:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    elevation_threshold : float
        _description_

    Returns
    -------
    pd.DataFrame
    """
    df_ = df.copy()
    df_["is_high_elev"] = np.where(
        df_["elev"].ge(elevation_threshold),
        True,
        False,
    )

    return (
        df_[df_["is_high_elev"].eq(True)]
        .drop(columns="is_high_elev")
        .reset_index(drop=True)
    )
