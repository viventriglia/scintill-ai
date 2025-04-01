from typing import Literal, Iterable

import pandas as pd
import numpy as np
import pvlib
from sklearn.cluster import KMeans

from . import LATITUDE, LONGITUDE, ALTITUDE, H_START, H_STOP


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
    """
    Preprocesses S4 data by denoising, filtering by elevation, and labelling scintillation levels.
    This utility computes also some satellite statistics (i.e., percentage of satellites with
    mild or strong scintillation, maximum and mean registered S4 values).

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    elevation_threshold : float
        Minimum elevation for the rows to be retained
    lower_S4_threshold : float
        S4 value above which the scintillation is considered mild
    higher_S4_threshold : float
        S4 value above which the scintillation is considered strong

    Returns
    -------
    pd.DataFrame
    """
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


@np.errstate(invalid="ignore")
def denoise_S4(s4_series: pd.Series, s4_corr_series: pd.Series) -> np.ndarray:
    """
    Denoises a given S4 signal by removing the thermal noise component

    Parameters
    ----------
    s4_series : pd.Series
        Series containing the S4 signal values
    s4_corr_series : pd.Series
        Series containing the corrections to be subtracted

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
    Filter data to include only rows where the elevation is greater than or equal to
    a specified threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input data, with an 'elev' column representing the elevation
    elevation_threshold : float
        Minimum elevation for the rows to be retained

    Returns
    -------
    pd.DataFrame
    """
    return df[df["elev"].ge(elevation_threshold)].reset_index(drop=True)


def get_categories(
    series: pd.Series, window: int = 10, n_categories: int = 3, zero_phase: bool = True
) -> tuple[pd.Series, np.ndarray]:
    """
    Convenience function which filters the time series with a exponentially-weighted
    moving average (EMA) or with a forward-backward (FB) EMA (if `zero_phase` is set
    to True); the function then fits a K-Means algorithm and returns the smoothed
    values along with the estimated labels (categories)

    Parameters
    ----------
    series : pd.Series
        Time series to filter and categorise
    window : int, optional
        Time window steps for smoothing, by default 10
    n_categories : int, optional
        Number of categories to extract, by default 3
    zero_phase : bool, optional
        Whether or not to make the filter zero-phase (i.e., a non-causal filter),
        by default True; if the filter is zero-phase, the smoothed series is not
        appropriate for prediction due to data leakage from future values

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        Smoothed series, estimated labels (categories)
    """
    filtered_series = series.ewm(span=window).mean()

    if zero_phase:
        # Backward filtering as well
        filtered_series = filtered_series[::-1].ewm(span=window).mean()[::-1]

    zeroes = filtered_series.lt(0).sum() > 0

    # Evaluate log-differences (with an offset in case of negative values)
    if not zeroes:
        log_diff = np.diff(np.log1p(filtered_series.values))
    else:
        log_diff = np.diff(
            np.log1p(filtered_series.values + abs(filtered_series.min()))
        )

    # Fit the clustering model
    km = KMeans(n_clusters=n_categories, n_init="auto", random_state=42).fit(
        log_diff.reshape(-1, 1)
    )
    lb = km.labels_

    # Change the labels to get some semblance of order
    cluster_centers = km.cluster_centers_.flatten()
    temp = [(cluster_centers[i], i) for i in range(n_categories)]
    temp = sorted(temp, key=lambda x: x[0])

    labels = np.zeros(len(lb), dtype=int)
    for i in range(1, n_categories):
        old_lb = temp[i][1]
        idx = np.where(lb == old_lb)[0]
        labels[idx] = i

    return filtered_series, labels


def get_time_filtering_and_features(
    df: pd.DataFrame,
    hour_start: int = H_START,
    hour_stop: int = H_STOP,
    ema_cols: dict[str, Iterable[int]] = None,
    lag_cols: dict[str, Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Filters the input DataFrame by time and adds exponential moving average (EMA) and lagged features

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a datetime index
    hour_start : int, optional
        Starting hour for filtering, by default 20
        The function filters out rows where the hour is before this value
    hour_stop : int, optional
        Ending hour for filtering, by default 6
        The function filters out rows where the hour is after this value
    ema_cols : dict[str, Iterable[int]], optional
        Dictionary where keys are column names and values are lists of integers specifying
        the windows (in minutes) for which to compute the EMAs; if not provided, no EMAs
        are computed
    lag_cols : dict[str, Iterable[int]], optional
        Dictionary where keys are column names and values are lists of integers specifying
        the time lags (in minutes) for which to compute theri lagged versions; if not provided,
        no lag features are computed

    Returns
    -------
    pd.DataFrame
        Time-filtered DataFrame with the added EMA and/or lag features, if any
    """
    max_window = max(
        max([max(v) for v in ema_cols.values()], default=0) if ema_cols else 0,
        max([max(v) for v in lag_cols.values()], default=0) if lag_cols else 0,
    )
    hrs_, mins_ = divmod(max_window, 60)

    # Pre-filtering
    if hour_start > hour_stop:
        pre_filter = (
            (df.index.hour > hour_start - 1)
            | (df.index.hour < hour_stop)
            | (
                (df.index.hour == (hour_start - 1 - hrs_))
                & (df.index.minute >= (60 - mins_))
            )
        )
    else:
        pre_filter = (df.index.hour >= (hour_start - hrs_)) & (
            df.index.hour < hour_stop
        )

    df = df[pre_filter].copy()

    # EMAs
    if ema_cols is not None:
        for col, windows in ema_cols.items():
            for w in windows:
                df[f"{col}_ema_{w}m"] = df[col].ewm(span=w).mean()

    # Lags
    if lag_cols is not None:
        for col, windows in lag_cols.items():
            for w in windows:
                df[f"{col}_lag_{w}m"] = df[col].shift(w)

    # Filtering
    if hour_start > hour_stop:
        df = df[(df.index.hour >= hour_start) | (df.index.hour < hour_stop)]
    else:
        df = df[(df.index.hour >= hour_start) & (df.index.hour < hour_stop)]

    return df
