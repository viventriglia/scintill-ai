from typing import Literal

import pandas as pd
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
