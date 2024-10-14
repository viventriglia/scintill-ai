from pathlib import Path
from io import StringIO
from urllib.parse import quote
import requests
import re
from time import sleep

import pandas as pd
import numpy as np
from dotenv import dotenv_values

from . import ELEVATION_THRESHOLD, LW_S4_THRESHOLD, UP_S4_THRESHOLD
from scintill_ai.utils import progressbar
from scintill_ai.preprocess import preprocess_S4_data


def read_iaga_file(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Convenience function to read time-series data in IAGA-2002 format

    Parameters
    ----------
    file_path : Path
        Input file path

    Returns
    -------
    pd.DataFrame
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("DATE"):
            start_idx = i
            break

    col_names = ["date", "time", "doy", "x", "y", "z", "g"]
    df = pd.read_csv(
        file_path,
        sep="\s+",
        header=None,
        skiprows=start_idx + 1,
        names=col_names,
        **kwargs,
    )

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

    return df.drop(columns=["date", "time", "doy", "g"]).set_index("datetime")


def read_omniweb_file(file_path: Path) -> pd.DataFrame:
    """
    Convenience function to read time-series data from NASA OMNIWeb service

    Parameters
    ----------
    file_path : Path
        Input file path

    Returns
    -------
    pd.DataFrame
    """
    with open(file_path, "r") as f:
        content = f.read()
        content_no_tags = re.sub(r"<.*?>", "", content)
        lines = content_no_tags.strip().split("\n")

    COLS = [
        "year",
        "doy",
        "hour",
        "minute",
        "id_imf_spacecraft",
        "id_swp_spacecraft",
        "field_magnitude_avg",
        "wind_speed",
        "wind_density",
        "wind_pressure",
        "eletric_field",
    ]

    COLWISE_NAN = {
        "id_imf_spacecraft": {99: np.nan},
        "id_swp_spacecraft": {99: np.nan},
        "field_magnitude_avg": {9999.99: np.nan},
        "wind_speed": {99999.9: np.nan},
        "wind_density": {999.99: np.nan},
        "wind_pressure": {99.99: np.nan},
        "eletric_field": {999.99: np.nan},
    }

    # Handling files without data (return empy DataFrame)
    try:
        header_line = next(line for line in lines if line.startswith("YYYY DOY HR MN"))
    except StopIteration:
        return pd.DataFrame()

    data = []
    for line in lines[lines.index(header_line) + 1 :]:
        # Keeping rows that are not empty and begin with a year (4 digits)
        if line.strip() and re.match(r"^\d{4}", line):
            values = line.split()
            data.append(dict(zip(COLS, values)))

    df = pd.DataFrame(data)

    df["datetime"] = pd.to_datetime(
        df["year"]
        + df["doy"].str.pad(3, fillchar="0")
        + df["hour"].str.pad(2, fillchar="0")
        + df["minute"].str.pad(2, fillchar="0"),
        format="%Y%j%H%M",
    )

    df = (
        df.drop(columns=["year", "doy", "hour", "minute"])
        .set_index("datetime")
        .astype(float)
        .replace(COLWISE_NAN)
    )

    return df


def get_magnetometer_data(data_path: Path) -> pd.DataFrame:
    """
    Convenience function to generate a single DataFrame containing the time series of a magnetometer's measurements

    Parameters
    ----------
    data_path : Path
        Input data path (including magnetometer acronym)

    Returns
    -------
    pd.DataFrame
    """
    data = dict()
    station = data_path.stem
    # Get list of years by scanning sub-directories (assuming they're named by year)
    years_list = [int(dir_.stem) for dir_ in data_path.iterdir()]
    years = range(years_list[0], years_list[-1] + 1)

    for yr_ in years:
        year_dir = Path(data_path, str(yr_))

        if not year_dir.exists():
            continue

        # Loop through files matching 'xxxYYYMMDDdmin.min' (per station and year)
        for file_path in year_dir.glob(f"{station.lower()}{yr_}*.min"):
            df = read_iaga_file(file_path, na_values=[99999.00])

            if yr_ not in data:
                data[yr_] = [df]
            else:
                data[yr_].append(df)

    df_mag = pd.concat([pd.concat(df_list) for df_list in data.values()])

    # The H component is the (square root of the) sum of the squares of X and Y components
    df_mag["h"] = np.round(
        np.sqrt(df_mag["x"].pow(2.0) + df_mag["y"].pow(2.0)),
        2,
    )

    return df_mag


def get_solar_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Function that downloads solar indices (F10.7, sunspot number) within a specified
    time interval as collected by GFZ German Research Centre for Geosciences

    Parameters
    ----------
    start_date : str
        Start date of the time interval in 'YYYY-MM-DD' format
    end_date : str
        End date of the time interval in 'YYYY-MM-DD' format

    Returns
    -------
    pd.DataFrame
    """
    response = requests.get(
        "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
    )
    if response.status_code != 200:
        raise Exception(f"Error while downloading data: {response.status_code}")

    data = StringIO(response.text)

    # Skipping the header
    data_lines = []
    for line in data:
        if not line.startswith("#"):
            data_lines.append(line)

    # Creating a DataFrame out of the .txt file
    df = pd.read_csv(
        StringIO("".join(data_lines)),
        sep="\s+",
        header=None,
        usecols=[0, 1, 2, 24, 26],
        names=[
            "year",
            "month",
            "day",
            "ssn",
            "f10.7_adj",
        ],
        na_values=[-1.0],
    )

    df["date"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-"
        + df["day"].astype(str)
    )

    df = df.drop(columns=["year", "month", "day"]).set_index("date")

    return df.loc[start_date:end_date]


def get_solar_wind_data(data_path: Path) -> pd.DataFrame:
    """
    Convenience function to generate a single DataFrame containing the time series
    of OMNIWeb solar wind measurements

    Parameters
    ----------
    data_path : Path
        Input data path

    Returns
    -------
    pd.DataFrame
    """
    data = dict()
    # Get list of years by scanning files in the directory
    years_list = [int(file_.stem) for file_ in data_path.iterdir()]
    years = range(years_list[0], years_list[-1] + 1)

    for yr_ in years:
        year_file = Path(data_path, f"{yr_}.txt")

        if not year_file.is_file():
            continue

        data[yr_] = read_omniweb_file(year_file)

    return pd.concat([data[yr_] for yr_ in years])


def get_gnss_data(
    start: str,
    end: str,
    station_name: str,
    fields: str,
) -> pd.DataFrame:
    """
    Convenience function to read GNSS receivers data from a station in the ISMR network

    Parameters
    ----------
    start : str
        Start date-time in 'YYYY-MM-DD HH:MM:SS' format
    end : str
        End date-time in 'YYYY-MM-DD HH:MM:SS' format
    station_name : str
        Station acronym to be retrived, e.g. 'PRU2' (full list at https://ismrquerytool.fct.unesp.br/is/)
    fields : str
        Columns for a customizable return

    Returns
    -------
    pd.DataFrame
    """
    ISMR_KEY = dotenv_values("../.env.secret")["ISMR_KEY"]
    fields_no_space = ",".join(s_.strip() for s_ in fields.split(","))

    url = f"http://is-cigala-calibra.fct.unesp.br/is/ismrtool/calc-var/service_loadISMR.php"
    url += f"?date_begin={quote(start)}&date_end={quote(end)}&stationName={station_name.strip()}&field_list={fields_no_space}&mode=csv&key={quote(ISMR_KEY.strip())}"

    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        # print(e)
        return pd.DataFrame(columns=[f_.strip() for f_ in fields.split(",")])


def get_aggregated_gnss_data(
    start: str,
    end: str,
    station_name: str,
    fields: str,
) -> pd.DataFrame:
    """
    Convenience function to get GNSS aggregated data (mean and max S4 for satellites
    above a set elevation, e.g. 60°) from a station in the ISMR network, while avoiding
    hitting servers with massive requests

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format (day starts at 00:00)
    end : str
        End date in 'YYYY-MM-DD' format (day ends at 23:59)
    station_name : str
        Station acronym to be retrived, e.g. 'PRU2' (full list at https://ismrquerytool.fct.unesp.br/is/)
    fields : str
        Columns for a customizable return

    Returns
    -------
    pd.DataFrame
    """
    date_range = pd.date_range(start, end)
    dfs = []
    for dt_ in progressbar(date_range, prefix="Downloading -- time for a ☕ "):
        dt_begin = dt_.strftime("%Y-%m-%d 00:00:00")
        dt_end = dt_.strftime("%Y-%m-%d 23:59:00")

        df_raw = get_gnss_data(dt_begin, dt_end, station_name, fields)
        dfs.append(
            preprocess_S4_data(
                df=df_raw,
                elevation_threshold=ELEVATION_THRESHOLD,
                lower_S4_threshold=LW_S4_THRESHOLD,
                higher_S4_threshold=UP_S4_THRESHOLD,
            )
        )
        # Let's wait a bit between requests
        sleep(np.random.random() / 2)

    return pd.concat(dfs, ignore_index=True)
