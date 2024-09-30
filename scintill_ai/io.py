from pathlib import Path
from io import StringIO
import requests

import pandas as pd
import numpy as np


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
