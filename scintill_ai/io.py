from pathlib import Path

import pandas as pd


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

    return pd.concat([pd.concat(df_list) for df_list in data.values()])
