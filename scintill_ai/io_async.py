from pathlib import Path
import io
import asyncio
from urllib.parse import quote
from datetime import datetime, timedelta

import aiohttp
import pandas as pd

from scintill_ai.preprocess import preprocess_S4_data
from scintill_ai import (
    ELEVATION_THRESHOLD,
    LW_S4_THRESHOLD,
    UP_S4_THRESHOLD,
    ISMR_KEY,
    MAX_CONCURRENT_REQUESTS,
)


async def get_gnss_data_async(
    session: aiohttp.ClientSession, start: str, end: str, station_name: str, fields: str
) -> str:
    """
    Asynchronous function to fetch GNSS data from a station in the ISMR network

    Parameters
    ----------
    session : aiohttp.ClientSession
        Session object used for asynchronous HTTP requests
    start : str
        Start date-time in 'YYYY-MM-DD HH:MM:SS' format
    end : str
        End date-time in 'YYYY-MM-DD HH:MM:SS' format
    station_name : str
        Station acronym to retrieve, e.g., 'PRU2' (full list available at https://ismrquerytool.fct.unesp.br/is/)
    fields : str
        Columns for a customizable return, specified as a comma-separated string

    Returns
    -------
    str
        Response text containing the requested GNSS data in CSV format
    """
    fields_no_space = ",".join(s_.strip() for s_ in fields.split(","))
    url = f"http://is-cigala-calibra.fct.unesp.br/is/ismrtool/calc-var/service_loadISMR.php"
    url += f"?date_begin={start}&date_end={end}&stationName={station_name.strip()}&field_list={fields_no_space}&mode=csv&key={quote(ISMR_KEY.strip())}"

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with semaphore:
        print(f"{start} -> {end}")
        async with session.get(url) as response:
            return await response.text()


async def process_day_data_async(
    session: aiohttp.ClientSession,
    dt_begin: str,
    dt_end: str,
    station_name: str,
    fields: str,
) -> pd.DataFrame:
    """
    Process GNSS data for a single day asynchronously

    Parameters
    ----------
    session : aiohttp.ClientSession
        Session object used for asynchronous HTTP requests
    dt_begin : str
        Start date-time in 'YYYY-MM-DD HH:MM:SS' format
    dt_end : str
        End date-time in 'YYYY-MM-DD HH:MM:SS' format
    station_name : str
        Station acronym to retrieve, e.g., 'PRU2' (full list available at https://ismrquerytool.fct.unesp.br/is/)
    fields : str
        Columns for a customizable return, specified as a comma-separated string

    Returns
    -------
    pd.DataFrame
    """
    raw_data = await get_gnss_data_async(
        session, dt_begin, dt_end, station_name, fields
    )
    df_raw = pd.read_csv(io.StringIO(raw_data))
    return preprocess_S4_data(
        df=df_raw,
        elevation_threshold=ELEVATION_THRESHOLD,
        lower_S4_threshold=LW_S4_THRESHOLD,
        higher_S4_threshold=UP_S4_THRESHOLD,
    )


async def get_aggregated_gnss_data_async(
    start: str, end: str, station_name: str, fields: str
) -> pd.DataFrame:
    """
    Asynchronous function to get aggregated GNSS data over a specified range of dates

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

    async with aiohttp.ClientSession() as session:
        # Create a list of coroutines for each day
        tasks = [
            process_day_data_async(
                session,
                dt_.strftime("%Y-%m-%d 00:00:00"),
                dt_.strftime("%Y-%m-%d 23:59:00"),
                station_name,
                fields,
            )
            for dt_ in date_range
        ]

        # Run all coroutines concurrently
        results = await asyncio.gather(*tasks)

        for result in results:
            dfs.append(result)

    return pd.concat(dfs, ignore_index=True)


async def get_aggregated_gnss_data_by_month_async(
    start: str, end: str, station_name: str, fields: str, output_dir: str
) -> None:
    """
    Download GNSS data aggregated asynchronously month by month, then save each
    month in a 'YYYY_MM.pickle' file

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
    output_dir: str
        Directory where to store the monthly pickle files
    """

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        while start_date <= end_date:
            month_end = (start_date.replace(day=28) + timedelta(days=4)).replace(
                day=1
            ) - timedelta(days=1)
            if month_end > end_date:
                month_end = end_date

            date_range = pd.date_range(start_date, month_end)

            tasks = [
                process_day_data_async(
                    session,
                    dt_.strftime("%Y-%m-%d 00:00:00"),
                    dt_.strftime("%Y-%m-%d 23:59:00"),
                    station_name,
                    fields,
                )
                for dt_ in date_range
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter errors and merge data
            dfs = [res for res in results if isinstance(res, pd.DataFrame)]
            if dfs:
                df_month = pd.concat(dfs, ignore_index=True)

                pickle_file = (
                    output_path / f"{start_date.year}_{start_date.month:02d}.pickle"
                )
                df_month.to_pickle(pickle_file)

                print(f"Saved: {pickle_file}")

            print("Waiting 30 seconds before the next batch...")
            await asyncio.sleep(30)

            start_date = month_end + timedelta(days=1)
