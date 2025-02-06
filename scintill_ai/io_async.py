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

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def get_gnss_data_async(session, start, end, station_name, fields):
    """
    Asynchronous function to fetch GNSS data.
    """
    fields_no_space = ",".join(s_.strip() for s_ in fields.split(","))
    url = f"http://is-cigala-calibra.fct.unesp.br/is/ismrtool/calc-var/service_loadISMR.php"
    url += f"?date_begin={start}&date_end={end}&stationName={station_name.strip()}&field_list={fields_no_space}&mode=csv&key={quote(ISMR_KEY.strip())}"
    async with semaphore:
        print(f"{start} -> {end}")
        async with session.get(url) as response:
            return await response.text()


async def process_day_data_async(session, dt_begin, dt_end, station_name, fields):
    """
    Process GNSS data for a single day asynchronously.
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
    Asynchronous function to get aggregated GNSS data over a specified range of dates.
    """
    date_range = pd.date_range(start, end)
    dfs = []

    async with aiohttp.ClientSession() as session:
        # Crea una lista di coroutines per ciascun giorno
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

        # Esegui tutte le coroutines in parallelo
        results = await asyncio.gather(*tasks)

        # Raccogli i risultati
        for result in results:
            dfs.append(result)

    return pd.concat(dfs, ignore_index=True)


async def get_aggregated_gnss_data_by_month_async(
    start: str, end: str, station_name: str, fields: str, output_dir: str
):
    """
    Scarica i dati GNSS aggregati mese per mese in modalità asincrona.
    Salva ogni mese in un file Pickle con nome 'YYYY_MM.pickle'.
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

            print(
                f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}..."
            )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filtra gli errori e unisci i dati
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
