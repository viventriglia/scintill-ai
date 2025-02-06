from pathlib import Path

from dotenv import dotenv_values

# GNSS (Presidente Prudente)
LATITUDE: float = -22.122112
LONGITUDE: float = -51.407095
ALTITUDE: float = 350_000
ELEVATION_THRESHOLD: float = 60
LW_S4_THRESHOLD: float = 0.4
UP_S4_THRESHOLD: float = 0.7

# ISMR
ISMR_KEY: str = dotenv_values("../.env.secret")["ISMR_KEY"]
MAX_CONCURRENT_REQUESTS: int = 5

# Data
DATA_IN: Path = Path("..", "data", "in")
