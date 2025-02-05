from pathlib import Path

from dotenv import dotenv_values

# GNSS (Presidente Prudente)
ISMR_KEY = dotenv_values("../.env.secret")["ISMR_KEY"]
LATITUDE = -22.122112
LONGITUDE = -51.407095
ALTITUDE = 350_000
ELEVATION_THRESHOLD = 60
LW_S4_THRESHOLD = 0.4
UP_S4_THRESHOLD = 0.7

# Data
DATA_IN = Path("..", "data", "in")
