"""Global configuration for Baseload Renewables Model."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SITES_DIR = DATA_DIR / "sites"
RESOURCE_DIR = DATA_DIR / "resource"
TIMESERIES_DIR = DATA_DIR / "timeseries"
SUMMARY_DIR = DATA_DIR / "summary"

# Create directories if they don't exist
for dir_path in [SITES_DIR, RESOURCE_DIR, TIMESERIES_DIR, SUMMARY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Site selection parameters
NUM_SITES = 10  # Testing with 10 sites
GRID_RESOLUTION = 0.5  # degrees for initial land grid

# API configuration (Renewables.ninja)
RENEWABLES_NINJA_TOKEN = os.environ.get("RENEWABLES_NINJA_TOKEN", "")
API_BASE_URL = "https://www.renewables.ninja/api"
SIMULATION_YEAR = 2022
API_RATE_LIMIT_DELAY = 1.0  # seconds between requests
API_MAX_RETRIES = 3
API_RETRY_DELAY = 5.0  # seconds

# System parameters
LOAD_GW = 1.0  # Constant baseload demand in GW
HOURS_PER_YEAR = 8760

# Capacity configuration ranges
SOLAR_CAPACITIES_GW = [1, 2, 3, 4, 5]
BATTERY_CAPACITIES_GWH = list(range(1, 16))  # 1 to 15 GWh

# Battery parameters
BATTERY_DURATION_HOURS = 4  # P_bat = E_bat / 4
BATTERY_MAX_POWER_GW = 5.0  # Optional cap on battery power
ROUND_TRIP_EFFICIENCY = 0.90
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
INITIAL_SOC_FRACTION = 0.5

# Parallelization settings
NUM_WORKERS = os.cpu_count() or 4
CHUNK_SIZE = 10  # Number of sites per chunk for parallel processing

# Output settings
SAVE_HOURLY_DATA = True  # Set to False to save disk space
COMPRESSION = "snappy"  # Parquet compression
