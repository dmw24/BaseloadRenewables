"""
API client for fetching renewable energy data from Renewables.ninja.

Handles rate limiting, retries, and local caching of results.
"""
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RENEWABLES_NINJA_TOKEN,
    API_BASE_URL,
    SIMULATION_YEAR,
    API_RATE_LIMIT_DELAY,
    API_MAX_RETRIES,
    API_RETRY_DELAY,
    RESOURCE_DIR,
    HOURS_PER_YEAR
)


class RenewablesNinjaClient:
    """Client for Renewables.ninja API."""

    def __init__(self, token: str = None):
        """
        Initialize the API client.

        Args:
            token: API token (get from renewables.ninja)
        """
        self.token = token or RENEWABLES_NINJA_TOKEN
        self.session = requests.Session()

        if self.token:
            self.session.headers.update({
                'Authorization': f'Token {self.token}'
            })

        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < API_RATE_LIMIT_DELAY:
            time.sleep(API_RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: dict) -> dict:
        """
        Make API request with retry logic.

        Args:
            url: API endpoint URL
            params: Request parameters

        Returns:
            JSON response data
        """
        self._rate_limit()

        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=60)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    raise PermissionError("Invalid API token or access denied")
                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    print(f"Request failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed after {API_MAX_RETRIES} attempts")

    def fetch_solar_profile(self, lat: float, lon: float, year: int = SIMULATION_YEAR) -> pd.Series:
        """
        Fetch hourly solar capacity factors for a location.

        Args:
            lat: Latitude
            lon: Longitude
            year: Simulation year

        Returns:
            Series with 8760 hourly capacity factors
        """
        url = f"{API_BASE_URL}/data/pv"

        params = {
            'lat': lat,
            'lon': lon,
            'date_from': f'{year}-01-01',
            'date_to': f'{year}-12-31',
            'dataset': 'merra2',
            'capacity': 1.0,  # 1 kW system -> returns capacity factor
            'system_loss': 0.1,
            'tracking': 0,  # Fixed tilt
            'tilt': abs(lat),  # Optimal tilt ≈ latitude
            'azim': 180 if lat >= 0 else 0,  # South-facing in N hemisphere
            'format': 'json'
        }

        data = self._make_request(url, params)

        # Extract hourly values
        cf_series = pd.Series(data['data']['electricity'])
        cf_series.index = pd.to_datetime(cf_series.index)

        # Ensure 8760 hours
        if len(cf_series) != HOURS_PER_YEAR:
            warnings.warn(f"Got {len(cf_series)} hours instead of {HOURS_PER_YEAR}")

        return cf_series


class NASAPowerClient:
    """Client for NASA POWER API (no authentication required)."""

    def __init__(self):
        """Initialize the NASA POWER API client."""
        self.base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        self.session = requests.Session()
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < API_RATE_LIMIT_DELAY:
            time.sleep(API_RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, params: dict) -> dict:
        """Make API request with retry logic."""
        self._rate_limit()

        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.session.get(self.base_url, params=params, timeout=120)

                if response.status_code == 200:
                    return response.json()
                else:
                    if attempt < API_MAX_RETRIES - 1:
                        wait_time = API_RETRY_DELAY * (2 ** attempt)
                        print(f"Request failed (status {response.status_code}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        response.raise_for_status()

            except requests.exceptions.RequestException as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    print(f"Request failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed after {API_MAX_RETRIES} attempts")

    def fetch_solar_profile(self, lat: float, lon: float, year: int = SIMULATION_YEAR) -> pd.Series:
        """
        Fetch hourly solar irradiance and convert to capacity factor.

        Uses Global Horizontal Irradiance (GHI) from NASA POWER.
        """
        params = {
            'start': f'{year}0101',
            'end': f'{year}1231',
            'latitude': lat,
            'longitude': lon,
            'community': 'RE',  # Renewable Energy
            'parameters': 'ALLSKY_SFC_SW_DWN',  # GHI in kW/m²
            'format': 'JSON',
            'time-standard': 'UTC'
        }

        data = self._make_request(params)

        # Extract hourly GHI values
        ghi_data = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']

        # Convert to hourly series
        hours = []
        ghi_values = []

        for date_hour, value in ghi_data.items():
            hours.append(date_hour)
            ghi_values.append(value if value != -999 else 0)  # -999 is missing data

        # Convert GHI to capacity factor
        # NASA POWER returns GHI in W/m² (NOT kW/m²)
        # Standard Test Conditions (STC) for PV: 1000 W/m² = 1 kW/m²
        # Capacity factor = (GHI / STC) * Performance_Ratio
        # Performance ratio accounts for temperature, inverter, wiring, dust losses (~80%)
        ghi_array = np.array(ghi_values)

        # Convert W/m² to capacity factor
        # STC = 1000 W/m²
        # Performance ratio = 0.80 (accounting for system losses)
        stc_irradiance = 1000.0  # W/m²
        performance_ratio = 0.80

        # CF = (GHI / STC) * PR
        # Peak GHI of 1000 W/m² gives CF of 0.80
        cf_values = (ghi_array / stc_irradiance) * performance_ratio

        # Clip to [0, 1]
        cf_values = np.clip(cf_values, 0, 1)

        # Ensure 8760 hours (may need padding for leap years)
        if len(cf_values) > HOURS_PER_YEAR:
            cf_values = cf_values[:HOURS_PER_YEAR]
        elif len(cf_values) < HOURS_PER_YEAR:
            # Pad with zeros if needed
            cf_values = np.pad(cf_values, (0, HOURS_PER_YEAR - len(cf_values)))

        time_index = pd.date_range(start=f'{year}-01-01', periods=HOURS_PER_YEAR, freq='h')
        return pd.Series(cf_values, index=time_index, name='electricity')


class SyntheticDataGenerator:
    """
    Generate synthetic renewable profiles when API is unavailable.

    Uses realistic patterns based on latitude and typical capacity factors.
    """

    def __init__(self, seed: int = None):
        """
        Initialize synthetic data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def generate_solar_profile(self, lat: float, lon: float, year: int = SIMULATION_YEAR) -> pd.Series:
        """
        Generate synthetic solar capacity factors.

        Models daily and seasonal patterns based on latitude.
        """
        hours = np.arange(HOURS_PER_YEAR)

        # Hour of day (0-23)
        hour_of_day = hours % 24

        # Day of year (0-364)
        day_of_year = hours // 24

        # Solar declination (seasonal effect)
        declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)

        # Approximate solar altitude at each hour
        lat_rad = np.radians(lat)
        decl_rad = np.radians(declination)

        # Hour angle (peak at noon)
        hour_angle = (hour_of_day - 12) * 15  # degrees
        hour_angle_rad = np.radians(hour_angle)

        # Solar elevation angle
        sin_elevation = (np.sin(lat_rad) * np.sin(decl_rad) +
                        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad))

        # Capacity factor based on elevation (only positive when sun is up)
        base_cf = np.maximum(0, sin_elevation) ** 1.3

        # Add cloud variability
        cloud_factor = 0.7 + 0.3 * self.rng.random(HOURS_PER_YEAR)

        # Combine
        cf_values = base_cf * cloud_factor

        # Scale to realistic annual average (latitude dependent)
        # Higher latitudes have lower average CF
        target_avg_cf = 0.25 - 0.15 * (abs(lat) / 90)
        target_avg_cf = max(0.1, target_avg_cf)

        current_avg = np.mean(cf_values)
        if current_avg > 0:
            cf_values = cf_values * (target_avg_cf / current_avg)

        # Clip to valid range
        cf_values = np.clip(cf_values, 0, 1)

        # Create time index
        time_index = pd.date_range(start=f'{year}-01-01', periods=HOURS_PER_YEAR, freq='h')

        return pd.Series(cf_values, index=time_index, name='electricity')


def fetch_site_data(site_id: int, lat: float, lon: float,
                   api_source: str = 'nasa_power', cache: bool = True) -> pd.DataFrame:
    """
    Fetch renewable data for a single site from external API.

    Args:
        site_id: Site identifier
        lat: Latitude
        lon: Longitude
        api_source: API to use - 'nasa_power', 'renewables_ninja', or 'synthetic'
        cache: If True, cache results locally

    Returns:
        DataFrame with hour, cf_solar columns
    """
    cache_path = RESOURCE_DIR / f"site_{site_id:03d}.parquet"

    # Check cache first
    if cache and cache_path.exists():
        print(f"Loading cached data for site {site_id}")
        return pd.read_parquet(cache_path)

    # Fetch data from API
    print(f"Fetching data for site {site_id} (lat={lat:.2f}, lon={lon:.2f}) from {api_source}...")

    if api_source == 'nasa_power':
        try:
            client = NASAPowerClient()
            solar_cf = client.fetch_solar_profile(lat, lon)
        except Exception as e:
            print(f"NASA POWER API failed: {e}. Falling back to synthetic data.")
            generator = SyntheticDataGenerator(seed=site_id)
            solar_cf = generator.generate_solar_profile(lat, lon)

    elif api_source == 'renewables_ninja':
        if not RENEWABLES_NINJA_TOKEN:
            raise ValueError("RENEWABLES_NINJA_TOKEN environment variable not set")
        try:
            client = RenewablesNinjaClient()
            solar_cf = client.fetch_solar_profile(lat, lon)
        except Exception as e:
            print(f"Renewables.ninja API failed: {e}. Falling back to synthetic data.")
            generator = SyntheticDataGenerator(seed=site_id)
            solar_cf = generator.generate_solar_profile(lat, lon)

    elif api_source == 'synthetic':
        generator = SyntheticDataGenerator(seed=site_id)
        solar_cf = generator.generate_solar_profile(lat, lon)

    else:
        raise ValueError(f"Unknown API source: {api_source}")

    # Create DataFrame
    df = pd.DataFrame({
        'hour': range(HOURS_PER_YEAR),
        'cf_solar': solar_cf.values
    })

    # Cache result
    if cache:
        df.to_parquet(cache_path, index=False)
        print(f"Cached data for site {site_id} at {cache_path}")

    return df


def fetch_all_sites(sites_df: pd.DataFrame, api_source: str = 'nasa_power',
                   max_sites: int = None) -> dict:
    """
    Fetch renewable data for all sites from external API.

    Args:
        sites_df: DataFrame with site_id, lat_deg, lon_deg
        api_source: API to use - 'nasa_power', 'renewables_ninja', or 'synthetic'
        max_sites: Maximum number of sites to fetch (for testing)

    Returns:
        Dictionary mapping site_id to resource DataFrame
    """
    from tqdm import tqdm

    if max_sites:
        sites_df = sites_df.head(max_sites)

    results = {}

    for _, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc=f"Fetching from {api_source}"):
        site_id = int(row['site_id'])
        lat = row['lat_deg']
        lon = row['lon_deg']

        results[site_id] = fetch_site_data(site_id, lat, lon, api_source=api_source)

    print(f"Fetched data for {len(results)} sites from {api_source}")

    return results


def load_site_resource(site_id: int) -> pd.DataFrame:
    """
    Load cached resource data for a site.

    Args:
        site_id: Site identifier

    Returns:
        DataFrame with hour, cf_solar columns
    """
    cache_path = RESOURCE_DIR / f"site_{site_id:03d}.parquet"

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached data for site {site_id}")

    return pd.read_parquet(cache_path)


if __name__ == "__main__":
    # Test synthetic data generation
    print("Testing synthetic data generator...")

    gen = SyntheticDataGenerator(seed=42)

    # Test at different latitudes
    test_locs = [
        (51.5, -0.1, "London"),
        (-22.3, 130.2, "Australia"),
        (38.6, 109.2, "China"),
        (0.0, 0.0, "Equator"),
        (70.0, 25.0, "Arctic")
    ]

    for lat, lon, name in test_locs:
        solar_cf = gen.generate_solar_profile(lat, lon)
        print(f"\n{name} (lat={lat:.1f}):")
        print(f"  Solar CF: avg={solar_cf.mean():.3f}, max={solar_cf.max():.3f}")

        # Save sample
        df = pd.DataFrame({
            'hour': range(HOURS_PER_YEAR),
            'cf_solar': solar_cf.values
        })

        sample_path = RESOURCE_DIR / f"sample_{name.lower().replace(' ', '_')}.parquet"
        df.to_parquet(sample_path, index=False)
        print(f"  Saved to {sample_path}")
