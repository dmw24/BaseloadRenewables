"""
Scalable simulation runner for processing all site-configuration pairs.

Uses parallel processing to handle 1000 sites × 375 configurations = 375,000 simulations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NUM_WORKERS,
    CHUNK_SIZE,
    TIMESERIES_DIR,
    SUMMARY_DIR,
    RESOURCE_DIR,
    SAVE_HOURLY_DATA,
    COMPRESSION
)
from src.dispatch_simulation import simulate_site_configs
from src.api_client import load_site_resource
from src.configurations import load_configurations
from src.site_selection import load_sites


def process_single_site(
    site_id: int,
    configs_df: pd.DataFrame,
    save_hourly: bool = SAVE_HOURLY_DATA
) -> pd.DataFrame:
    """
    Process all configurations for a single site.

    Args:
        site_id: Site identifier
        configs_df: DataFrame with all configurations
        save_hourly: Whether to save hourly data

    Returns:
        Summary DataFrame for this site
    """
    # Load resource data for site
    try:
        resource_df = load_site_resource(site_id)
    except FileNotFoundError:
        print(f"Warning: No resource data for site {site_id}, skipping")
        return pd.DataFrame()

    cf_solar = resource_df['cf_solar'].values

    # Run simulations for all configurations
    hourly_df, summary_df = simulate_site_configs(site_id, cf_solar, configs_df)

    # Save hourly data if requested
    if save_hourly:
        hourly_path = TIMESERIES_DIR / f"site_{site_id:03d}.parquet"
        hourly_df.to_parquet(hourly_path, index=False, compression=COMPRESSION)

    return summary_df


def process_site_batch(
    site_ids: List[int],
    configs_df: pd.DataFrame,
    save_hourly: bool = SAVE_HOURLY_DATA
) -> pd.DataFrame:
    """
    Process a batch of sites.

    Args:
        site_ids: List of site identifiers
        configs_df: DataFrame with all configurations
        save_hourly: Whether to save hourly data

    Returns:
        Combined summary DataFrame for all sites in batch
    """
    summaries = []

    for site_id in site_ids:
        summary = process_single_site(site_id, configs_df, save_hourly)
        if not summary.empty:
            summaries.append(summary)

    if summaries:
        return pd.concat(summaries, ignore_index=True)
    else:
        return pd.DataFrame()


def run_all_simulations(
    sites_df: pd.DataFrame = None,
    configs_df: pd.DataFrame = None,
    n_workers: int = NUM_WORKERS,
    chunk_size: int = CHUNK_SIZE,
    save_hourly: bool = SAVE_HOURLY_DATA,
    max_sites: int = None
) -> pd.DataFrame:
    """
    Run simulations for all sites and configurations.

    Args:
        sites_df: DataFrame with site information
        configs_df: DataFrame with configurations
        n_workers: Number of parallel workers
        chunk_size: Number of sites per chunk
        save_hourly: Whether to save hourly data
        max_sites: Maximum number of sites to process (for testing)

    Returns:
        Complete summary DataFrame
    """
    # Load data if not provided
    if sites_df is None:
        sites_df = load_sites()
    if configs_df is None:
        configs_df = load_configurations()

    # Limit sites for testing
    if max_sites:
        sites_df = sites_df.head(max_sites)

    site_ids = sites_df['site_id'].tolist()
    n_sites = len(site_ids)
    n_configs = len(configs_df)

    print(f"\nRunning simulations:")
    print(f"  Sites: {n_sites}")
    print(f"  Configurations: {n_configs}")
    print(f"  Total simulations: {n_sites * n_configs:,}")
    print(f"  Total timestep evaluations: {n_sites * n_configs * 8760:,}")
    print(f"  Workers: {n_workers}")
    print(f"  Chunk size: {chunk_size}")

    start_time = time.time()

    # Create chunks of sites
    chunks = [site_ids[i:i + chunk_size] for i in range(0, len(site_ids), chunk_size)]

    print(f"  Chunks: {len(chunks)}")

    # Process chunks in parallel
    if n_workers > 1:
        results = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(process_site_batch)(chunk, configs_df, save_hourly)
            for chunk in chunks
        )
    else:
        # Single-threaded for debugging
        results = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            result = process_site_batch(chunk, configs_df, save_hourly)
            results.append(result)

    # Combine all summaries
    all_summaries = [df for df in results if not df.empty]
    if all_summaries:
        final_summary = pd.concat(all_summaries, ignore_index=True)
    else:
        final_summary = pd.DataFrame()

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"  Total rows in summary: {len(final_summary):,}")

    if not final_summary.empty:
        print(f"  Simulations per second: {(n_sites * n_configs) / elapsed:.1f}")
        print(f"  Timesteps per second: {(n_sites * n_configs * 8760) / elapsed:,.0f}")

    return final_summary


def save_summary(summary_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Save final summary data.

    Args:
        summary_df: Summary DataFrame
        output_path: Path to save file

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = SUMMARY_DIR / "summary.parquet"

    summary_df.to_parquet(output_path, index=False, compression=COMPRESSION)
    print(f"Saved summary to {output_path}")

    # Also save as CSV for convenience
    csv_path = output_path.with_suffix('.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    return output_path


def load_summary(input_path: Path = None) -> pd.DataFrame:
    """
    Load summary data.

    Args:
        input_path: Path to summary file

    Returns:
        Summary DataFrame
    """
    if input_path is None:
        input_path = SUMMARY_DIR / "summary.parquet"

    return pd.read_parquet(input_path)


def get_site_timeseries(site_id: int, config_id: int = None) -> pd.DataFrame:
    """
    Load hourly timeseries data for a site.

    Args:
        site_id: Site identifier
        config_id: Optional configuration ID to filter

    Returns:
        Hourly DataFrame
    """
    path = TIMESERIES_DIR / f"site_{site_id:03d}.parquet"
    df = pd.read_parquet(path)

    if config_id is not None:
        df = df[df['config_id'] == config_id].copy()

    return df


def add_site_coordinates(summary_df: pd.DataFrame, sites_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lat/lon coordinates to summary DataFrame.

    Args:
        summary_df: Summary DataFrame
        sites_df: Sites DataFrame with coordinates

    Returns:
        Summary with coordinates added
    """
    return summary_df.merge(
        sites_df[['site_id', 'lat_deg', 'lon_deg']],
        on='site_id',
        how='left'
    )


if __name__ == "__main__":
    # Test runner with synthetic data
    print("Testing simulation runner...")

    # Generate test data if not exists
    from src.site_selection import select_global_sites, save_sites
    from src.configurations import generate_configurations, save_configurations
    from src.api_client import SyntheticDataGenerator

    # Create a small test set
    print("\n1. Creating test sites...")
    test_sites = pd.DataFrame({
        'site_id': range(5),
        'lat_deg': [51.5, -22.3, 38.6, 0.0, 70.0],
        'lon_deg': [-0.1, 130.2, 109.2, 0.0, 25.0]
    })
    save_sites(test_sites)

    print("\n2. Generating test resource data...")
    gen = SyntheticDataGenerator(seed=42)
    for _, row in test_sites.iterrows():
        site_id = int(row['site_id'])
        lat = row['lat_deg']
        lon = row['lon_deg']

        solar_cf = gen.generate_solar_profile(lat, lon)

        df = pd.DataFrame({
            'hour': range(8760),
            'cf_solar': solar_cf.values
        })

        cache_path = RESOURCE_DIR / f"site_{site_id:03d}.parquet"
        df.to_parquet(cache_path, index=False)
        print(f"  Created resource data for site {site_id}")

    print("\n3. Generating configurations...")
    configs_df = generate_configurations()
    save_configurations(configs_df)

    print("\n4. Running simulations (test mode)...")
    # Use only first 10 configs for speed
    test_configs = configs_df.head(10)

    summary_df = run_all_simulations(
        sites_df=test_sites,
        configs_df=test_configs,
        n_workers=1,  # Single-threaded for testing
        save_hourly=True
    )

    print("\n5. Summary results:")
    print(summary_df.head(10))

    print("\n6. Saving summary...")
    save_summary(summary_df)

    print("\nTest complete!")
