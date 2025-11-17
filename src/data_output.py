"""
Data output handlers for saving and analyzing simulation results.

Handles both hourly timeseries data and summary statistics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TIMESERIES_DIR,
    SUMMARY_DIR,
    SITES_DIR,
    COMPRESSION
)


def create_hourly_output(
    site_id: int,
    config_id: int,
    hours: np.ndarray,
    G_solar: np.ndarray,
    G_total: np.ndarray,
    P_ch: np.ndarray,
    P_dis: np.ndarray,
    SoC: np.ndarray,
    P_served: np.ndarray,
    P_unserved: np.ndarray,
    P_curtail: np.ndarray
) -> pd.DataFrame:
    """
    Create hourly output DataFrame for a single site-config pair.

    Args:
        site_id: Site identifier
        config_id: Configuration identifier
        hours: Hour indices (0-8759)
        G_solar: Solar generation (GW)
        G_total: Total generation (GW)
        P_ch: Charging power (GW)
        P_dis: Discharging power (GW)
        SoC: State of charge (GWh)
        P_served: Power served to load (GW)
        P_unserved: Unserved load (GW)
        P_curtail: Curtailed power (GW)

    Returns:
        DataFrame with hourly data
    """
    return pd.DataFrame({
        'site_id': site_id,
        'config_id': config_id,
        'hour': hours,
        'G_solar_GW': G_solar,
        'G_total_GW': G_total,
        'P_ch_GW': P_ch,
        'P_dis_GW': P_dis,
        'SoC_GWh': SoC,
        'P_served_GW': P_served,
        'P_unserved_GW': P_unserved,
        'P_curtail_GW': P_curtail
    })


def create_summary_row(
    site_id: int,
    lat_deg: float,
    lon_deg: float,
    config_id: int,
    C_solar_GW: float,
    E_bat_GWh: float,
    P_bat_GW: float,
    hours_fully_served_frac: float,
    energy_served_frac: float,
    avg_system_cf: float,
    total_curtailed_GWh: float,
    total_unserved_GWh: float
) -> Dict:
    """
    Create a summary row for a single site-config pair.

    Args:
        site_id: Site identifier
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        config_id: Configuration identifier
        C_solar_GW: Solar capacity (GW)
        E_bat_GWh: Battery energy capacity (GWh)
        P_bat_GW: Battery power capacity (GW)
        hours_fully_served_frac: Fraction of hours fully served
        energy_served_frac: Fraction of total energy served
        avg_system_cf: Average system capacity factor
        total_curtailed_GWh: Total curtailed energy (GWh)
        total_unserved_GWh: Total unserved energy (GWh)

    Returns:
        Dictionary for summary row
    """
    return {
        'site_id': site_id,
        'lat_deg': lat_deg,
        'lon_deg': lon_deg,
        'config_id': config_id,
        'C_solar_GW': C_solar_GW,
        'E_bat_GWh': E_bat_GWh,
        'P_bat_GW': P_bat_GW,
        'hours_fully_served_frac': hours_fully_served_frac,
        'energy_served_frac': energy_served_frac,
        'avg_system_cf': avg_system_cf,
        'total_curtailed_GWh': total_curtailed_GWh,
        'total_unserved_GWh': total_unserved_GWh
    }


def save_site_hourly_data(site_id: int, hourly_df: pd.DataFrame,
                          output_dir: Path = TIMESERIES_DIR) -> Path:
    """
    Save hourly data for a single site (all configs).

    Args:
        site_id: Site identifier
        hourly_df: DataFrame with hourly data for all configs
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = output_dir / f"site_{site_id:03d}.parquet"
    hourly_df.to_parquet(output_path, index=False, compression=COMPRESSION)
    return output_path


def load_site_hourly_data(site_id: int, config_id: Optional[int] = None,
                          input_dir: Path = TIMESERIES_DIR) -> pd.DataFrame:
    """
    Load hourly data for a site.

    Args:
        site_id: Site identifier
        config_id: Optional config ID to filter
        input_dir: Input directory

    Returns:
        DataFrame with hourly data
    """
    input_path = input_dir / f"site_{site_id:03d}.parquet"
    df = pd.read_parquet(input_path)

    if config_id is not None:
        df = df[df['config_id'] == config_id].copy()

    return df


def aggregate_summary_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate statistics across all sites.

    Args:
        summary_df: Full summary DataFrame

    Returns:
        DataFrame with aggregate statistics
    """
    # Group by configuration
    config_stats = summary_df.groupby('config_id').agg({
        'C_solar_GW': 'first',
        'E_bat_GWh': 'first',
        'P_bat_GW': 'first',
        'hours_fully_served_frac': ['mean', 'std', 'min', 'max'],
        'energy_served_frac': ['mean', 'std', 'min', 'max'],
        'avg_system_cf': ['mean', 'std'],
        'total_curtailed_GWh': ['mean', 'sum'],
        'total_unserved_GWh': ['mean', 'sum']
    }).reset_index()

    # Flatten column names
    config_stats.columns = ['_'.join(col).strip('_') for col in config_stats.columns]

    return config_stats


def find_best_configurations(
    summary_df: pd.DataFrame,
    min_energy_served: float = 0.99,
    sort_by: str = 'total_capacity'
) -> pd.DataFrame:
    """
    Find best configurations meeting reliability criteria.

    Args:
        summary_df: Full summary DataFrame
        min_energy_served: Minimum energy served fraction
        sort_by: Sort criterion ('total_capacity', 'battery_size', 'curtailment')

    Returns:
        DataFrame with best configurations per site
    """
    # Filter to configurations meeting criteria
    reliable = summary_df[summary_df['energy_served_frac'] >= min_energy_served].copy()

    if reliable.empty:
        print(f"Warning: No configurations meet {min_energy_served:.1%} reliability threshold")
        return pd.DataFrame()

    # Add total capacity column (solar only)
    reliable['total_capacity_GW'] = reliable['C_solar_GW']

    # Sort by criterion
    if sort_by == 'total_capacity':
        reliable = reliable.sort_values('total_capacity_GW')
    elif sort_by == 'battery_size':
        reliable = reliable.sort_values('E_bat_GWh')
    elif sort_by == 'curtailment':
        reliable = reliable.sort_values('total_curtailed_GWh')

    # Get best per site
    best_per_site = reliable.groupby('site_id').first().reset_index()

    return best_per_site


def export_summary_to_csv(summary_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Export summary to CSV format.

    Args:
        summary_df: Summary DataFrame
        output_path: Output path

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = SUMMARY_DIR / "summary.csv"

    summary_df.to_csv(output_path, index=False)
    print(f"Exported summary to {output_path}")

    return output_path


def calculate_site_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics per site across all configurations.

    Args:
        summary_df: Full summary DataFrame

    Returns:
        DataFrame with per-site statistics
    """
    site_stats = summary_df.groupby('site_id').agg({
        'lat_deg': 'first',
        'lon_deg': 'first',
        'hours_fully_served_frac': ['mean', 'max'],
        'energy_served_frac': ['mean', 'max'],
        'avg_system_cf': 'mean',
        'total_curtailed_GWh': 'mean',
        'total_unserved_GWh': 'mean'
    }).reset_index()

    # Flatten column names
    site_stats.columns = ['_'.join(col).strip('_') for col in site_stats.columns]

    return site_stats


def validate_outputs(summary_df: pd.DataFrame) -> Dict:
    """
    Validate output data for consistency.

    Args:
        summary_df: Summary DataFrame

    Returns:
        Dictionary with validation results
    """
    results = {
        'total_rows': len(summary_df),
        'unique_sites': summary_df['site_id'].nunique(),
        'unique_configs': summary_df['config_id'].nunique(),
        'expected_rows': summary_df['site_id'].nunique() * summary_df['config_id'].nunique(),
        'energy_served_range': (
            summary_df['energy_served_frac'].min(),
            summary_df['energy_served_frac'].max()
        ),
        'hours_served_range': (
            summary_df['hours_fully_served_frac'].min(),
            summary_df['hours_fully_served_frac'].max()
        ),
        'any_negative_values': (
            (summary_df['total_curtailed_GWh'] < 0).any() or
            (summary_df['total_unserved_GWh'] < 0).any()
        ),
        'any_impossible_values': (
            (summary_df['energy_served_frac'] > 1).any() or
            (summary_df['hours_fully_served_frac'] > 1).any()
        )
    }

    results['valid'] = (
        results['total_rows'] == results['expected_rows'] and
        not results['any_negative_values'] and
        not results['any_impossible_values']
    )

    return results


if __name__ == "__main__":
    # Test data output functions
    print("Testing data output handlers...")

    # Create sample data
    sample_summary = pd.DataFrame({
        'site_id': [0, 0, 0, 1, 1, 1],
        'lat_deg': [51.5, 51.5, 51.5, -22.3, -22.3, -22.3],
        'lon_deg': [-0.1, -0.1, -0.1, 130.2, 130.2, 130.2],
        'config_id': [0, 1, 2, 0, 1, 2],
        'C_solar_GW': [1, 2, 3, 1, 2, 3],
        'E_bat_GWh': [1, 1, 1, 1, 1, 1],
        'P_bat_GW': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        'hours_fully_served_frac': [0.85, 0.92, 0.98, 0.88, 0.95, 0.99],
        'energy_served_frac': [0.90, 0.95, 0.99, 0.92, 0.97, 0.995],
        'avg_system_cf': [0.25, 0.28, 0.27, 0.30, 0.32, 0.31],
        'total_curtailed_GWh': [100, 250, 500, 120, 280, 550],
        'total_unserved_GWh': [876, 438, 87.6, 700, 262, 43.8]
    })

    print("\n1. Sample summary data:")
    print(sample_summary)

    print("\n2. Aggregate statistics by config:")
    agg_stats = aggregate_summary_statistics(sample_summary)
    print(agg_stats)

    print("\n3. Best configurations (>95% energy served):")
    best = find_best_configurations(sample_summary, min_energy_served=0.95)
    print(best)

    print("\n4. Site statistics:")
    site_stats = calculate_site_statistics(sample_summary)
    print(site_stats)

    print("\n5. Validation results:")
    validation = validate_outputs(sample_summary)
    for key, value in validation.items():
        print(f"  {key}: {value}")

    print("\nData output handlers test complete!")
