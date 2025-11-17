"""
Capacity configuration generator for solar, wind, and battery systems.

Generates all combinations of:
- Solar: 1-5 GW
- Wind: 1-5 GW
- Battery: 1-15 GWh
"""
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SOLAR_CAPACITIES_GW,
    WIND_CAPACITIES_GW,
    BATTERY_CAPACITIES_GWH,
    BATTERY_DURATION_HOURS,
    BATTERY_MAX_POWER_GW,
    LOAD_GW,
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    ROUND_TRIP_EFFICIENCY,
    INITIAL_SOC_FRACTION,
    DATA_DIR
)


def generate_configurations() -> pd.DataFrame:
    """
    Generate all capacity configurations.

    Returns:
        DataFrame with all configuration parameters
    """
    # Create all combinations
    configs = list(product(SOLAR_CAPACITIES_GW, WIND_CAPACITIES_GW, BATTERY_CAPACITIES_GWH))

    # Build DataFrame
    df = pd.DataFrame(configs, columns=['C_solar_GW', 'C_wind_GW', 'E_bat_GWh'])

    # Add config_id
    df.insert(0, 'config_id', range(len(df)))

    # Calculate battery power (4-hour duration)
    df['P_bat_GW'] = df['E_bat_GWh'] / BATTERY_DURATION_HOURS

    # Apply optional cap on battery power
    df['P_bat_GW'] = df['P_bat_GW'].clip(upper=BATTERY_MAX_POWER_GW)

    # Add fixed battery parameters
    df['eta_charge'] = CHARGE_EFFICIENCY
    df['eta_discharge'] = DISCHARGE_EFFICIENCY
    df['eta_roundtrip'] = ROUND_TRIP_EFFICIENCY
    df['initial_soc_frac'] = INITIAL_SOC_FRACTION

    # Calculate initial SoC in GWh
    df['SoC_0_GWh'] = df['E_bat_GWh'] * INITIAL_SOC_FRACTION

    # Add load
    df['L_GW'] = LOAD_GW

    # Add total renewable capacity
    df['C_total_GW'] = df['C_solar_GW'] + df['C_wind_GW']

    print(f"Generated {len(df)} configurations")
    print(f"Solar range: {df['C_solar_GW'].min()}-{df['C_solar_GW'].max()} GW")
    print(f"Wind range: {df['C_wind_GW'].min()}-{df['C_wind_GW'].max()} GW")
    print(f"Battery range: {df['E_bat_GWh'].min()}-{df['E_bat_GWh'].max()} GWh")
    print(f"Battery power range: {df['P_bat_GW'].min():.2f}-{df['P_bat_GW'].max():.2f} GW")

    return df


def save_configurations(configs_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Save configurations to file.

    Args:
        configs_df: DataFrame with configurations
        output_path: Path to save file

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = DATA_DIR / "configurations.parquet"

    configs_df.to_parquet(output_path, index=False)
    print(f"Saved {len(configs_df)} configurations to {output_path}")

    return output_path


def load_configurations(input_path: Path = None) -> pd.DataFrame:
    """
    Load configurations from file.

    Args:
        input_path: Path to configuration file

    Returns:
        DataFrame with configurations
    """
    if input_path is None:
        input_path = DATA_DIR / "configurations.parquet"

    configs_df = pd.read_parquet(input_path)
    print(f"Loaded {len(configs_df)} configurations from {input_path}")

    return configs_df


def get_config_by_id(configs_df: pd.DataFrame, config_id: int) -> dict:
    """
    Get configuration parameters by ID.

    Args:
        configs_df: DataFrame with all configurations
        config_id: Configuration identifier

    Returns:
        Dictionary of configuration parameters
    """
    row = configs_df[configs_df['config_id'] == config_id].iloc[0]
    return row.to_dict()


def configuration_summary(configs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics of configurations.

    Args:
        configs_df: DataFrame with configurations

    Returns:
        Summary DataFrame
    """
    summary = pd.DataFrame({
        'parameter': ['C_solar_GW', 'C_wind_GW', 'E_bat_GWh', 'P_bat_GW'],
        'min': [
            configs_df['C_solar_GW'].min(),
            configs_df['C_wind_GW'].min(),
            configs_df['E_bat_GWh'].min(),
            configs_df['P_bat_GW'].min()
        ],
        'max': [
            configs_df['C_solar_GW'].max(),
            configs_df['C_wind_GW'].max(),
            configs_df['E_bat_GWh'].max(),
            configs_df['P_bat_GW'].max()
        ],
        'unique_values': [
            configs_df['C_solar_GW'].nunique(),
            configs_df['C_wind_GW'].nunique(),
            configs_df['E_bat_GWh'].nunique(),
            configs_df['P_bat_GW'].nunique()
        ]
    })

    return summary


if __name__ == "__main__":
    # Generate and display configurations
    configs_df = generate_configurations()

    print("\nFirst 10 configurations:")
    print(configs_df.head(10))

    print("\nConfiguration summary:")
    print(configuration_summary(configs_df))

    print(f"\nTotal configurations: {len(configs_df)}")
    print(f"Expected: {len(SOLAR_CAPACITIES_GW)} × {len(WIND_CAPACITIES_GW)} × {len(BATTERY_CAPACITIES_GWH)} = "
          f"{len(SOLAR_CAPACITIES_GW) * len(WIND_CAPACITIES_GW) * len(BATTERY_CAPACITIES_GWH)}")

    # Save configurations
    save_configurations(configs_df)
