"""
Battery dispatch simulation engine.

Simulates hourly dispatch of solar and battery storage
to meet a constant baseload demand.
"""
import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LOAD_GW,
    HOURS_PER_YEAR,
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    INITIAL_SOC_FRACTION
)


@dataclass
class DispatchResults:
    """Container for dispatch simulation results."""
    # Hourly arrays
    G_solar: np.ndarray       # Solar generation (GW)
    G_total: np.ndarray       # Total generation (GW)
    P_charge: np.ndarray      # Battery charging power (GW)
    P_discharge: np.ndarray   # Battery discharging power (GW)
    SoC: np.ndarray           # State of charge (GWh)
    P_served: np.ndarray      # Power served to load (GW)
    P_unserved: np.ndarray    # Unserved load (GW)
    P_curtailed: np.ndarray   # Curtailed power (GW)

    # Summary metrics
    hours_fully_served: int
    hours_fully_served_frac: float
    energy_served_frac: float
    avg_system_cf: float
    total_curtailed_GWh: float
    total_unserved_GWh: float


@jit(nopython=True)
def simulate_dispatch_numba(
    cf_solar: np.ndarray,
    C_solar: float,
    E_bat: float,
    P_bat: float,
    L: float,
    eta_charge: float,
    eta_discharge: float,
    SoC_0: float,
    n_hours: int
) -> Tuple[np.ndarray, ...]:
    """
    Numba-optimized dispatch simulation.

    Args:
        cf_solar: Hourly solar capacity factors
        C_solar: Solar nameplate capacity (GW)
        E_bat: Battery energy capacity (GWh)
        P_bat: Battery power limit (GW)
        L: Load (GW)
        eta_charge: Charging efficiency
        eta_discharge: Discharging efficiency
        SoC_0: Initial state of charge (GWh)
        n_hours: Number of hours to simulate

    Returns:
        Tuple of arrays: (G_solar, G_total, P_charge, P_discharge,
                         SoC, P_served, P_unserved, P_curtailed)
    """
    # Initialize arrays
    G_solar = np.zeros(n_hours)
    G_total = np.zeros(n_hours)
    P_charge = np.zeros(n_hours)
    P_discharge = np.zeros(n_hours)
    SoC = np.zeros(n_hours + 1)  # One extra for final state
    P_served = np.zeros(n_hours)
    P_unserved = np.zeros(n_hours)
    P_curtailed = np.zeros(n_hours)

    # Initial state
    SoC[0] = SoC_0

    # Simulate each hour
    for t in range(n_hours):
        # Compute solar generation
        G_solar[t] = cf_solar[t] * C_solar
        G_total[t] = G_solar[t]

        # Serve load from generation first
        served_from_gen = min(G_total[t], L)

        # Compute surplus or deficit
        surplus = max(0.0, G_total[t] - L)
        deficit = max(0.0, L - G_total[t])

        # Dispatch battery
        if surplus > 0:
            # Charge battery
            # Max charge possible considering capacity and power limits
            charge_headroom = (E_bat - SoC[t]) / eta_charge
            charge_possible = min(P_bat, charge_headroom)
            P_charge[t] = min(surplus, charge_possible)
            P_discharge[t] = 0.0

        elif deficit > 0:
            # Discharge battery
            # Max discharge possible considering SoC and power limits
            discharge_possible = min(P_bat, eta_discharge * SoC[t])
            P_discharge[t] = min(deficit, discharge_possible)
            P_charge[t] = 0.0

        else:
            P_charge[t] = 0.0
            P_discharge[t] = 0.0

        # Update state of charge
        SoC[t + 1] = SoC[t] + eta_charge * P_charge[t] - (P_discharge[t] / eta_discharge)

        # Ensure SoC stays within bounds (numerical safety)
        SoC[t + 1] = max(0.0, min(E_bat, SoC[t + 1]))

        # Compute delivered power and metrics
        P_served[t] = served_from_gen + P_discharge[t]
        P_unserved[t] = max(0.0, L - P_served[t])
        P_curtailed[t] = max(0.0, G_total[t] - L - P_charge[t])

    return (G_solar, G_total, P_charge, P_discharge,
            SoC[:-1], P_served, P_unserved, P_curtailed)


def simulate_dispatch(
    cf_solar: np.ndarray,
    C_solar: float,
    E_bat: float,
    P_bat: float,
    L: float = LOAD_GW,
    eta_charge: float = CHARGE_EFFICIENCY,
    eta_discharge: float = DISCHARGE_EFFICIENCY,
    initial_soc_frac: float = INITIAL_SOC_FRACTION
) -> DispatchResults:
    """
    Run dispatch simulation for a single configuration.

    Args:
        cf_solar: Hourly solar capacity factors (8760 values)
        C_solar: Solar nameplate capacity (GW)
        E_bat: Battery energy capacity (GWh)
        P_bat: Battery power limit (GW)
        L: Load (GW)
        eta_charge: Charging efficiency
        eta_discharge: Discharging efficiency
        initial_soc_frac: Initial SoC as fraction of E_bat

    Returns:
        DispatchResults object with hourly data and summary metrics
    """
    # Ensure numpy arrays
    cf_solar = np.asarray(cf_solar, dtype=np.float64)

    n_hours = len(cf_solar)

    # Initial state of charge
    SoC_0 = initial_soc_frac * E_bat

    # Run optimized simulation
    (G_solar, G_total, P_charge, P_discharge,
     SoC, P_served, P_unserved, P_curtailed) = simulate_dispatch_numba(
        cf_solar, C_solar, E_bat, P_bat,
        L, eta_charge, eta_discharge, SoC_0, n_hours
    )

    # Calculate summary metrics
    hours_fully_served = int(np.sum(P_served >= L - 1e-9))  # Allow small tolerance
    hours_fully_served_frac = hours_fully_served / n_hours

    total_load = L * n_hours
    total_served = np.sum(P_served)
    energy_served_frac = total_served / total_load

    total_generation = np.sum(G_total)
    total_capacity = C_solar * n_hours
    avg_system_cf = total_generation / total_capacity if total_capacity > 0 else 0

    total_curtailed_GWh = np.sum(P_curtailed)
    total_unserved_GWh = np.sum(P_unserved)

    return DispatchResults(
        G_solar=G_solar,
        G_total=G_total,
        P_charge=P_charge,
        P_discharge=P_discharge,
        SoC=SoC,
        P_served=P_served,
        P_unserved=P_unserved,
        P_curtailed=P_curtailed,
        hours_fully_served=hours_fully_served,
        hours_fully_served_frac=hours_fully_served_frac,
        energy_served_frac=energy_served_frac,
        avg_system_cf=avg_system_cf,
        total_curtailed_GWh=total_curtailed_GWh,
        total_unserved_GWh=total_unserved_GWh
    )


def simulate_site_configs(
    site_id: int,
    cf_solar: np.ndarray,
    configs_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate all configurations for a single site.

    Args:
        site_id: Site identifier
        cf_solar: Hourly solar capacity factors
        configs_df: DataFrame with all configurations

    Returns:
        Tuple of (hourly_df, summary_df)
    """
    hourly_data = []
    summary_data = []

    for _, config in configs_df.iterrows():
        config_id = int(config['config_id'])

        # Run simulation
        results = simulate_dispatch(
            cf_solar=cf_solar,
            C_solar=config['C_solar_GW'],
            E_bat=config['E_bat_GWh'],
            P_bat=config['P_bat_GW'],
            L=config.get('L_GW', LOAD_GW),
            eta_charge=config.get('eta_charge', CHARGE_EFFICIENCY),
            eta_discharge=config.get('eta_discharge', DISCHARGE_EFFICIENCY),
            initial_soc_frac=config.get('initial_soc_frac', INITIAL_SOC_FRACTION)
        )

        # Store hourly data (append config_id column)
        hourly_df = pd.DataFrame({
            'site_id': site_id,
            'config_id': config_id,
            'hour': np.arange(len(results.G_solar)),
            'G_solar_GW': results.G_solar,
            'G_total_GW': results.G_total,
            'P_ch_GW': results.P_charge,
            'P_dis_GW': results.P_discharge,
            'SoC_GWh': results.SoC,
            'P_served_GW': results.P_served,
            'P_unserved_GW': results.P_unserved,
            'P_curtail_GW': results.P_curtailed
        })
        hourly_data.append(hourly_df)

        # Store summary
        summary_data.append({
            'site_id': site_id,
            'config_id': config_id,
            'C_solar_GW': config['C_solar_GW'],
            'E_bat_GWh': config['E_bat_GWh'],
            'P_bat_GW': config['P_bat_GW'],
            'hours_fully_served_frac': results.hours_fully_served_frac,
            'energy_served_frac': results.energy_served_frac,
            'avg_system_cf': results.avg_system_cf,
            'total_curtailed_GWh': results.total_curtailed_GWh,
            'total_unserved_GWh': results.total_unserved_GWh
        })

    # Combine all hourly data for site
    combined_hourly = pd.concat(hourly_data, ignore_index=True)
    summary_df = pd.DataFrame(summary_data)

    return combined_hourly, summary_df


def results_to_dataframe(results: DispatchResults) -> pd.DataFrame:
    """
    Convert DispatchResults to a DataFrame.

    Args:
        results: DispatchResults object

    Returns:
        DataFrame with hourly data
    """
    return pd.DataFrame({
        'hour': np.arange(len(results.G_solar)),
        'G_solar_GW': results.G_solar,
        'G_total_GW': results.G_total,
        'P_ch_GW': results.P_charge,
        'P_dis_GW': results.P_discharge,
        'SoC_GWh': results.SoC,
        'P_served_GW': results.P_served,
        'P_unserved_GW': results.P_unserved,
        'P_curtail_GW': results.P_curtailed
    })


if __name__ == "__main__":
    # Test dispatch simulation
    print("Testing dispatch simulation...")

    # Create synthetic test data
    np.random.seed(42)
    n_hours = 8760

    # Simulate solar pattern (peaks during day)
    hour_of_day = np.arange(n_hours) % 24
    day_of_year = np.arange(n_hours) // 24

    # Simple solar profile
    solar_base = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    solar_noise = 0.7 + 0.3 * np.random.random(n_hours)
    cf_solar = solar_base * solar_noise * 0.25

    # Test configuration
    C_solar = 3.0  # GW
    E_bat = 10.0   # GWh
    P_bat = 2.5    # GW

    print(f"\nConfiguration:")
    print(f"  Solar: {C_solar} GW")
    print(f"  Battery: {E_bat} GWh / {P_bat} GW")
    print(f"  Load: {LOAD_GW} GW")

    # Run simulation
    results = simulate_dispatch(cf_solar, C_solar, E_bat, P_bat)

    print(f"\nResults:")
    print(f"  Hours fully served: {results.hours_fully_served} ({results.hours_fully_served_frac:.2%})")
    print(f"  Energy served fraction: {results.energy_served_frac:.2%}")
    print(f"  Average system CF: {results.avg_system_cf:.2%}")
    print(f"  Total curtailed: {results.total_curtailed_GWh:.2f} GWh")
    print(f"  Total unserved: {results.total_unserved_GWh:.2f} GWh")

    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"\nHourly data shape: {df.shape}")
    print(df.head())

    # Test multiple configs
    print("\n\nTesting multiple configurations...")
    from configurations import generate_configurations

    configs_df = generate_configurations()
    sample_configs = configs_df.head(5)  # Just test first 5

    hourly, summary = simulate_site_configs(0, cf_solar, sample_configs)

    print(f"Hourly data shape: {hourly.shape}")
    print(f"Summary shape: {summary.shape}")
    print("\nSummary:")
    print(summary)
