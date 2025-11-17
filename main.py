#!/usr/bin/env python3
"""
Main orchestration script for Baseload Renewables Model.

This script runs the complete pipeline:
1. Select ~1000 global land sites
2. Fetch renewable resource data (solar capacity factors)
3. Generate capacity configurations
4. Run dispatch simulations for all site-config pairs
5. Save results (hourly and summary data)
"""
import argparse
import time
from pathlib import Path

from config import (
    NUM_SITES,
    NUM_WORKERS,
    SAVE_HOURLY_DATA,
    SITES_DIR,
    RESOURCE_DIR,
    SUMMARY_DIR,
    TIMESERIES_DIR
)
from src.site_selection import select_global_sites, save_sites, load_sites
from src.api_client import fetch_all_sites, SyntheticDataGenerator
from src.configurations import generate_configurations, save_configurations, load_configurations
from src.runner import run_all_simulations, save_summary, add_site_coordinates
from src.data_output import validate_outputs, find_best_configurations, calculate_site_statistics


def step_1_select_sites(n_sites: int = NUM_SITES, force: bool = False) -> None:
    """Step 1: Select ~1000 global land sites."""
    print("\n" + "="*60)
    print("STEP 1: Site Selection")
    print("="*60)

    sites_file = SITES_DIR / "sites.csv"

    if sites_file.exists() and not force:
        print(f"Sites file already exists: {sites_file}")
        sites_df = load_sites(sites_file)
        print(f"Loaded {len(sites_df)} sites")
    else:
        print(f"Selecting {n_sites} globally distributed land sites...")
        sites_df = select_global_sites(n_sites=n_sites)
        save_sites(sites_df)
        print(f"Saved {len(sites_df)} sites")

    return sites_df


def step_2_fetch_resource_data(sites_df, api_source: str = 'nasa_power',
                                max_sites: int = None, force: bool = False) -> None:
    """Step 2: Fetch renewable resource data from external API for each site."""
    print("\n" + "="*60)
    print("STEP 2: Fetch Renewable Resource Data from API")
    print("="*60)

    # Check how many sites already have data
    existing_count = sum(1 for f in RESOURCE_DIR.glob("site_*.parquet"))

    if max_sites:
        target_sites = min(max_sites, len(sites_df))
    else:
        target_sites = len(sites_df)

    if existing_count >= target_sites and not force:
        print(f"Resource data already exists for {existing_count} sites (target: {target_sites})")
        return

    print(f"Using {api_source} API to fetch renewable resource data...")
    print(f"  API: {api_source}")
    print(f"  Sites to fetch: {target_sites}")
    print(f"  This will make {target_sites} API calls (solar per site)")

    fetch_all_sites(sites_df.head(target_sites) if max_sites else sites_df,
                   api_source=api_source)


def step_3_generate_configurations(force: bool = False):
    """Step 3: Generate all capacity configurations."""
    print("\n" + "="*60)
    print("STEP 3: Generate Configurations")
    print("="*60)

    from config import DATA_DIR
    config_file = DATA_DIR / "configurations.parquet"

    if config_file.exists() and not force:
        print(f"Configurations file already exists: {config_file}")
        configs_df = load_configurations(config_file)
    else:
        configs_df = generate_configurations()
        save_configurations(configs_df)

    return configs_df


def step_4_run_simulations(sites_df, configs_df, n_workers: int = NUM_WORKERS,
                           save_hourly: bool = SAVE_HOURLY_DATA,
                           max_sites: int = None, max_configs: int = None):
    """Step 4: Run all dispatch simulations."""
    print("\n" + "="*60)
    print("STEP 4: Run Dispatch Simulations")
    print("="*60)

    if max_sites:
        sites_df = sites_df.head(max_sites)
    if max_configs:
        configs_df = configs_df.head(max_configs)

    summary_df = run_all_simulations(
        sites_df=sites_df,
        configs_df=configs_df,
        n_workers=n_workers,
        save_hourly=save_hourly,
        max_sites=None  # Already limited above
    )

    # Add site coordinates to summary
    summary_df = add_site_coordinates(summary_df, sites_df)

    return summary_df


def step_5_save_and_analyze(summary_df, sites_df):
    """Step 5: Save results and perform basic analysis."""
    print("\n" + "="*60)
    print("STEP 5: Save Results and Analysis")
    print("="*60)

    from src.visualization import generate_all_plots

    # Save summary (Parquet and CSV)
    save_summary(summary_df)

    # Export additional CSV files
    print("\nExporting CSV files...")

    # Export best configurations per site
    best_per_site = summary_df.groupby('site_id').apply(
        lambda x: x.loc[x['energy_served_frac'].idxmax()]
    ).reset_index(drop=True)
    best_csv_path = SUMMARY_DIR / "best_configurations_per_site.csv"
    best_per_site.to_csv(best_csv_path, index=False)
    print(f"  Saved best configurations: {best_csv_path}")

    # Export site statistics
    site_stats = calculate_site_statistics(summary_df)
    site_stats_path = SUMMARY_DIR / "site_statistics.csv"
    site_stats.to_csv(site_stats_path, index=False)
    print(f"  Saved site statistics: {site_stats_path}")

    # Export aggregate configuration statistics
    config_stats = summary_df.groupby(['C_solar_GW', 'E_bat_GWh']).agg({
        'energy_served_frac': ['mean', 'std', 'min', 'max'],
        'hours_fully_served_frac': ['mean', 'std'],
        'avg_system_cf': 'mean',
        'total_curtailed_GWh': 'mean',
        'total_unserved_GWh': 'mean'
    }).reset_index()
    config_stats.columns = ['_'.join(col).strip('_') for col in config_stats.columns]
    config_stats_path = SUMMARY_DIR / "configuration_statistics.csv"
    config_stats.to_csv(config_stats_path, index=False)
    print(f"  Saved configuration statistics: {config_stats_path}")

    # Validate outputs
    print("\nValidating outputs...")
    validation = validate_outputs(summary_df)
    for key, value in validation.items():
        print(f"  {key}: {value}")

    if not validation['valid']:
        print("WARNING: Output validation failed!")

    # Basic analysis
    print("\n--- Basic Analysis ---")

    # Best configurations for 99% reliability
    print("\nBest configurations (≥99% energy served):")
    best_99 = find_best_configurations(summary_df, min_energy_served=0.99)
    if not best_99.empty:
        print(f"  Sites meeting criteria: {len(best_99)}/{len(sites_df)}")
        print(f"  Mean solar capacity: {best_99['C_solar_GW'].mean():.2f} GW")
        print(f"  Mean battery capacity: {best_99['E_bat_GWh'].mean():.2f} GWh")
    else:
        print("  No configurations meet 99% reliability")

    # Best for 95% reliability
    print("\nBest configurations (≥95% energy served):")
    best_95 = find_best_configurations(summary_df, min_energy_served=0.95)
    if not best_95.empty:
        print(f"  Sites meeting criteria: {len(best_95)}/{len(sites_df)}")
        print(f"  Mean solar capacity: {best_95['C_solar_GW'].mean():.2f} GW")
        print(f"  Mean battery capacity: {best_95['E_bat_GWh'].mean():.2f} GWh")

    # Site statistics
    print("\nSite-level statistics:")
    print(f"  Mean max energy served: {site_stats['energy_served_frac_max'].mean():.2%}")
    print(f"  Mean avg system CF: {site_stats['avg_system_cf_mean'].mean():.2%}")

    # Generate visualization plots
    print("\n--- Generating Visualization Plots ---")
    try:
        plots = generate_all_plots(sites_df, summary_df)
        print(f"Generated {len(plots)} visualization plots")
    except Exception as e:
        print(f"Warning: Could not generate all plots: {e}")

    return summary_df


def run_full_pipeline(
    n_sites: int = NUM_SITES,
    n_workers: int = NUM_WORKERS,
    api_source: str = 'nasa_power',
    save_hourly: bool = SAVE_HOURLY_DATA,
    max_sites: int = None,
    max_configs: int = None,
    force: bool = False
):
    """
    Run the complete pipeline.

    Args:
        n_sites: Number of sites to select
        n_workers: Number of parallel workers
        api_source: API source - 'nasa_power', 'renewables_ninja', or 'synthetic'
        save_hourly: Save hourly timeseries data
        max_sites: Limit number of sites (for testing)
        max_configs: Limit number of configs (for testing)
        force: Force regeneration of all data
    """
    print("="*60)
    print("BASELOAD RENEWABLES MODEL")
    print("="*60)
    print(f"Configuration:")
    print(f"  Target sites: {n_sites}")
    print(f"  Workers: {n_workers}")
    print(f"  API source: {api_source}")
    print(f"  Save hourly data: {save_hourly}")
    if max_sites:
        print(f"  Max sites (test mode): {max_sites}")
    if max_configs:
        print(f"  Max configs (test mode): {max_configs}")

    start_time = time.time()

    # Step 1: Site selection
    sites_df = step_1_select_sites(n_sites, force=force)

    # Step 2: Resource data from API
    step_2_fetch_resource_data(sites_df, api_source=api_source, max_sites=max_sites, force=force)

    # Step 3: Configurations
    configs_df = step_3_generate_configurations(force=force)

    # Step 4: Simulations
    summary_df = step_4_run_simulations(
        sites_df, configs_df,
        n_workers=n_workers,
        save_hourly=save_hourly,
        max_sites=max_sites,
        max_configs=max_configs
    )

    # Step 5: Save and analyze
    step_5_save_and_analyze(summary_df, sites_df)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)

    return summary_df


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Baseload Renewables Model - Global renewable energy simulation"
    )

    parser.add_argument(
        '--sites', type=int, default=NUM_SITES,
        help=f'Number of sites to select (default: {NUM_SITES})'
    )
    parser.add_argument(
        '--workers', type=int, default=NUM_WORKERS,
        help=f'Number of parallel workers (default: {NUM_WORKERS})'
    )
    parser.add_argument(
        '--api-source', type=str, default='nasa_power',
        choices=['nasa_power', 'renewables_ninja', 'synthetic'],
        help='API source for renewable data (default: nasa_power)'
    )
    parser.add_argument(
        '--no-hourly', action='store_true',
        help='Skip saving hourly timeseries data (saves disk space)'
    )
    parser.add_argument(
        '--max-sites', type=int, default=None,
        help='Limit number of sites (for testing)'
    )
    parser.add_argument(
        '--max-configs', type=int, default=None,
        help='Limit number of configurations (for testing)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force regeneration of all data'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode (3 sites, 10 configs, uses NASA POWER API)'
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.max_sites = 3
        args.max_configs = 10
        args.workers = 1
        print("Running in TEST MODE (3 sites, 10 configs, NASA POWER API)")

    run_full_pipeline(
        n_sites=args.sites,
        n_workers=args.workers,
        api_source=args.api_source,
        save_hourly=not args.no_hourly,
        max_sites=args.max_sites,
        max_configs=args.max_configs,
        force=args.force
    )


if __name__ == "__main__":
    main()
