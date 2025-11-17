"""
Visualization module for creating charts and plots from simulation results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, LOAD_GW


def setup_plotting_style():
    """Set up matplotlib plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_global_site_map(sites_df: pd.DataFrame, summary_df: pd.DataFrame = None,
                         output_path: Path = None) -> Path:
    """
    Create a global map showing selected sites.

    Args:
        sites_df: DataFrame with site coordinates
        summary_df: Optional summary with best configuration per site
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = DATA_DIR / "plots" / "global_site_map.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot all sites
    scatter = ax.scatter(sites_df['lon_deg'], sites_df['lat_deg'],
                        c='blue', s=20, alpha=0.6, label='Selected Sites')

    # If summary provided, color by performance
    if summary_df is not None:
        # Get best config per site
        best_per_site = summary_df.groupby('site_id').apply(
            lambda x: x.loc[x['energy_served_frac'].idxmax()]
        ).reset_index(drop=True)

        scatter = ax.scatter(best_per_site['lon_deg'], best_per_site['lat_deg'],
                           c=best_per_site['energy_served_frac'],
                           s=30, cmap='RdYlGn', vmin=0, vmax=1,
                           edgecolors='black', linewidths=0.5,
                           label='Best Config Performance')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Max Energy Served Fraction', rotation=270, labelpad=20)

    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(f'Global Distribution of {len(sites_df)} Land Sites')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved global site map to {output_path}")
    return output_path


def plot_capacity_vs_reliability(summary_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Plot total capacity vs reliability (energy served fraction).

    Args:
        summary_df: Summary DataFrame
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = DATA_DIR / "plots" / "capacity_vs_reliability.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate total capacity
    summary_df['total_capacity_GW'] = summary_df['C_solar_GW'] + summary_df['C_wind_GW']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Total capacity vs energy served
    ax = axes[0, 0]
    for battery_size in sorted(summary_df['E_bat_GWh'].unique())[:3]:  # Show first 3 battery sizes
        data = summary_df[summary_df['E_bat_GWh'] == battery_size]
        ax.scatter(data['total_capacity_GW'], data['energy_served_frac'],
                  alpha=0.3, s=10, label=f'Battery: {battery_size} GWh')
    ax.set_xlabel('Total Renewable Capacity (GW)')
    ax.set_ylabel('Energy Served Fraction')
    ax.set_title('Renewable Capacity vs Reliability')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
    ax.axhline(y=0.99, color='g', linestyle='--', label='99% Target')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Battery size vs energy served
    ax = axes[0, 1]
    for total_cap in [2, 4, 6, 8, 10]:  # Show several capacity levels
        data = summary_df[summary_df['total_capacity_GW'] == total_cap]
        if len(data) > 0:
            ax.scatter(data['E_bat_GWh'], data['energy_served_frac'],
                      alpha=0.3, s=10, label=f'Total: {total_cap} GW')
    ax.set_xlabel('Battery Capacity (GWh)')
    ax.set_ylabel('Energy Served Fraction')
    ax.set_title('Battery Capacity vs Reliability')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Solar/wind ratio vs reliability
    ax = axes[1, 0]
    summary_df['solar_fraction'] = summary_df['C_solar_GW'] / summary_df['total_capacity_GW']
    scatter = ax.scatter(summary_df['solar_fraction'], summary_df['energy_served_frac'],
                        c=summary_df['E_bat_GWh'], s=10, alpha=0.3, cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Battery (GWh)')
    ax.set_xlabel('Solar Fraction')
    ax.set_ylabel('Energy Served Fraction')
    ax.set_title('Solar/Wind Mix vs Reliability')
    ax.grid(True, alpha=0.3)

    # 4. Cost proxy analysis
    ax = axes[1, 1]
    # Simplified cost proxy: $1000/kW solar, $1500/kW wind, $200/kWh battery
    summary_df['cost_proxy_M$'] = (summary_df['C_solar_GW'] * 1000 +
                                    summary_df['C_wind_GW'] * 1500 +
                                    summary_df['E_bat_GWh'] * 200)
    scatter = ax.scatter(summary_df['cost_proxy_M$'], summary_df['energy_served_frac'],
                        c=summary_df['total_capacity_GW'], s=10, alpha=0.3, cmap='plasma')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Capacity (GW)')
    ax.set_xlabel('Approximate Cost (M$)')
    ax.set_ylabel('Energy Served Fraction')
    ax.set_title('Cost vs Reliability')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved capacity vs reliability plot to {output_path}")
    return output_path


def plot_geographic_analysis(summary_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Plot geographic analysis of renewable potential.

    Args:
        summary_df: Summary DataFrame with lat/lon
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = DATA_DIR / "plots" / "geographic_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get best performance per site
    best_per_site = summary_df.groupby('site_id').apply(
        lambda x: x.loc[x['energy_served_frac'].idxmax()]
    ).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Best energy served by location
    ax = axes[0, 0]
    scatter = ax.scatter(best_per_site['lon_deg'], best_per_site['lat_deg'],
                        c=best_per_site['energy_served_frac'],
                        s=30, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                        edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Best Energy Served')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Best Achievable Reliability by Location')
    ax.grid(True, alpha=0.3)

    # 2. Required capacity by latitude
    ax = axes[0, 1]
    best_per_site['total_capacity'] = best_per_site['C_solar_GW'] + best_per_site['C_wind_GW']
    scatter = ax.scatter(best_per_site['lat_deg'], best_per_site['total_capacity'],
                        c=best_per_site['energy_served_frac'],
                        s=30, cmap='RdYlGn', alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy Served')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Total Capacity (GW)')
    ax.set_title('Required Capacity vs Latitude')
    ax.grid(True, alpha=0.3)

    # 3. Battery requirements by location
    ax = axes[1, 0]
    scatter = ax.scatter(best_per_site['lon_deg'], best_per_site['lat_deg'],
                        c=best_per_site['E_bat_GWh'],
                        s=30, cmap='plasma',
                        edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Battery (GWh)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Battery Requirements by Location')
    ax.grid(True, alpha=0.3)

    # 4. Average system CF by latitude
    ax = axes[1, 1]
    scatter = ax.scatter(best_per_site['lat_deg'], best_per_site['avg_system_cf'],
                        c=best_per_site['energy_served_frac'],
                        s=30, cmap='RdYlGn', alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy Served')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Average System CF')
    ax.set_title('Renewable Resource Quality vs Latitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved geographic analysis plot to {output_path}")
    return output_path


def plot_pareto_frontier(summary_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Plot Pareto frontier of cost vs reliability.

    Args:
        summary_df: Summary DataFrame
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = DATA_DIR / "plots" / "pareto_frontier.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate cost proxy
    summary_df['cost_proxy_M$'] = (summary_df['C_solar_GW'] * 1000 +
                                    summary_df['C_wind_GW'] * 1500 +
                                    summary_df['E_bat_GWh'] * 200)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. All configurations
    ax = axes[0]
    for site_id in np.random.choice(summary_df['site_id'].unique(), min(10, len(summary_df['site_id'].unique())), replace=False):
        site_data = summary_df[summary_df['site_id'] == site_id].sort_values('cost_proxy_M$')
        ax.plot(site_data['cost_proxy_M$'], site_data['energy_served_frac'],
               alpha=0.3, linewidth=1)

    ax.set_xlabel('Approximate Cost (M$)')
    ax.set_ylabel('Energy Served Fraction')
    ax.set_title('Cost vs Reliability (10 Random Sites)')
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Target')
    ax.axhline(y=0.99, color='g', linestyle='--', alpha=0.5, label='99% Target')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Average across all sites
    ax = axes[1]
    config_avg = summary_df.groupby(['C_solar_GW', 'C_wind_GW', 'E_bat_GWh']).agg({
        'energy_served_frac': 'mean',
        'cost_proxy_M$': 'first'
    }).reset_index().sort_values('cost_proxy_M$')

    ax.scatter(config_avg['cost_proxy_M$'], config_avg['energy_served_frac'],
              s=30, alpha=0.5, c='blue')
    ax.set_xlabel('Approximate Cost (M$)')
    ax.set_ylabel('Mean Energy Served Fraction')
    ax.set_title('Average Cost vs Reliability Across All Sites')
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Target')
    ax.axhline(y=0.99, color='g', linestyle='--', alpha=0.5, label='99% Target')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Pareto frontier plot to {output_path}")
    return output_path


def plot_configuration_heatmap(summary_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Create heatmaps showing configuration performance.

    Args:
        summary_df: Summary DataFrame
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = DATA_DIR / "plots" / "configuration_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Select battery sizes to show
    battery_sizes = [1, 5, 10]

    for idx, bat_size in enumerate(battery_sizes):
        data = summary_df[summary_df['E_bat_GWh'] == bat_size]

        # Average across all sites
        pivot = data.groupby(['C_solar_GW', 'C_wind_GW'])['energy_served_frac'].mean().reset_index()
        pivot_table = pivot.pivot(index='C_wind_GW', columns='C_solar_GW', values='energy_served_frac')

        # Top row: Energy served
        ax = axes[0, idx]
        im = ax.imshow(pivot_table, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Solar Capacity (GW)')
        ax.set_ylabel('Wind Capacity (GW)')
        ax.set_title(f'Energy Served - Battery: {bat_size} GWh')
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        # Bottom row: Curtailment
        pivot_curt = data.groupby(['C_solar_GW', 'C_wind_GW'])['total_curtailed_GWh'].mean().reset_index()
        pivot_table_curt = pivot_curt.pivot(index='C_wind_GW', columns='C_solar_GW', values='total_curtailed_GWh')

        ax = axes[1, idx]
        im = ax.imshow(pivot_table_curt, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Solar Capacity (GW)')
        ax.set_ylabel('Wind Capacity (GW)')
        ax.set_title(f'Curtailment (GWh/yr) - Battery: {bat_size} GWh')
        ax.set_xticks(range(len(pivot_table_curt.columns)))
        ax.set_xticklabels(pivot_table_curt.columns)
        ax.set_yticks(range(len(pivot_table_curt.index)))
        ax.set_yticklabels(pivot_table_curt.index)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved configuration heatmap to {output_path}")
    return output_path


def generate_all_plots(sites_df: pd.DataFrame, summary_df: pd.DataFrame,
                      output_dir: Path = None) -> List[Path]:
    """
    Generate all visualization plots.

    Args:
        sites_df: Sites DataFrame
        summary_df: Summary DataFrame
        output_dir: Output directory for plots

    Returns:
        List of paths to generated plots
    """
    if output_dir is None:
        output_dir = DATA_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plotting_style()

    print("\nGenerating visualization plots...")

    plots = []

    try:
        plots.append(plot_global_site_map(sites_df, summary_df, output_dir / "global_site_map.png"))
    except Exception as e:
        print(f"Warning: Could not create site map: {e}")

    try:
        plots.append(plot_capacity_vs_reliability(summary_df, output_dir / "capacity_vs_reliability.png"))
    except Exception as e:
        print(f"Warning: Could not create capacity plot: {e}")

    try:
        plots.append(plot_geographic_analysis(summary_df, output_dir / "geographic_analysis.png"))
    except Exception as e:
        print(f"Warning: Could not create geographic analysis: {e}")

    try:
        plots.append(plot_pareto_frontier(summary_df, output_dir / "pareto_frontier.png"))
    except Exception as e:
        print(f"Warning: Could not create Pareto frontier: {e}")

    try:
        plots.append(plot_configuration_heatmap(summary_df, output_dir / "configuration_heatmap.png"))
    except Exception as e:
        print(f"Warning: Could not create configuration heatmap: {e}")

    print(f"\nGenerated {len(plots)} visualization plots in {output_dir}")

    return plots


if __name__ == "__main__":
    # Test with sample data
    print("Testing visualization module...")

    from src.site_selection import load_sites
    from src.runner import load_summary

    sites_df = load_sites()
    summary_df = load_summary()

    generate_all_plots(sites_df, summary_df)

    print("Visualization test complete!")
