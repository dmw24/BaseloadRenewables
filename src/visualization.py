"""
Visualization module for creating charts and plots from simulation results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List
import geopandas as gpd
from shapely.geometry import Point

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
    Create a global map showing selected sites with world coastlines.

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

    fig, ax = plt.subplots(figsize=(20, 12))

    # Load world map
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except:
        # Fallback: create simple world boundaries
        from shapely.geometry import box
        world = gpd.GeoDataFrame(geometry=[box(-180, -90, 180, 90)], crs="EPSG:4326")

    # Plot world map
    world.boundary.plot(ax=ax, linewidth=0.8, color='black', alpha=0.5)
    world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.3)

    # If summary provided, color by performance
    if summary_df is not None:
        # Get best config per site
        best_per_site = summary_df.groupby('site_id').apply(
            lambda x: x.loc[x['energy_served_frac'].idxmax()],
            include_groups=False
        ).reset_index()

        # Create color map
        scatter = ax.scatter(best_per_site['lon_deg'], best_per_site['lat_deg'],
                           c=best_per_site['energy_served_frac'],
                           s=60, cmap='RdYlGn', vmin=0, vmax=1,
                           edgecolors='black', linewidths=1.0,
                           alpha=0.8, zorder=5)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Best Energy Served Fraction', rotation=270, labelpad=25, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    else:
        # Plot all sites
        ax.scatter(sites_df['lon_deg'], sites_df['lat_deg'],
                  c='blue', s=40, alpha=0.6, edgecolors='black',
                  linewidths=0.5, zorder=5, label='Selected Sites')
        ax.legend(fontsize=12)

    ax.set_xlabel('Longitude (degrees)', fontsize=14)
    ax.set_ylabel('Latitude (degrees)', fontsize=14)
    ax.set_title(f'Global Distribution of {len(sites_df)} Land-Based Sites for Baseload Renewable Energy',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add tick marks
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
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

    # Calculate total capacity (solar only)
    summary_df['total_capacity_GW'] = summary_df['C_solar_GW']

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

    # 3. Solar capacity vs reliability
    ax = axes[1, 0]
    scatter = ax.scatter(summary_df['C_solar_GW'], summary_df['energy_served_frac'],
                        c=summary_df['E_bat_GWh'], s=10, alpha=0.3, cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Battery (GWh)')
    ax.set_xlabel('Solar Capacity (GW)')
    ax.set_ylabel('Energy Served Fraction')
    ax.set_title('Solar Capacity vs Reliability')
    ax.grid(True, alpha=0.3)

    # 4. Cost proxy analysis
    ax = axes[1, 1]
    # Simplified cost proxy: $1000/kW solar, $200/kWh battery
    summary_df['cost_proxy_M$'] = (summary_df['C_solar_GW'] * 1000 +
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
    Plot geographic analysis of renewable potential with world map background.

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
        lambda x: x.loc[x['energy_served_frac'].idxmax()],
        include_groups=False
    ).reset_index()

    # Load world map
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except:
        from shapely.geometry import box
        world = gpd.GeoDataFrame(geometry=[box(-180, -90, 180, 90)], crs="EPSG:4326")

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # 1. Best energy served by location
    ax = axes[0, 0]
    world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.3, alpha=0.3)
    scatter = ax.scatter(best_per_site['lon_deg'], best_per_site['lat_deg'],
                        c=best_per_site['energy_served_frac'],
                        s=50, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                        edgecolors='black', linewidths=0.8, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Best Energy Served', fontsize=11)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Best Achievable Reliability by Location', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # 2. Required capacity by latitude
    ax = axes[0, 1]
    best_per_site['total_capacity'] = best_per_site['C_solar_GW']
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
    world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.3, alpha=0.3)
    scatter = ax.scatter(best_per_site['lon_deg'], best_per_site['lat_deg'],
                        c=best_per_site['E_bat_GWh'],
                        s=50, cmap='plasma',
                        edgecolors='black', linewidths=0.8, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Battery (GWh)', fontsize=11)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Battery Requirements by Location', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

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
    config_avg = summary_df.groupby(['C_solar_GW', 'E_bat_GWh']).agg({
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

        # Average across all sites - now showing Solar vs Battery
        pivot = data.groupby('C_solar_GW')['energy_served_frac'].mean().reset_index()

        # Top row: Energy served by solar capacity
        ax = axes[0, idx]
        ax.bar(pivot['C_solar_GW'], pivot['energy_served_frac'], color='green', alpha=0.7)
        ax.set_xlabel('Solar Capacity (GW)')
        ax.set_ylabel('Energy Served Fraction')
        ax.set_title(f'Energy Served - Battery: {bat_size} GWh')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (solar, frac) in enumerate(zip(pivot['C_solar_GW'], pivot['energy_served_frac'])):
            ax.text(solar, frac, f'{frac:.2f}', ha='center', va='bottom', fontsize=8)

        # Bottom row: Curtailment by solar capacity
        pivot_curt = data.groupby('C_solar_GW')['total_curtailed_GWh'].mean().reset_index()

        ax = axes[1, idx]
        ax.bar(pivot_curt['C_solar_GW'], pivot_curt['total_curtailed_GWh'], color='red', alpha=0.7)
        ax.set_xlabel('Solar Capacity (GW)')
        ax.set_ylabel('Curtailment (GWh/yr)')
        ax.set_title(f'Curtailment - Battery: {bat_size} GWh')
        ax.grid(True, alpha=0.3, axis='y')

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
