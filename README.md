# Baseload Renewables Model

A global renewable energy baseload simulation model that analyzes solar, wind, and battery storage configurations across 1000+ land-based locations worldwide.

## Overview

This model:
1. Selects ~1000 spatially-distributed land locations globally using k-means clustering
2. Retrieves hourly solar and wind capacity factors from Renewables.ninja API
3. Simulates 375 different capacity configurations (solar: 1-5 GW, wind: 1-5 GW, battery: 1-15 GWh)
4. Runs battery dispatch simulation to serve a constant 1 GW baseload
5. Outputs hourly timeseries and summary statistics

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
BaseloadRenewables/
├── src/
│   ├── site_selection.py      # Land mask and k-means site selection
│   ├── api_client.py          # Renewables.ninja API client
│   ├── configurations.py      # Capacity configuration generator
│   ├── dispatch_simulation.py # Battery dispatch simulation engine
│   ├── runner.py              # Scalable simulation runner
│   └── data_output.py         # Data storage and output handlers
├── data/
│   ├── sites/                 # Selected site coordinates
│   ├── resource/              # API-fetched renewable profiles
│   ├── timeseries/            # Hourly simulation outputs
│   └── summary/               # Summary statistics
├── main.py                    # Main orchestration script
├── config.py                  # Global configuration
└── requirements.txt
```

## Usage

### Full Production Run (1000 sites, 375,000 simulations)

```bash
# Full run with 1000 sites (takes ~2-3 hours)
python main.py --workers 8

# This will:
# - Select 1000 globally distributed land sites
# - Fetch 2000 API calls from NASA POWER (solar + wind per site)
# - Run 375,000 simulations (1000 sites × 375 configurations)
# - Generate CSV exports and visualization plots
```

**Note:** The full run requires ~2-3 hours:
- API calls: ~35 minutes (2000 calls @ 1 sec/call)
- Simulations: ~10-15 minutes with 8 workers
- Visualization: ~2-3 minutes

### Quick Demonstration (50 sites)

```bash
# Run with 50 sites for quick results (~6 minutes)
python main.py --max-sites 50 --workers 4
```

### Test Mode (3 sites)

```bash
# Quick test with 3 sites (~1 minute)
python main.py --test
```

### Individual Steps

```bash
# Run individual pipeline steps
python -m src.site_selection    # Select 1000 land sites
python -m src.api_client        # Fetch renewable data
python -m src.runner            # Run simulations
```

## Configuration

Edit `config.py` to adjust:
- Number of sites (default: 1000)
- Simulation year (default: 2022)
- Capacity ranges
- Battery parameters
- API settings

## Output

### CSV Files (in `data/summary/`)

- `summary.csv`: Complete results for all 375,000 site-config pairs (18,750 rows for 50 sites)
- `best_configurations_per_site.csv`: Best performing configuration for each site
- `site_statistics.csv`: Aggregate statistics per site
- `configuration_statistics.csv`: Average performance across all sites per configuration

### Parquet Files (in `data/summary/` and `data/timeseries/`)

- `summary.parquet`: Complete results (compressed)
- `site_*.parquet`: Hourly dispatch timeseries for each site (8760 hours × configs)

### Visualization Charts (in `data/plots/`)

- `global_site_map.png`: World map showing all selected sites colored by performance
- `capacity_vs_reliability.png`: Analysis of capacity requirements vs reliability targets
- `geographic_analysis.png`: Geographic patterns in renewable resource quality
- `pareto_frontier.png`: Cost-reliability trade-off curves
- `configuration_heatmap.png`: Performance heatmaps for different solar/wind/battery combinations

### Resource Data (in `data/resource/`)

- `site_*.parquet`: Hourly capacity factors from NASA POWER API (8760 hours per site)
