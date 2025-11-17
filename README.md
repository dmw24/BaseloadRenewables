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

```bash
# Run complete pipeline
python main.py

# Or run individual steps
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

- `data/summary/summary.parquet`: Summary metrics for all site-config pairs
- `data/timeseries/site_*.parquet`: Hourly dispatch data per site
