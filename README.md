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

## Interactive Web Visualization

An interactive HTML frontend is available for exploring simulation results through a map-based interface. The visualization can be deployed to GitHub Pages for easy sharing and access.

### Features

- **Two Visualization Modes:**
  - **By System Mix:** Filter locations by specific solar/wind/battery configurations
  - **By Max LCOE:** View best-performing configurations under a specified LCOE threshold
- **Voronoi Heatmap:** Smooth color-coded regions showing system capacity factor across the globe
- **Interactive Tooltips:** Click markers to see detailed performance metrics and optimal configurations
- **Color-Coded Legend:** Visual gradient from blue (0% CF) through green, orange, red to dark red (100% CF)
- **Responsive Design:** Works on desktop and mobile devices

### Usage

1. **Open the visualization:**
   ```bash
   # Open index.html in your web browser
   open index.html  # macOS
   xdg-open index.html  # Linux
   start index.html  # Windows
   ```

2. **Upload your data:**
   - Click "Upload your LCOE File"
   - Select `data/summary/summary.csv` from your simulation results

3. **Explore the results:**
   - **Mode 1 - By System Mix:**
     - Use the sliders to select specific solar (0-10 GW), wind (0-4 GW), and battery (0-15 GWh) capacities
     - The map updates to show system capacity factor for all locations with that configuration

   - **Mode 2 - By Max LCOE:**
     - Use the slider to set a maximum LCOE threshold ($20-200/MWh)
     - The map shows the best capacity factor achievable at each location under that LCOE limit
     - Tooltips reveal the optimal solar/wind/battery mix for each location

4. **Interact with the map:**
   - Zoom and pan to explore different regions
   - Hover over markers to see detailed performance metrics
   - The Voronoi heatmap provides a smooth visualization of performance gradients

### Technical Details

The frontend uses:
- **Leaflet.js** for the interactive map with dark theme tiles
- **D3.js** for Voronoi diagram generation clipped to land masses
- **Tailwind CSS** for responsive styling
- **GeoJSON** for world country boundaries (loaded from public dataset)

The visualization runs entirely in the browser with no backend required.

### GitHub Pages Deployment

Deploy the visualization to GitHub Pages for easy sharing:

#### Initial Setup

1. **Enable GitHub Pages** for your repository:
   - Go to your repository settings on GitHub
   - Navigate to **Pages** in the left sidebar
   - Under **Source**, select **Deploy from a branch**
   - Choose **main** branch and **/ (root)** folder
   - Click **Save**

2. **Update the data** (after running simulations):
   ```bash
   # Copy your simulation results to the root directory
   cp data/summary/summary.csv ./summary.csv

   # Add and commit the data
   git add summary.csv
   git commit -m "Update visualization data with latest simulation results"
   git push
   ```

3. **Access your visualization**:
   - Your site will be available at: `https://<username>.github.io/<repository-name>/`
   - Example: `https://dmw24.github.io/BaseloadRenewables/`

#### Updating Data

The visualization automatically loads `summary.csv` from the repository root. To update:

```bash
# Run new simulations
python main.py --max-sites 100

# Replace the web data
cp data/summary/summary.csv ./summary.csv

# Commit and push
git add summary.csv
git commit -m "Update data: $(date +%Y-%m-%d)"
git push
```

GitHub Pages will automatically rebuild and deploy within 1-2 minutes.

#### Data Size Considerations

- GitHub has a 100 MB file size limit
- For large datasets (1000 sites × 375 configs = ~375,000 rows):
  - The CSV file may be 20-50 MB (acceptable)
  - If too large, filter to key configurations or sample sites
  - Alternatively, use GitHub Large File Storage (LFS)

#### Alternative: Local Deployment

If you prefer local deployment without GitHub Pages:

```bash
# Simple Python HTTP server
python -m http.server 8000

# Or using Node.js
npx http-server

# Then open: http://localhost:8000
```
