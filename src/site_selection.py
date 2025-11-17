"""
Site selection module for selecting ~1000 land-only global locations.

Uses Natural Earth data for land mask and k-means clustering on 3D sphere
coordinates for spatial distribution.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_SITES, GRID_RESOLUTION, SITES_DIR


def generate_global_grid(resolution: float = 0.5) -> np.ndarray:
    """
    Generate a global grid of latitude/longitude coordinates.

    Args:
        resolution: Grid spacing in degrees

    Returns:
        Array of shape (N, 2) with [lat, lon] pairs
    """
    lats = np.arange(-90 + resolution/2, 90, resolution)
    lons = np.arange(-180 + resolution/2, 180, resolution)

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten to coordinate pairs
    coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    return coords


def get_land_mask_from_natural_earth() -> gpd.GeoDataFrame:
    """
    Download and load Natural Earth land polygons.

    Returns:
        GeoDataFrame with land polygons
    """
    # Try multiple approaches to get land data

    # Approach 1: Try to use geodatasets if available (geopandas >= 1.0)
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path('naturalearth land'))
        return world
    except (ImportError, Exception):
        pass

    # Approach 2: Try legacy geopandas datasets (geopandas < 1.0)
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        land = world.dissolve()
        return land
    except (AttributeError, Exception):
        pass

    # Approach 3: Use simplified continental bounding boxes as fallback
    print("Using simplified continental boundaries as land mask fallback")
    from shapely.geometry import box, MultiPolygon

    # Approximate continental boundaries (simplified)
    continents = [
        # North America
        box(-170, 15, -50, 72),
        # South America
        box(-82, -56, -34, 12),
        # Europe
        box(-25, 35, 40, 71),
        # Africa
        box(-18, -35, 52, 37),
        # Asia (split into parts for better coverage)
        box(25, 10, 180, 78),
        box(95, -11, 155, 10),  # Southeast Asia/Indonesia
        # Australia/Oceania
        box(110, -45, 180, -10),
        # Antarctica (optional, excluded by lat filter)
        # box(-180, -90, 180, -60),
    ]

    land_geom = MultiPolygon(continents)
    land_gdf = gpd.GeoDataFrame(geometry=[land_geom], crs="EPSG:4326")
    return land_gdf


def filter_land_points(coords: np.ndarray, land_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Filter coordinates to keep only land points.

    Args:
        coords: Array of [lat, lon] pairs
        land_gdf: GeoDataFrame with land polygons

    Returns:
        Array of land-only coordinates
    """
    print(f"Filtering {len(coords)} grid points for land...")

    # Create Points for each coordinate
    points = gpd.GeoSeries([Point(lon, lat) for lat, lon in coords])

    # Get the combined land geometry
    land_geom = land_gdf.geometry.unary_union

    # Check which points are on land (vectorized operation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        is_land = points.within(land_geom)

    land_coords = coords[is_land.values]
    print(f"Found {len(land_coords)} land points")

    return land_coords


def latlon_to_xyz(coords: np.ndarray) -> np.ndarray:
    """
    Convert latitude/longitude to 3D unit sphere coordinates.

    Args:
        coords: Array of [lat, lon] pairs in degrees

    Returns:
        Array of [x, y, z] coordinates on unit sphere
    """
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack([x, y, z])


def xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    """
    Convert 3D unit sphere coordinates back to latitude/longitude.

    Args:
        xyz: Array of [x, y, z] coordinates

    Returns:
        Array of [lat, lon] pairs in degrees
    """
    lat_rad = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    lon_rad = np.arctan2(xyz[:, 1], xyz[:, 0])

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    return np.column_stack([lat_deg, lon_deg])


def select_sites_kmeans(land_coords: np.ndarray, n_sites: int = 1000) -> np.ndarray:
    """
    Select n_sites locations using k-means clustering on 3D sphere.

    Args:
        land_coords: Array of land [lat, lon] pairs
        n_sites: Number of sites to select

    Returns:
        Array of selected [lat, lon] pairs
    """
    print(f"Running k-means clustering to select {n_sites} sites...")

    # Convert to 3D coordinates
    xyz = latlon_to_xyz(land_coords)

    # Run k-means
    kmeans = KMeans(n_clusters=n_sites, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(xyz)

    # Get cluster centers
    centers_xyz = kmeans.cluster_centers_

    # Normalize to unit sphere (k-means centers may not be exactly on sphere)
    norms = np.linalg.norm(centers_xyz, axis=1, keepdims=True)
    centers_xyz = centers_xyz / norms

    # Convert back to lat/lon
    centers_latlon = xyz_to_latlon(centers_xyz)

    return centers_latlon


def snap_to_nearest_land(centers: np.ndarray, land_coords: np.ndarray) -> np.ndarray:
    """
    Snap cluster centers to nearest actual land points.

    This ensures all selected sites are definitely on land.

    Args:
        centers: Array of cluster center [lat, lon] pairs
        land_coords: Array of all land [lat, lon] pairs

    Returns:
        Array of snapped [lat, lon] pairs
    """
    print("Snapping cluster centers to nearest land points...")

    # Convert both to 3D for distance calculation
    centers_xyz = latlon_to_xyz(centers)
    land_xyz = latlon_to_xyz(land_coords)

    # Find nearest land point for each center
    snapped = np.zeros_like(centers)

    for i, center in enumerate(centers_xyz):
        # Calculate distances (on sphere, use dot product)
        distances = np.sum((land_xyz - center) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        snapped[i] = land_coords[nearest_idx]

    return snapped


def select_global_sites(n_sites: int = NUM_SITES, resolution: float = GRID_RESOLUTION) -> pd.DataFrame:
    """
    Main function to select globally distributed land sites.

    Args:
        n_sites: Number of sites to select
        resolution: Grid resolution for initial land mask

    Returns:
        DataFrame with site_id, lat_deg, lon_deg
    """
    print(f"Selecting {n_sites} globally distributed land sites...")

    # Step 1: Generate global grid
    grid_coords = generate_global_grid(resolution)
    print(f"Generated {len(grid_coords)} grid points at {resolution}° resolution")

    # Step 2: Get land mask
    print("Loading Natural Earth land data...")
    land_gdf = get_land_mask_from_natural_earth()

    # Step 3: Filter to land points
    land_coords = filter_land_points(grid_coords, land_gdf)

    if len(land_coords) < n_sites:
        raise ValueError(f"Only {len(land_coords)} land points found, need {n_sites}")

    # Step 4: K-means clustering
    selected_centers = select_sites_kmeans(land_coords, n_sites)

    # Step 5: Snap to actual land points (ensures on land)
    selected_sites = snap_to_nearest_land(selected_centers, land_coords)

    # Create DataFrame
    sites_df = pd.DataFrame({
        'site_id': range(n_sites),
        'lat_deg': selected_sites[:, 0],
        'lon_deg': selected_sites[:, 1]
    })

    print(f"Selected {len(sites_df)} land-based sites")
    print(f"Latitude range: {sites_df['lat_deg'].min():.2f} to {sites_df['lat_deg'].max():.2f}")
    print(f"Longitude range: {sites_df['lon_deg'].min():.2f} to {sites_df['lon_deg'].max():.2f}")

    return sites_df


def save_sites(sites_df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Save selected sites to CSV.

    Args:
        sites_df: DataFrame with site information
        output_path: Path to save file (default: SITES_DIR/sites.csv)

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = SITES_DIR / "sites.csv"

    sites_df.to_csv(output_path, index=False)
    print(f"Saved {len(sites_df)} sites to {output_path}")

    return output_path


def load_sites(input_path: Path = None) -> pd.DataFrame:
    """
    Load previously selected sites from CSV.

    Args:
        input_path: Path to CSV file (default: SITES_DIR/sites.csv)

    Returns:
        DataFrame with site information
    """
    if input_path is None:
        input_path = SITES_DIR / "sites.csv"

    sites_df = pd.read_csv(input_path)
    print(f"Loaded {len(sites_df)} sites from {input_path}")

    return sites_df


if __name__ == "__main__":
    # Run site selection
    sites_df = select_global_sites()
    save_sites(sites_df)

    # Display sample
    print("\nSample of selected sites:")
    print(sites_df.head(10))
