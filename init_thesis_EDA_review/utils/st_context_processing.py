import h3
import duckdb
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

import json
import os
import bisect
from typing import Optional, List, Dict, Tuple, Any

load_dotenv()

METADATA_PATH = os.getenv("ST_METADATA_PATH", "../")
metadata_file = os.path.join(METADATA_PATH, "spatio-temporal_metadata.json")
with open(metadata_file, "r") as f:
    st_metadata = json.load(f)
ZSTD_COMPRESSION = int(os.getenv('ZSTD_PARQUET_COMPRESSION_LEVEL', 10))
DB_PATH = os.getenv('DUCKDB_DIR', os.path.join(os.getcwd(), 'datasets', 'db'))

def get_duckdb_connection(db_file: str):

    """
    Get a DuckDB connection with the required extensions loaded.
    
    Args:
        db_file: Path to the DuckDB database file.
        
    Returns:
        DuckDB connection object.
    """
    db_conn = duckdb.connect(database=db_file)
    # make sure we have the required extensions
    db_conn.install_extension("spatial")
    db_conn.load_extension("spatial")
    db_conn.install_extension("httpfs")
    db_conn.load_extension("httpfs")
    db_conn.install_extension("h3", repository="community")
    db_conn.load_extension("h3")
    
    return db_conn

# Add an H3 column to a DuckDB table and populate it with H3 indices.
def ddb_alter_table_add_h3(
    db_file: str,
    table_name: str,
    h3_resolution: int,
    h3_col_name: str = None
):


    db_conn = get_duckdb_connection(db_file)
    h3_col_name = f"h3_res_{h3_resolution}" if h3_col_name is None else h3_col_name

    # add the H3 column to the table
    db_conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {h3_col_name} UINT64")
    # get lat, lng from geometry centroids to avoid field name inconsistencies and excessive arguments
    db_conn.execute(f"UPDATE {table_name} SET {h3_col_name} = h3_latlng_to_cell(ST_X(ST_Centroid(geometry)), ST_Y(ST_Centroid(geometry)), {h3_resolution})")

    # check if the column was added and populated
    col_result = db_conn.execute(f"SELECT count(*) FROM {table_name} WHERE {h3_col_name} IS NOT NULL").fetchone()[0]
    db_conn.close()

    if col_result > 0:
        print(f"Added and populated {h3_col_name} column in {table_name} table")
        return True
    else:
        raise ValueError(f"h3 col exists but not populated in {table_name} table")
        return False

def ddb_save_subtype_geoms(
    db_file: str,
    div_table: str,
    pv_table: str,
    division_type = "country",
    filter_dict: Optional[Dict[str, Tuple[str, Any]]] = None,
):
    # connect to the database
    db_conn = duckdb.connect(database=db_file)

    # by default, we will save all the country geometries
    table_name = f"ov_divisions_{division_type}_geoms"

    where_clause = "WHERE ov.subtype = 'country'" if division_type == "country" else f"WHERE ov.subtype = '{division_type}' and pv.division_subtype = '{division_type}'"
    # Get the list of country geometries
    division_geoms = f"""
    SELECT ov.division_id, ov.country as country_iso, names.primary AS division_name, ov.bbox, ov.geometry,
    count_if(pv.division_id is not null) AS {division_type}_pv_count,
    SUM(ifnull(pv.area_m2, 0.0)) AS {division_type}_pv_area_m2
    FROM {div_table} ov
    LEFT JOIN {pv_table} AS pv ON ov.country = pv.division_country
    {where_clause}
    GROUP BY ov.division_id, ov.country, names.primary, ov.bbox, ov.geometry
    """

    try:
        create_table_sql = f"""
        CREATE OR REPLACE TABLE {division_type}_geoms AS
        {division_geoms}
        """
        db_conn.execute(create_table_sql)
        print(f"Created table {table_name} with {division_type} geometries")
        return True
    except Exception as e:
        print(f"Error creating table for {division_type}: {e}")
        return False

# assumes both tables already have columns with a shared h3 index resolution
def ddb_save_div_matches(
    db_file: str,
    labels_table: str,
    divisions_table: str,
    division_subtypes: List[str],
    h3_col_name: str = None,
    use_bbox: bool = False,
    force : bool = False
):
    db_conn = get_duckdb_connection(db_file)

    cols_to_add = ["division_id", "division_country", "division_region", "division_name", "div_bbox", "division_subtype"]
    # check if the columns already exist
    existing_cols = db_conn.execute(f"""
    SELECT DISTINCT column_name
    FROM information_schema.columns
    WHERE table_name = '{{ov_divisions_table}}' AND table_schema = 'main' ORDER BY ordinal_position;""").fetchall()
    if all(col in existing_cols for col in cols_to_add) and h3_col_name in existing_cols and not force:
        print(f"Columns already exist in {labels_table} table, skipping spatial join")
        print("You can override this behavior by setting force=True")
        return True

    join_clause = None
    if h3_col_name is None and use_bbox:
        # use the bounding box for the join
        join_clause = f"""
        ON ST_Intersects(ST_MakeEnvelope(pv.bbox.xmin, pv.bbox.ymin, pv.bbox.xmax, pv.bbox.ymax), ST_MakeEnvelope(ov.bbox.xmin, ov.bbox.ymin, ov.bbox.xmax, ov.bbox.ymax))
        """
    elif h3_col_name is not None:
        join_clause = f"ON pv.{h3_col_name} = ov.{h3_col_name}"
    else:
        print(f"ERROR! Divisions table must be joined on h3 index column {h3_col_name} or use_bbox=True")
        return False

    # alter our labels table with 3 new columns from matching divisions: division_id, names.primary, bbox as div_bbox, and subtype
    div_spatial_join = f"""
    SELECT
        pv.*,
        ov.division_id,
        ov.country as division_country,
        ov.region as division_region,
        ov.names.primary AS division_name, -- save the primary name from struct
        ov.bbox as div_bbox,
        ov.subtype AS division_subtype
    FROM {labels_table} AS pv
    JOIN {divisions_table} AS ov 
    {join_clause}
    -- Returns true if the first geometry is within the second
    AND ST_Within(pv.geometry, ov.geometry) 
    """
    try:
        # replace the existing table with our query
        db_conn.execute(f"CREATE OR REPLACE TABLE {labels_table} AS {div_spatial_join}")
        # check if the columns were added and populated
        col_result = db_conn.execute(f"SELECT count(*) FROM {labels_table} WHERE division_id IS NOT NULL").fetchone()[0]
        if col_result > 0:
            print(f"Added and populated metadata fields for {col_result} Overture division areas")
            return True
        else:
            raise ValueError(f"Division metadata fields not populated in {labels_table} table")
    except Exception as e:
        print(f"Error performing spatial join between {labels_table} and {divisions_table}: {str(e)}")
        raise ValueError(f"Failed to add division metadata to {labels_table}: {str(e)}")
        return False


def _get_h3_coverage(geom, resolution: int) -> List[str]:
    """Get all H3 cells that cover a geometry"""
    if geom.is_empty:
        return []
    
    # Convert to GeoJSON format for H3 polyfill
    try:
        # Get GeoJSON coordinates - handling both polygons and multipolygons
        if geom.geom_type == 'Polygon':
            coords = [list(geom.exterior.coords)]
            h3_cells = h3.LatLngPoly(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            h3_cells = []
            polys_coords = []
            for poly in geom.geoms:
                coords = [list(poly.exterior.coords)]
                polys_coords.append(coords)
                h3_cells.extend(cells)
            cells = h3.LatLngMultiPoly(polys_coords)
        else:
            # For points or lines, just use the centroid
            h3_cells = [h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, resolution)]
            
        return list(h3_cells)
    except Exception as e:
        print(f"Error getting H3 coverage: {e}")
        return []

def _get_representative_h3_cell(geom, resolution: int) -> str:
    """Get the H3 cell with the largest overlap with the geometry"""
    cells = _get_h3_coverage(geom, resolution)
    if not cells:
        # If no cells, use centroid
        return h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, resolution)
    
    # For simple cases, just return the first cell
    if len(cells) == 1:
        return cells[0]
    
    # For complex cases, get the cell with the largest overlap
    # This is computationally more expensive, but more accurate
    max_overlap = 0
    best_cell = cells[0]
    
    for cell in cells:
        cell_poly = Polygon(h3.h3_to_geo_boundary(cell, geo_json=True))
        overlap_area = geom.intersection(cell_poly).area
        if overlap_area > max_overlap:
            max_overlap = overlap_area
            best_cell = cell
    
    return best_cell

def add_h3_index_to_pv_labels(
    gdf: gpd.GeoDataFrame,
    h3_resolution: int = 10,
    method: str = "centroid",
    geom_col: str = "geometry",
    out_col: str = "h3_index"
) -> gpd.GeoDataFrame:
    """
    Add H3 spatial indices to PV installation geometries.
    
    Args:
        gdf: GeoDataFrame containing PV installation geometries
        h3_resolution: Resolution of H3 cells (0-15), higher is more precise. See table here: https://h3geo.org/docs/3.x/core-library/restable
        method: Method to use for H3 assignment
               "centroid" - assign based on centroid location (faster)
               "coverage" - assign all cells that cover the polygon (more accurate)
               "representative" - assign the cell containing the largest portion
        geom_col: Column name for geometry
        out_col: Output column name for H3 index
        
    Returns:
        GeoDataFrame with H3 index column added
    """
    # Create a copy to avoid modifying the original
    result = gdf.copy()
    global st_metadata

    # sorted list of Avg km² per H3 resolution at index
    h3_sqkm_at_res = st_metadata["h3"]["resolution_sqkm_map"]
    
    # Get list of resolutions and their areas, already sorted
    resolutions = list(range(len(h3_sqkm_at_res)))
    sqkm_areas = h3_sqkm_at_res
    
    
    def find_optimal_h3_resolution(area_m2):
        """Find optimal H3 resolution that first fully covers this area"""
        # Convert m2 to km2, with a small buffer factor for safety
        area_km2 = area_m2 / 1_000_000 * 1.05  # Add 5% buffer around label area
        if area_km2 == 0:
            return len(resolutions) - 1  # If area is zero, return the highest resolution assuming a point label
        
        # Find insertion point - this gives us the index where area_km2 would be inserted
        # to maintain sorted order (ascending)
        idx = bisect.bisect_left(sqkm_areas[::-1], area_km2)  # Search in reverse (cells get smaller as resolution increases)
        
        # Convert the index back to our original list ordering and bounds check
        idx = len(sqkm_areas) - 1 - idx
        idx = max(0, min(idx, len(resolutions) - 1))
        return resolutions[idx]
    
    if h3_resolution is None:
        # Calculate optimal resolution for each geometry based on area
        result['optimal_h3_resolution'] = result['area_m2'].apply(find_optimal_h3_resolution)
    else:
        # Use fixed resolution specified by user
        result['optimal_h3_resolution'] = h3_resolution
    
    if method == "centroid":
        # Apply centroid method using the calculated resolution
        result[out_col] = result.apply(
            lambda row: h3.latlng_to_cell(
                lat=row[geom_col].centroid.y,
                lng=row[geom_col].centroid.x,
                res=row['optimal_h3_resolution']
            ),
            axis=1
        )
        
    elif method == "coverage":
        # Use vectorized operations with calculated resolution
        result[f"{out_col}_set"] = result.apply(
            lambda row: _get_h3_coverage(row[geom_col], row['optimal_h3_resolution']),
            axis=1
        )
        # Use the first cell as the primary index
        result[out_col] = result[f"{out_col}_set"].apply(
            lambda cell_set: cell_set[0] if cell_set else None
        )
        
    elif method == "representative":
        # Get most representative cell with calculated resolution
        result[out_col] = result.apply(
            lambda row: _get_representative_h3_cell(row[geom_col], row['optimal_h3_resolution']),
            axis=1
        )
    
    else:
        raise ValueError(f"Unknown h3 cell assignment method: {method}. Use 'centroid', 'coverage', or 'representative'")
    
    # Drop the temporary column
    # if 'optimal_h3_resolution' not in gdf.columns and h3_resolution is None:
    #     result = result.drop('optimal_h3_resolution', axis=1)
    
    return result

def group_pv_by_h3_cells(
    gdf: gpd.GeoDataFrame,
    h3_column: str = "h3_index",
    group_cols: List[str] = None
) -> pd.DataFrame:
    """
    Group PV installations by H3 cells for efficient STAC queries.
    
    Args:
        gdf: GeoDataFrame with H3 indices
        h3_column: Column name containing H3 indices
        group_cols: Additional columns to include in grouping
        
    Returns:
        DataFrame with H3 cells and counts/stats of contained PV installations
    """
    if group_cols is None:
        group_cols = []
    
    # Group by H3 cell and count installations
    grouped = gdf.groupby([h3_column] + group_cols).agg({
        'area_m2': ['count', 'sum', 'mean'],
        'geometry': lambda x: gpd.GeoSeries(x).unary_union  # Combine geometries
    }).reset_index()
    
    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Add H3 cell geometry
    grouped['h3_geometry'] = grouped[h3_column].apply(
        lambda h: Polygon(h3.h3_to_geo_boundary(h, geo_json=True))
    )
    
    return grouped

def spatial_join_stac_items_with_h3(
    pv_h3_groups: pd.DataFrame,
    stac_items_gdf: gpd.GeoDataFrame,
    h3_geometry_col: str = "h3_geometry"
) -> gpd.GeoDataFrame:
    """
    Join STAC items with H3 cells containing PV installations.
    
    Args:
        pv_h3_groups: DataFrame with H3 cells and PV installation groups
        stac_items_gdf: GeoDataFrame with STAC items and their geometries
        h3_geometry_col: Column name in pv_h3_groups with H3 cell geometry
        
    Returns:
        GeoDataFrame with STAC items joined to H3 cells
    """
    # Convert pv_h3_groups to GeoDataFrame if it's not already
    if not isinstance(pv_h3_groups, gpd.GeoDataFrame):
        pv_h3_groups = gpd.GeoDataFrame(
            pv_h3_groups, 
            geometry=h3_geometry_col,
            crs="EPSG:4326"
        )
    
    # Perform spatial join
    joined = gpd.sjoin(
        stac_items_gdf,
        pv_h3_groups,
        how="inner",
        predicate="intersects"
    )
    
    return joined

def create_h3_stac_fetch_plan(
    pv_gdf: gpd.GeoDataFrame,
    h3_resolution: int = 7,
    sensor_gsd: float = 10,  # GSD in meters
    max_items_per_query: int = 100,
    area_threshold: float = 10000,  # m²
) -> Dict:
    """
    Create an optimized plan for fetching STAC items.
    Uses different H3 resolutions depending on installation size.
    
    Args:
        pv_gdf: GeoDataFrame with PV installations
        h3_resolution: Base H3 resolution
        max_items_per_query: Maximum number of PV items to include in a single query
        area_threshold: Area threshold to use different resolutions
        
    Returns:
        Dict with H3 cells and their contained PV installations
    """
    # Make a copy to avoid modifying original
    gdf = pv_gdf.copy()
    
    # Assign H3 indices based on size
    large_pv = gdf[gdf['area_m2'] >= area_threshold].copy()
    small_pv = gdf[gdf['area_m2'] < area_threshold].copy()
    
    # Use different resolutions - higher resolution for smaller installations
    if not large_pv.empty:
        large_pv = add_h3_index_to_pv_labels(
            large_pv, h3_resolution=h3_resolution-1, method="representative"
        )
    
    if not small_pv.empty:
        small_pv = add_h3_index_to_pv_labels(
            small_pv, h3_resolution=h3_resolution+1, method="centroid"
        )
    
    # Combine results
    result = pd.concat([large_pv, small_pv])
    
    # Group by H3 cells
    grouped = group_pv_by_h3_cells(result)
    
    # Create fetch plan
    fetch_plan = {
        "h3_cells": grouped['h3_index'].tolist(),
        "pv_counts": grouped['area_m2_count'].tolist(),
        "total_pv_installations": len(gdf),
        "total_h3_cells": len(grouped),
        "h3_cell_details": grouped.to_dict(orient="records")
    }
    
    return fetch_plan

# Refactored functions for consolidating Overture geoparquets into a single DuckDB database
# and optionally skipping saving raw geoparquet files to save storage.
def fetch_overture_maps_theme(
    s3_path: str,
    db_file: str,  
    table_name: str,
    out_gpq_path: str = None,  
    s3_region: str = "us-west-2",
    geom_col: str = "geometry",
    subset_bbox: Optional[List[float]] = None,
    filter_dict: Optional[Dict[str, Tuple[str, Any]]] = None,
    save_raw: bool = False  
):
    """
    Fetches Overture division area geoparquet files from S3 and saves the data into a table
    called 'overture_divisions' in a consolidated DuckDB database. Optionally, a raw geoparquet
    export of the table can be saved.

    Args:
        s3_path: S3 path to Overture's division boundaries (parquet files)
        db_file: Path of the consolidated DuckDB database file.
        table_name: Name of the table to create in the DuckDB database.
        out_gpq_path: Optional output path for the raw geoparquet export.
        s3_region: AWS S3 region.
        geom_col: Name of the geometry column.
        subset_bbox: Optional bounding box filter [min_lon, min_lat, max_lon, max_lat]
        division_filter: Optional filter for division type.
        save_raw: If True and out_gpq_path is provided, also save the table as a raw geoparquet file.
    """
    print(f"Fetching Overture division areas from S3:\n {s3_path}")
    print(f"Using DuckDB database file:\n `{db_file}` and saving to table:\n `{table_name}`")

    conn = None
    try:
        # Connect to the consolidated DuckDB database
        conn = duckdb.connect(database=db_file, read_only=False)
        print("  Loading DuckDB extensions...")
        conn.install_extension("httpfs")
        conn.load_extension("httpfs")
        conn.install_extension("spatial")
        conn.load_extension("spatial")
        
        # Configure S3 access (anonymous for public Overture data)
        conn.execute("SET s3_use_ssl=true;")
        conn.execute(f"SET s3_region='{s3_region}';")
        
        
        where_clauses = []
        if subset_bbox:
            if len(subset_bbox) == 4:
                min_lon, min_lat, max_lon, max_lat = subset_bbox
                where_clauses.append(
                    f"ST_Intersects(ST_GeomFromWKB({geom_col}), ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326))"
                )
            else:
                print("Warning: Bounding box should have 4 coordinates [min_lon, min_lat, max_lon, max_lat]. Ignoring bbox filter.")
        if filter_dict:
            # print(f"  Applying filters: {filter_dict}")
            # key is column with tuple of (operator, rvalue)
            # filter_clauses = []
            for col_name, cond in filter_dict.items():
                operator, rvalue = cond
                # if list convert to SQL syntax
                if isinstance(rvalue, list):
                    # need to add quotes to string values
                    rvalue = ", ".join([f"'{v}'" for v in rvalue])
                    rvalue = f"({rvalue})"
                where_clauses.append(f"{col_name} {operator.upper()} {rvalue}")
                
            
        if len(where_clauses) > 0:
            # TODO: verify for multiple filters
            where_clause = where_clauses[0] if len(where_clauses) == 1 else '\nAND '.join(where_clauses) 
            print(f"Applying parsed where statement: {where_clause}")
        else:
            where_clauses = "" # convert to string so it can be used in the SQL query without error when empty

        if isinstance(where_clauses, list):
            where_clauses = where_clauses[0]
        # Build the SQL query with optional filters
        sql = f"""CREATE TABLE IF NOT EXISTS {table_name} AS 
            SELECT *
            FROM read_parquet('{s3_path}', hive_partitioning=1)
            WHERE {where_clauses}"""
        # print(f"  SQL query: {sql}")
        
        # Load data into raw DuckDB table "overture_divisions"
        print(f" Inserting data into DuckDB table '{table_name}'...")
        conn.execute(sql)

        # Determine and report the count of matching records
        file_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if filter_dict and "subtype" in filter_dict:
            subtype_clauses = [c for c in where_clauses.split(" AND ") if "subtype" in c]
            print(f"  Filtered {file_count} records from {s3_path} using filters: {' AND '.join(subtype_clauses)}")
        print(f"Saved {file_count} division boundary records to database ")

        # create a spatial index on the geometry column
        conn.execute(f"CREATE INDEX {table_name}_idx ON {table_name} USING RTREE({geom_col});")
        
        # Optionally export table to a raw geoparquet file (to out_gpq_path) if save_raw=True
        if save_raw and out_gpq_path:
            print(f"  Saving raw geoparquet export to {out_gpq_path}")
            conn.execute(f"""
            COPY (SELECT * FROM {table_name})
            TO '{out_gpq_path}' (FORMAT 'PARQUET', COMPRESSION 'zstd');
            """)
        
        print("Overture divisions have been saved into the DuckDB database")
        
    except Exception as e:
        print(f"Error fetching Overture division areas: {e}")
        return False
    finally:
        if conn:
            conn.close()

    return True

def spatial_join_pv_overture_duckdb(
    pv_gpq_path: str, # expects existing h3 index column at sh
    db_file: str,  # Path to the DuckDB database file with overture divisions table
    out_gpq_path: str,
    filter_dict: Optional[Dict[str, Tuple[str, Any]]] = None,
    geom_col: str = "geometry",
    join_predicate: str = "ST_Within",  # e.g., 'ST_Intersects', 'ST_Within'
    use_centroid: bool = False
):
    """
    Performs a spatial join between a local PV GeoParquet file and Overture division areas stored in a
    consolidated DuckDB database table 'raw_overture_divisions'. For each PV polygon, we find the intersecting 
    administrative division area using duckdb's grouping functions to obtain the division intersects at each admin_level
    
    Args:
        pv_gpq_path: Path to the input GeoParquet file containing PV polygons.
        db_file: Path to the consolidated DuckDB database (with 'raw_overture_divisions' table).
        out_gpq_path: Path to save the resulting joined GeoParquet file.
        overture_cols_to_keep: List of columns to keep from the Overture data.
        geom_col: Name of the geometry column in both datasets.
        join_predicate: DuckDB spatial function for the join (e.g., 'ST_Intersects', 'ST_Within').
        use_centroid: If True, use centroids of PV geometries for the join (faster for complex polygons).
    """
    print(f"Starting DuckDB spatial join...")
    print(f"  PV Data: {pv_gpq_path}")
    print(f"  Using consolidated DB: {db_file}")
    print(f"  Output: {out_gpq_path}")
    print(f"  Using {join_predicate} with {'centroids' if use_centroid else 'full geometries'}")
    
    conn = None
    try:
        # Connect to the consolidated DuckDB database
        conn = duckdb.connect(database=db_file, read_only=False)
        print("  Loading DuckDB spatial extension...")
        conn.install("spatial")
        conn.load("spatial")
        
        # Format the list of columns to keep from Overture
        ov_cols_str = ", ".join([f"ov.{col} AS ov_{col}" for col in overture_cols_to_keep])
        
        # Create the geometry expression for PV data
        pv_geom_expr = f"ST_Centroid(pv.{geom_col})" if use_centroid else f"pv.{geom_col}"
        
        # Build the spatial join query using the table 'overture_divisions'
        join_sql = f"""
        CREATE TABLE result AS
        SELECT
            pv.*,
            {ov_cols_str}
        FROM
            read_parquet('{pv_gpq_path}') AS pv
        LEFT JOIN
            overture_divisions AS ov
        ON
            {join_predicate}({pv_geom_expr}, ov.{geom_col})
        """
        
        print("  Executing spatial join (this may take a while)...")
        start_time = time.time()
        conn.execute(join_sql)
        
        # Count rows in result
        row_count = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
        print(f"  Join completed with {row_count} records")
        
        # Save to GeoParquet, preserving the geometry column
        print(f"  Saving joined result to {out_gpq_path}")
        conn.execute(f"""
        COPY (
            SELECT * FROM result
        ) TO '{out_gpq_path}' (FORMAT 'PARQUET', COMPRESSION zstd);
        """)
        
        elapsed_time = time.time() - start_time
        print(f"  Spatial join completed in {elapsed_time:.2f} seconds")
        
        # (Optional) Show sample of result
        print("\nSample of joined data:")
        sample = conn.execute("SELECT * FROM result LIMIT 5").fetchdf()
        print(sample)
        
        return True
        
    except Exception as e:
        print(f"Error in spatial join: {e}")
        return False
    finally:
        if conn:
            conn.close()