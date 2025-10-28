"""
Utilities for fetching and preprocessing PV installation datasets.
This module handles downloading, extraction, and standardization of
various PV installation datasets from different repositories.
"""

import os
import json
import time
import re
import urllib.parse
import tempfile
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile
import random

import numpy as np
import cv2
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
import requests
from tqdm import tqdm
from seedir import seedir
from dotenv import load_dotenv

import datahugger
import duckdb


load_dotenv()
# Default threshold for considering geometries as duplicates (75%)
OVERLAP_THRESH = float(os.getenv('GEOM_OVERLAP_THRESHOLD', 0.75))
ZSTD_COMPRESSION = int(os.getenv('ZSTD_PARQUET_COMPRESSION_LEVEL', 10))
DB_PATH = os.getenv('DUCKDB_DIR', os.path.join(os.getcwd(), 'datasets', 'db'))

# ================= DATASET PREP FUNCTIONS =================

def yolo_labels_to_binary_mask(label_file_path, image_height, image_width, output_mask_path):
    """
    Converts a YOLO bounding box annotation file to a single binary segmentation mask image.
    All valid bounding boxes in the label file will be drawn as white rectangles
    on a black background.

    Args:
        label_file_path (str or Path): Path to the YOLO .txt label file.
        image_height (int): Height of the corresponding image chip.
        image_width (int): Width of the corresponding image chip.
        output_mask_path (str or Path): Path to save the generated binary mask image (e.g., .png).
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    found_boxes = False
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        # YOLO format: class_id, x_center_norm, y_center_norm, width_norm, height_norm
                        # We assume all class_ids in the file are the objects of interest (e.g., solar panels)
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])

                        # Denormalize coordinates
                        x_center_abs = x_center_norm * image_width
                        y_center_abs = y_center_norm * image_height
                        box_width_abs = width_norm * image_width
                        box_height_abs = height_norm * image_height

                        # Calculate top-left (x1, y1) and bottom-right (x2, y2) pixel coordinates
                        x1 = int(round(x_center_abs - (box_width_abs / 2)))
                        y1 = int(round(y_center_abs - (box_height_abs / 2)))
                        x2 = int(round(x_center_abs + (box_width_abs / 2)))
                        y2 = int(round(y_center_abs + (box_height_abs / 2)))

                        # Clip coordinates to be within image bounds
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(image_width, x2); y2 = min(image_height, y2) # cv2.rectangle is exclusive for x2,y2

                        # Draw a filled white rectangle on the mask if the box is valid
                        if x1 < x2 and y1 < y2:
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1) # 255 for white, -1 for filled
                            found_boxes = True
                    except ValueError:
                        print(f"Warning: Skipping malformed line in {label_file_path}: {line.strip()}")
                        continue

        if not found_boxes and Path(label_file_path).stat().st_size > 0 : # Only warn if file was not empty
            print(f"Warning: No valid bounding boxes found or parsed in {label_file_path}. Mask will be empty.")

        # Save the binary mask image
        cv2.imwrite(str(output_mask_path), mask)
        return True # Indicate success

    except FileNotFoundError:
        print(f"Error: Label file not found: {label_file_path}")
        return False
    except Exception as e:
        print(f"An error occurred processing {label_file_path}: {e}")
        return False

def process_yolo_label_directory(yolo_label_dir, image_chips_dir, output_mask_dir,
                                 chip_dims_info, default_chip_type='native'):
    """
    Processes all .txt label files in a directory, determines chip dimensions,
    and saves corresponding binary masks.

    Args:
        yolo_label_dir (str or Path): Directory containing YOLO .txt label files.
        image_chips_dir (str or Path): Directory containing corresponding image chips.
                                      Used to infer dimensions if not explicitly known or if needed.
        output_mask_dir (str or Path): Directory to save the generated mask images.
        chip_dims_info (dict or tuple):
            If dict: Maps chip type (e.g., 'native', 'hd') to (height, width) tuples.
                     Filename must contain chip type.
            If tuple: (height, width) to be used for all chips.
        default_chip_type (str): If chip_dims_info is a dict and type cannot be inferred
                                 from filename, use these dimensions.
    """
    yolo_label_dir = Path(yolo_label_dir)
    image_chips_dir = Path(image_chips_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    error_count = 0

    label_files = list(yolo_label_dir.glob("*.txt"))
    if not label_files:
        print(f"No .txt label files found in {yolo_label_dir}")
        return

    for label_file in tqdm(label_files, desc="Processing YOLO labels", unit="file"):
        img_h, img_w = None, None

        if isinstance(chip_dims_info, tuple):
            img_h, img_w = chip_dims_info
        elif isinstance(chip_dims_info, dict):
            fn_lower = label_file.name.lower() # Make case insensitive
            if 'native' in fn_lower:
                img_h, img_w = chip_dims_info.get('native')
            elif 'hd' in fn_lower:
                img_h, img_w = chip_dims_info.get('hd')
            
            if not (img_h and img_w): # If type not in filename or key not in dict
                 img_h, img_w = chip_dims_info.get(default_chip_type)


        if not (img_h and img_w): # Try to infer from image file if not determined
            # This assumes image files have the same stem and are in image_chips_dir
            potential_img_paths = [image_chips_dir / (label_file.stem + ext) for ext in ['.tif', '.png', '.jpg', '.jpeg']]
            actual_img_path = next((p for p in potential_img_paths if p.exists() and p.is_file()), None)
            if actual_img_path:
                try:
                    # Using PIL to get dimensions as it's already imported and handles various formats
                    with Image.open(actual_img_path) as temp_img_pil:
                        img_w, img_h = temp_img_pil.size # PIL size is (width, height)
                    # print(f"Inferred dimensions ({img_h}x{img_w}) for {label_file.name} from image file: {actual_img_path}")
                except Exception as e_img:
                    print(f"Warning: Error reading image {actual_img_path} for dimensions: {e_img}. Skipping {label_file.name}.")
                    error_count += 1
                    continue
            else:
                print(f"Warning: Could not determine/infer chip dimensions for {label_file.name}. Skipping.")
                error_count += 1
                continue
        
        output_mask_file = output_mask_dir / f"{label_file.stem}.png" # Save masks as PNG
        yolo_labels_to_binary_mask(label_file, img_h, img_w, output_mask_file)
        processed_count += 1

    print(f"\nFinished processing directory: {yolo_label_dir}")
    print(f"Successfully generated {processed_count} masks in {output_mask_dir}.")
    if error_count > 0:
        print(f"Encountered errors or skips for {error_count} files.")

# ================= REPOSITORY FETCH FUNCTIONS =================

def fetch_from_datahugger(doi, dst, max_mb=100, force=False):
    """
    Fetch dataset files using datahugger for repositories like Zenodo and Figshare.
    
    Args:
        doi (str): DOI or URL of the dataset
        dst (str): Path to save the dataset
        max_mb (int): Maximum file size in MB
        force (bool): Force download even if files exist
        
    Returns:
        dict: A dictionary with dataset information
    """
    max_dl = max_mb * 1024 * 1024
    dst_p = Path(dst)
    geofile_regex = r'^(.*\.(geojson|json|shp|zip|csv|gpkg))$'
    
    if force and dst_p.exists():
        shutil.rmtree(dst_p)
        print(f"Removed existing dataset directory: {os.path.relpath(dst_p)}")
    
    # Handle unzipping manually to get progress bar for download
    ds_tree = datahugger.get(doi, dst, max_file_size=max_dl, force_download=force, unzip=False)
    
    # Compare files to be fetched with existing files  
    files_to_fetch = [f['name'] for f in ds_tree.dataset.files if f['size'] <= max_dl]
    ds_files = [os.path.join(root, fname) for root, dirs, files in os.walk(dst_p) for fname in files if re.match(geofile_regex, fname)]
    
    # Handle zip files if present
    if any(f.endswith('.zip') for f in ds_files):
        print(f"Dataset contains zip files that need extraction.")
        # Check if the zip file was fetched and directly extract if it's the only file in the dataset
        extracted_files = []
        if len(ds_files) <= 2 and any(f.endswith('.zip') for f in ds_files):
            zip_file = next((dst_p / f for f in ds_files if f.endswith('.zip')), None)
            if zip_file:
                print(f"Found zip file: {zip_file}")
                # Extract the zip file
                with ZipFile(zip_file, 'r') as zip_ref:
                    extracted_files = zip_ref.namelist()
                    zip_ref.extractall(dst)
                
                # Check if zip file consisted of a single dir and move contents up one level
                top_level_dir = None
                if extracted_files:
                    top_level_dir = dst_p / extracted_files[0]
                    if top_level_dir.is_dir():
                        # Move only first level dirs and files to our dataset dir
                        for item in top_level_dir.iterdir():
                            if item.name.endswith('.zip'):
                                continue
                            # Don't copy if already exists
                            elif (dst_p / item.name).exists():
                                print(f"Skipping {item} as it already exists in {os.path.relpath(dst)}")
                                continue
                            elif item.parent == top_level_dir and re.match(geofile_regex, item.name):
                                print(f"Moving {item} to {os.path.relpath(dst)}")
                                shutil.move(str(item), dst)
                        # Remove the top level dir
                        shutil.rmtree(top_level_dir)
    
    # Get updated file list 
    ds_files = [os.path.join(root, fname) for root, dirs, files in os.walk(dst_p) for fname in files if re.match(geofile_regex, fname)]
    
    return {
        'output_folder': ds_tree.output_folder,
        'files': ds_files,
        'fs_tree': seedir(dst_p, depthlimit=5, printout=False, regex=True, include_files=geofile_regex)
    }

def fetch_from_github(doi, dst, max_mb=100, force=False):
    """
    Fetch dataset files from GitHub repositories.
    
    Args:
        doi (str): GitHub URL of the dataset
        dst (str): Path to save the dataset
        max_mb (int): Maximum file size in MB
        force (bool): Force download even if files exist
        
    Returns:
        dict: A dictionary with dataset information
    """
    max_dl = max_mb * 1024 * 1024
    dst_p = Path(dst)
    geofile_regex = r'^(.*\.(geojson|json|shp|zip|csv|gpkg))$'
    
    # Create destination directory if it doesn't exist
    os.makedirs(dst, exist_ok=True)
    
    # Check if local path exists and contains expected files
    if os.path.exists(dst) and any(os.path.splitext(fname)[1] in ['.geojson', '.json', '.shp', '.zip'] 
                                  for fname in os.listdir(dst)) and not force:
        print(f"Destination path already exists and contains expected files.")
        print(f"\033[1mSkipping Download!\033[0m")
        # Get dataset dir info
        ds_files = [os.path.join(root, fname) for root, dirs, files in os.walk(dst_p) 
                   for fname in files if re.match(geofile_regex, fname)]
        return {
            'output_folder': dst,
            'files': ds_files,
            'fs_tree': seedir(dst_p, depthlimit=5, printout=False, regex=True, include_files=geofile_regex)
        }
    
    # Parse the GitHub URL
    parts = doi.replace('https://github.com/', '').split('/')
    repo_path = f"{parts[0]}/{parts[1]}"
    
    # Extract branch and path
    branch = 'main'  # Default branch
    path = ''
    
    # Check if it's a folder/repository or a single file
    if '/blob/' not in doi and 'raw.githubusercontent.com' not in doi:
        try:
            if 'tree' in parts:
                tree_index = parts.index('tree')
                branch = parts[tree_index + 1]
                path = '/'.join(parts[tree_index + 2:]) if len(parts) > tree_index + 2 else ''
            
            # Create a temporary directory for the sparse checkout
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize git repository with sparse checkout
                cmd = f"git clone --filter=blob:limit={max_mb}m --depth 1 https://github.com/{repo_path}.git {os.path.basename(dst)}"
                
                process = subprocess.run(cmd, shell=True, cwd=temp_dir, 
                                        capture_output=True, text=True)
                if process.returncode != 0:
                    raise Exception(f"Git command failed: {cmd}\n{process.stderr}")
                
                # Copy only the files in the dir specified in DOI/URL
                repo_ds_dir = os.path.join(temp_dir, os.path.basename(dst), path) if path else os.path.join(temp_dir, os.path.basename(dst))
                files_list = []
                
                for root, _, files in os.walk(repo_ds_dir):
                    for file in files:
                        if file.startswith('.git'):
                            continue
                        src_file = os.path.join(root, file)
                        # Create relative path
                        rel_path = os.path.relpath(src_file, repo_ds_dir)
                        dst_file = os.path.join(dst, rel_path)
                        
                        # Create destination directory if needed
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        
                        # Copy the file
                        shutil.copy2(src_file, dst_file)
                        files_list.append(dst_file)
                        print(f"Copied {rel_path} to {os.path.relpath(dst_file)}")
            
            ds_files = [f for f in files_list if re.match(geofile_regex, f)]
            return {
                'output_folder': dst,
                'files': ds_files,
                'fs_tree': seedir(dst_p, depthlimit=5, printout=False, regex=True, include_files=geofile_regex)
            }
                
        except Exception as e:
            print(f"Error performing git clone: {e}")
            return None
    else:
        # It's a single file (raw URL or blob URL)
        try:
            # Convert blob URL to raw URL if needed
            if '/blob/' in doi:
                raw_url = doi.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            else:
                raw_url = doi
            
            # Extract filename from URL
            filename = os.path.basename(urllib.parse.urlparse(raw_url).path)
            local_file_path = os.path.join(dst, filename)
            
            # Download the file
            response = requests.get(raw_url, stream=True)
            response.raise_for_status()
            
            # Check file size
            file_size = int(response.headers.get('content-length', 0))
            if file_size > max_dl:
                print(f"File size ({file_size} bytes) exceeds maximum allowed size ({max_dl} bytes)")
                return None
            
            with open(local_file_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {filename}", unit='KB'):
                    f.write(chunk)
            print(f"Downloaded {filename} to {os.path.relpath(local_file_path)}")
            
            return {
                'output_folder': dst,
                'files': [local_file_path],
                'fs_tree': seedir(dst_p, depthlimit=5, printout=False, regex=True, include_files=geofile_regex)
            }
            
        except Exception as e:
            print(f"Error downloading GitHub file: {e}")
            return None

def fetch_from_sciencebase(doi, dst, max_mb=100, force=False):
    """
    Fetch dataset files from ScienceBase repositories.
    
    Args:
        doi (str): ScienceBase DOI or ID
        dst (str): Path to save the dataset
        max_mb (int): Maximum file size in MB
        force (bool): Force download even if files exist
        
    Returns:
        dict: A dictionary with dataset information or None if not implemented
    """
    print("ScienceBase fetching not implemented yet")
    return None

def fetch_dataset_files(dataset_name, metadata, max_mb=100, force=False):
    """
    Master function to fetch dataset files based on repository type.
    
    Args:
        dataset_name (str): Name of the dataset
        metadata (dict): Metadata dictionary for the dataset
        max_mb (int): Maximum file size in MB
        force (bool): Force download even if files exist
        
    Returns:
        dict: A dictionary with dataset information
    """
    doi = metadata['doi']
    repo = metadata['repo']
    compression = metadata['compression']
    label_fmt = metadata['label_fmt']
    
    # Convert to bytes
    max_dl = max_mb * 1024 * 1024
    dataset_dir = os.path.join(os.getenv('DATA_PATH', '.'), 'raw', 'labels', dataset_name)
    dst = os.path.join(os.getcwd(), dataset_dir)
    dst_p = Path(dst)
    
    dataset_tree = {}
    
    # Use appropriate fetch function based on repository
    if repo in ['figshare', 'zenodo']:
        dataset_tree = fetch_from_datahugger(doi, dst, max_mb, force)
    elif repo == 'github':
        dataset_tree = fetch_from_github(doi, dst, max_mb, force)
    elif repo == 'sciencebase':
        dataset_tree = fetch_from_sciencebase(doi, dst, max_mb, force)
    else:
        print(f"Unsupported repository type: {repo}")
        return None
    
    if dataset_tree:
        print(f"Fetched {len(dataset_tree['files'])} dataset files for {dataset_name} in {os.path.relpath(dataset_tree['output_folder'])}:")
        print(dataset_tree['fs_tree'])

    
    return dataset_tree

# ================= DATASET-SPECIFIC PREPROCESSING FUNCTIONS =================

def france_eur_pv_preprocess(ds_metadata, ds_subdir, metadata_dir='raw', crs=None, geom_type='Polygon'):
    """
    Preprocess France/Western Europe PV installations dataset.
    
    Args:
        ds_metadata (dict): Dataset metadata
        ds_subdir (str): Dataset subdirectory
        metadata_dir (str): Metadata directory
        crs (str): Coordinate reference system
        geom_type (str): Geometry type (Polygon or Point)
        
    Returns:
        tuple: (GeoDataFrame, DataFrame) with processed data
    """
    ds_dir = Path(ds_metadata['output_dir'])
    data_dir = ds_dir / ds_subdir
    metadata_file = 'raw-metadata_df.csv' if metadata_dir == 'raw' else 'metadata_df.csv'
    metadata_file = ds_dir / metadata_dir / metadata_file
    coords_file = "polygon-analysis.json" if geom_type == 'Polygon' else "point-analysis.json"
    
    # Keep files in the specified subdir with the right filename
    geom_files = [fpath for fpath in ds_metadata['files'] if str(fpath).startswith(str(data_dir)) and str(fpath).endswith(coords_file)]
    crs = crs or 'EPSG:4326'  # Default to WGS84

    # Load the metadata file
    metadata_df = pd.read_csv(metadata_file)
    print(f"Loaded '{metadata_file.split('/')[-1]}' with {len(metadata_df)} rows")

    # Load into geopandas, inspect the data, and add metadata_df to separate pd dataframe
    raw_features = []
    for geom_file_path in geom_files:
        campaign_name = Path(geom_file_path).parent.name
        print(f"Processing {campaign_name} campaign...")
        
        with open(geom_file_path, 'r') as f:
            geom_data = json.load(f)
        feat_types = set([f['type'] for f in geom_data])
        print(f"Feature types: {feat_types}")
    
        for idx, feature_dict in enumerate(geom_data):
            # Skip empty dictionaries
            if not feature_dict:
                continue
            
            try:
                feature_id = feature_dict.get('id', idx)  # Use index if ID is not present

                # Extract geometry and coordinates
                if geom_type == 'Polygon':
                    # feat_dict = [{'polygons': [{'points': {'x': <px_coord>, 'y': <px_coord>}, ...}]}, ...]
                    coords = feature_dict['polygons']
                    if isinstance(coords, list) and len(coords) > 0:
                        # Handle multiple polygons
                        polygons = []
                        for poly_coords in coords:
                            if len(poly_coords) >= 3:  # Need at least 3 points for a polygon
                                polygons.append(Polygon(poly_coords))
                        
                        if len(polygons) == 1:
                            geometry = polygons[0]
                        else:
                            geometry = MultiPolygon(polygons)
                            
                        # Create feature dictionary with properties
                        feature = {
                            'id': feature_id,
                            'campaign': campaign_name,
                            'geometry': geometry
                        }
                        raw_features.append(feature)
                elif geom_type == 'Point':
                    # feat_dict = [{'clicks': [{'@type': 'Point', 'x': <px_coord>, 'y': <px_coord>}, ...]}, ...]
                    coords = feature_dict['clicks']
                    if isinstance(coords, list) and len(coords) > 0:
                        points = []
                        for point_coords in coords:
                            if 'x' not in point_coords or 'y' not in point_coords:
                                continue
                            else:
                                points.append(Point(point_coords['x'], point_coords['y']))
                        raw_features.extend(points)
            except Exception as e:
                print(f"Error processing feature {feature_dict}: {e}")
                continue

    if raw_features:
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(raw_features, crs=crs)
        
        # Add metadata to the gdf
        if 'id' in gdf.columns:
            gdf['id'] = gdf['id'].astype(str)
            
        # Ensure CRS is set
        if gdf.crs is None:
            gdf.set_crs(crs, inplace=True)
        elif str(gdf.crs) != crs:
            gdf = gdf.to_crs(crs)
    
    return gdf, metadata_df

def global_pv_inventory_spot_processing(gdf, dataset_name, output_dir, subset_bbox=None, geom_type='Polygon', rm_invalid=True):
    """
    Process global PV inventory Sentinel-2/SPOT dataset.
    
    Args:
        gdf (GeoDataFrame): Input geodataframe
        dataset_name (str): Dataset name
        output_dir (str): Output directory
        subset_bbox (tuple): Optional bounding box
        geom_type (str): Geometry type
        rm_invalid (bool): Remove invalid geometries
        
    Returns:
        GeoDataFrame: Processed geodataframe
    """
    all_cols = [
        'unique_id', 'area', 'confidence', 'install_date', 'iso-3166-1', 'iso-3166-2', 'gti', 'pvout', 
        'capacity_mw', 'match_id', 'wdpa_10km', 'LC_CLC300_1992', 'LC_CLC300_1993',
        'LC_CLC300_1994', 'LC_CLC300_1995', 'LC_CLC300_1996', 'LC_CLC300_1997', 'LC_CLC300_1998', 'LC_CLC300_1999', 
        'LC_CLC300_2000', 'LC_CLC300_2001', 'LC_CLC300_2002', 'LC_CLC300_2003', 'LC_CLC300_2004', 'LC_CLC300_2005', 
        'LC_CLC300_2006', 'LC_CLC300_2007', 'LC_CLC300_2008', 'LC_CLC300_2009', 'LC_CLC300_2010', 'LC_CLC300_2011',
        'LC_CLC300_2012', 'LC_CLC300_2013', 'LC_CLC300_2014', 'LC_CLC300_2015', 'LC_CLC300_2016', 'LC_CLC300_2017', 
        'LC_CLC300_2018', 'mean_ai', 'GCR', 'eff', 'ILR', 'area_error', 'lc_mode', 'lc_arid', 'lc_vis', 'geometry', 
        'aoi_idx', 'aoi', 'id', 'Country', 'Province', 'WRI_ref', 'Polygon Source', 'Date', 'building',
        'operator', 'generator_source', 'amenity', 'landuse', 'power_source', 'shop', 'sport', 'tourism', 'way_area', 
        'access', 'denomination', 'historic', 'leisure', 'man_made', 'natural', 'ref', 'religion', 'surface', 
        'z_order', 'layer', 'name', 'barrier', 'addr_housenumber', 'office', 'power',  'military'
    ]
    
    # Remove unwanted columns and keep only essential ones
    keep_cols = ['unique_id', 'confidence', 'install_date', 'capacity_mw', 'iso-3166-2', 'pvout', 'geometry', 'area']
    print(f"Filtering from {len(all_cols)} columns  to {len(keep_cols)} columns:\n{keep_cols}")
    
    # Filter columns
    return gdf[keep_cols]

def global_pv_inventory_sent2_processing(gdf, dataset_name, output_dir, subset_bbox=None, geom_type='Polygon', rm_invalid=True):
    """Process global PV inventory Sentinel-2 dataset (placeholder)"""
    return gdf

def india_pv_solar_farms_processing(gdf, dataset_name, output_dir, subset_bbox=None, geom_type='Polygon', rm_invalid=True):
    """Process India PV solar farms dataset (placeholder)"""
    all_cols = ['State', 'Area', 'Latitude', 'Longitude', 'fid', 'geometry', 'dataset', 'area_m2', 'centroid_lon', 'centroid_lat']
    keep_cols = ['fid', 'State', 'Area', 'Latitude', 'Longitude', 'geometry']
    return gdf

def usa_cali_usgs_pv_processing(gdf, dataset_name, output_dir, subset_bbox=None, geom_type='Polygon', rm_invalid=True):
    """Process USA California USGS PV dataset (placeholder)"""
    all_cols = ['polygon_id', 'centroid_latitude', 'centroid_longitude', 'centroid_latitude_pixels', 'centroid_longitude_pixels', 'city', 'area_pixels', 'area_meters', 'image_name', 'nw_corner_of_image_latitude', 'nw_corner_of_image_longitude', 'se_corner_of_image_latitude', 'se_corner_of_image_longitude', 'datum', 'projection_zone', 'resolution', 'jaccard_index', 'polygon_vertices_pixels', 'geometry']
    keep_cols = ['polygon_id', 'datum', 'centroid_latitude', 'centroid_longitude', 'area_meters', 'image_name', 'geometry']
    return gdf[keep_cols]

def usa_eia_large_scale_pv_processing(gdf, dataset_name, output_dir, subset_bbox=None, geom_type='Polygon', rm_invalid=True):
    """Process USA EIA large-scale PV dataset (placeholder)"""
    return gdf

def global_harmonized_large_solar_farms_processing(gdf, dataset_name, output_dir, subset_bbox=None, geom_type='Polygon', rm_invalid=True):
    """
    Process global harmonized large solar farms dataset.
    
    Args:
        gdf (GeoDataFrame): Input geodataframe
        dataset_name (str): Dataset name
        output_dir (str): Output directory
        subset_bbox (tuple): Optional bounding box
        geom_type (str): Geometry type
        rm_invalid (bool): Remove invalid geometries
        
    Returns:
        GeoDataFrame: Processed geodataframe
    """
    all_cols = ['sol_id', 'GID_0', 'panels', 'panel.area', 'landscape.area', 'urban', 'power', 'geometry']
    # Remove unwanted columns
    keep_cols = ['sol_id', 'panels', 'panel.area', 'landscape.area', 'water', 'urban', 'power', 'geometry']
    print(f"Filtering from {len(all_cols)} columns to {len(keep_cols)} columns:\n{keep_cols}")
    
    # Filter columns
    return gdf[keep_cols]


# ================= GEOPANDAS HELPER FUNCTIONS =================

def filter_gdf_duplicates(gdf, geom_type='Polygon', overlap_thresh=OVERLAP_THRESH):
    """
    Remove duplicate geometries from a GeoDataFrame based on a specified overlap threshold,
    keeping the geometry with the smaller area when two overlap substantially.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
        geom_type (str): Geometry type to filter by (default: 'Polygon')
        overlap_thresh (float): Overlap threshold for removing duplicates (default: 0.8)
        
    Returns:
        GeoDataFrame: GeoDataFrame with duplicates removed
    """

    init_rows = len(gdf)
    print(f"Filtering {init_rows} geometries for duplicates and overlaps")
    # First identify exact duplicates
    gdf = gdf.drop_duplicates('geometry')
    
    # Identify geometries that overlap substantially
    overlaps = []
    # Use spatial index for efficiency
    spatial_index = gdf.sindex
    
    for idx, geom in enumerate(gdf.geometry):
        # Find potential overlaps using the spatial index
        # see here on other spatial index methods and supported spatial predicates: https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.query.html
        possible_matches = list(spatial_index.intersection(geom.bounds))
        # Remove self from matches
        if idx in possible_matches:
            possible_matches.remove(idx)
        
        for other_idx in possible_matches:
            other_geom = gdf.iloc[other_idx].geometry
            if geom.intersects(other_geom):
                # Calculate overlap percentage (relative to the smaller polygon)
                intersection_area = geom.intersection(other_geom).area
                min_area = min(geom.area, other_geom.area)
                overlap_percentage = intersection_area / min_area if min_area > 0 else 0.0
                
                # If overlap is significant (e.g., > threshold)
                if overlap_percentage > overlap_thresh:
                    # Keep the geometry with the smaller area (presumably more precise PV annotation)
                    if geom.area < other_geom.area:
                        overlaps.append(other_idx)
                        break
                    else:
                        overlaps.append(idx)
    
    # Remove overlapping geometries
    if overlaps:
        gdf = gdf.drop(gdf.index[overlaps]).reset_index(drop=True)
    print(f"Removed {init_rows - len(gdf)} geometries with >{overlap_thresh*100}% overlap")
    
    return gdf

# columns for datasts
# ind_pv_solar_farms_2022 = ['State', 'Area', 'Latitude', 'Longitude', 'fid', 'geometry', 'dataset', 'area_m2', 'centroid_lon', 'centroid_lat']
# global_harmonized_large_solar_farms_2020 =  ['sol_id', 'GID_0', 'panels', 'panel.area', 'landscape.area', 'water', 'urban', 'power', 'geometry']
# usa_cali_usgs_pv_2016 = ['polygon_id', 'centroid_latitude', 'centroid_longitude', 'centroid_latitude_pixels', 'centroid_longitude_pixels', 'city', 'area_pixels', 'area_meters', 'image_name', 'nw_corner_of_image_latitude', 'nw_corner_of_image_longitude', 'se_corner_of_image_latitude', 'se_corner_of_image_longitude', 'datum', 'projection_zone', 'resolution', 'jaccard_index', 'polygon_vertices_pixels', 'geometry']
# global_pv_inventory_sent2_spot_2021 = ['unique_id', 'area', 'confidence', 'install_date', 'iso-3166-1', 'iso-3166-2', 'gti', 'pvout', 'capacity_mw', 'match_id', 'wdpa_10km', 'LC_CLC300_1992', 'LC_CLC300_1993', 'LC_CLC300_1994', 'LC_CLC300_1995', 'LC_CLC300_1996', 'LC_CLC300_1997', 'LC_CLC300_1998', 'LC_CLC300_1999', 'LC_CLC300_2000', 'LC_CLC300_2001', 'LC_CLC300_2002', 'LC_CLC300_2003', 'LC_CLC300_2004', 'LC_CLC300_2005', 'LC_CLC300_2006', 'LC_CLC300_2007', 'LC_CLC300_2008', 'LC_CLC300_2009', 'LC_CLC300_2010', 'LC_CLC300_2011', 'LC_CLC300_2012', 'LC_CLC300_2013', 'LC_CLC300_2014', 'LC_CLC300_2015', 'LC_CLC300_2016', 'LC_CLC300_2017', 'LC_CLC300_2018', 'mean_ai', 'GCR', 'eff', 'ILR', 'area_error', 'lc_mode', 'lc_arid', 'lc_vis', 'geometry', 'key', 'resolution', 'pad', 'tilesize', 'zone', 'cs_code', 'ti', 'tj', 'proj4', 'wkt', 'ISO_A2', 'ISO_A3', 'idx', 'aoi_idx', 'aoi', 'id', 'Country', 'Province', 'Project', 'WRI_ref', 'Polygon Source', 'Date', 'building', 'operator', 'generator_source', 'amenity', 'landuse', 'power_source', 'shop', 'sport', 'tourism', 'way_area', 'access', 'construction', 'denomination', 'historic', 'leisure', 'man_made', 'natural', 'ref', 'religion', 'surface', 'z_order', 'layer', 'name', 'barrier', 'addr_housenumber', 'office', 'power', 'osm_id', 'military']

# could include power/capacity fields and leave as NULL for the others
# consolidated = [unify_ids(fid, sol_id/GID_0, polygon_id, unique_id), 'area_m2', 'centroid_lon', 'centroid_lat', 'dataset', 'geometry', 'bbox']

def process_vector_geoms(
    geom_files, 
    dataset_name, 
    output_dir=None, 
    subset_bbox=None, 
    geom_type='Polygon', 
    rm_invalid=True, 
    dedup_geoms=False,
    overlap_thresh=OVERLAP_THRESH, 
    out_fmt='geoparquet'):
    """
    Process vector geometry files and return a GeoDataFrame.
    
    Args:
        geom_files (list): List of paths to geometry files
        dataset_name (str): Name of the dataset
        output_dir (str): Output directory
        subset_bbox (tuple): Optional bounding box
        geom_type (str): Geometry type
        rm_invalid (bool): Remove invalid geometries
        dedup_geoms (bool): Deduplicate geometries
        overlap_thresh (float): Overlap threshold for deduplication
        out_fmt (str): Output format
        
    Returns:
        GeoDataFrame: Processed geodataframe
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    ds_dataframes = []

    for fname in geom_files:
        # Check if the file is a valid GeoJSON, shapefile, etc.
        if fname.endswith('.geojson') or fname.endswith('.json') or fname.endswith('.shp') or fname.endswith('.gpkg'):
            try:
                gdf = gpd.read_file(fname)
                ds_dataframes.append(gdf)
            except Exception as e:
                print(f"Error reading {os.path.relpath(fname)}: {e}")
                continue
    
    if len(ds_dataframes) == 0:
        print(f"No valid GeoJSON files found in {dataset_name}.")
        print(f"Skipping dataset {dataset_name}")
        return None
        
    # Concatenate all dataframes into a single GeoDataFrame
    gdf = gpd.GeoDataFrame(pd.concat(ds_dataframes, ignore_index=True))
    
    # Make sure the geometry column is included and named correctly
    if 'geometry' not in gdf.columns:
        gdf['geometry'] = gdf.geometry

    # TODO: filter out non Polygon/Multi-Polygon geometries

    # Basic info about the dataset
    print(f"Loaded geodataframe with raw counts of {len(gdf)} PV installations")
    print(f"Coordinate reference system: {gdf.crs}")
    print(f"Available columns: {gdf.columns.tolist()}")
    
    # Add dataset name as a new column
    gdf['dataset'] = dataset_name
    
    # Convert to WGS84 if not already in that CRS
    if gdf.crs is not None and gdf.crs.to_string() != 'EPSG:4326':
        # Convert to WGS84 in cases of other crs (eg NAD83 for Cali dataset)
        gdf = gdf.to_crs(epsg=4326)
        
    if subset_bbox is not None:
        # Filter the GeoDataFrame by the georeferenced bounding box
        gdf = gdf.cx[subset_bbox[0]:subset_bbox[2], subset_bbox[1]:subset_bbox[3]]
    
    # Quality checks and cleaning
    # Check for missing and invalid geometries
    invalid_geoms = gdf[gdf.geometry.is_empty | ~gdf.geometry.is_valid]
    if len(invalid_geoms) > 0 and rm_invalid:
        print(f"Warning: {len(invalid_geoms)} invalid or empty geometries found and will be removed.")
        # Optionally remove invalid geometries
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].reset_index(drop=True)
        
    # Eliminating duplicates and geometries that overlap too much
    if geom_type == 'Polygon' and dedup_geoms:
        gdf = filter_gdf_duplicates(gdf, geom_type=geom_type, overlap_thresh=overlap_thresh)

    # Perform any additional processing specific to the dataset for metadata and other attributes
    if dataset_name == 'global_pv_inventory_sent2_2024':
        print("Processing global_pv_inventory_sent2_2024 metadata")
        gdf = global_pv_inventory_sent2_processing(gdf, dataset_name, output_dir, subset_bbox=subset_bbox, geom_type=geom_type)
    elif dataset_name == 'global_pv_inventory_sent2_spot_2021':
        print("Processing global_pv_inventory_sent2_spot_2021 metadata")
        gdf = global_pv_inventory_spot_processing(gdf, dataset_name, output_dir, subset_bbox=subset_bbox, geom_type=geom_type)
    elif dataset_name == 'ind_pv_solar_farms_2022':
        print("Processing ind_pv_solar_farms_2022 metadata")
        gdf = india_pv_solar_farms_processing(gdf, dataset_name, output_dir, subset_bbox=subset_bbox, geom_type=geom_type)
    elif dataset_name == 'usa_cali_usgs_pv_2016':
        print("Processing usa_cali_usgs_pv_2016 metadata")
        gdf = usa_cali_usgs_pv_processing(gdf, dataset_name, output_dir, subset_bbox=subset_bbox, geom_type=geom_type)
    elif dataset_name == 'usa_eia_large_scale_pv_2023':
        print("Processing usa_eia_large_scale_pv_2023 metadata")
        gdf = usa_eia_large_scale_pv_processing(gdf, dataset_name, output_dir, subset_bbox=subset_bbox, geom_type=geom_type)
    elif dataset_name == 'global_harmonized_large_solar_farms_2020':
        print("Processing global_harmonized_large_solar_farms_2020 metadata")
        gdf = global_harmonized_large_solar_farms_processing(gdf, dataset_name, output_dir, subset_bbox=subset_bbox, geom_type=geom_type)
    
    # Re-project to a projected CRS (e.g. EPSG:3857) for accurate area (meters) calculations
    gdf_proj = gdf.geometry.to_crs(epsg=3857)
    gdf['area_m2'] = gdf_proj.area
    # Add centroid coordinates
    gdf['centroid_lon'] = gdf_proj.centroid.x
    # gdf['centroid_lon'] = gdf.geometry.centroid.x
    gdf['centroid_lat'] = gdf_proj.centroid.y
    # gdf['centroid_lat'] = gdf.geometry.centroid.y

    print(f"After filtering and cleaning, we have {len(gdf)} PV installations")

    if output_dir:
        out_path = os.path.join(output_dir, f"{dataset_name}_processed.{out_fmt}")

        if out_fmt == 'geoparquet':
            gdf.to_parquet(out_path, 
                index=None, 
                compression='snappy',
                geometry_encoding='WKB', 
                write_covering_bbox=True,
                schema_version='1.1.0')
        else:
            # geopackage, shapefile, or geojson
            fmt_driver_map = {'geojson': 'GeoJSON', 'shp': 'ESRI Shapefile', 'gpkg': 'GPKG'}
            gdf.to_file(out_path, driver=fmt_driver_map[out_fmt], index=None)
        print(f"Saved processed GeoDataFrame to {os.path.relpath(out_path)}")
    
    return gdf

def ddb_filter_duplicates(
    db_file: str,
    table_name: str,
    overlap_thresh: float = OVERLAP_THRESH,
    h3_res: int = 2,
    geom_column: str = "geometry",
    has_spatial_index: bool = True
):

    """
    Naive deduplication of geometries in a DuckDB table using h3 duckdb bindings and duckdb spatial functions.
    
    Args:
        db_conn (duckdb.DuckDBPyConnection): DuckDB connection object
        table_name (str): Name of the table to deduplicate
        geom_column (str): Name of the geometry column
        has_spatial_index (bool): If True, uses spatial index for deduplication

    Returns:
        bool: True if deduplication was successful, False otherwise.
    """

    db_conn = duckdb.connect(db_file)
    # use coarse resolution h3 grid to aggregate labels which we'll self join on to 
    db_conn.install_extension("spatial")
    db_conn.load_extension("spatial")
    db_conn.install_extension(extension="h3", repository="community")
    db_conn.load_extension("h3")
    add_h3_sql = f"""
    SELECT 
        unified_id,
        area_m2,
        spheroid_area,
        centroid_lon,
        centroid_lat,
        dataset,
        bbox,
        {geom_column},
        h3_latlng_to_cell(centroid_lat, centroid_lon, {h3_res}) AS h3_cell
    FROM {table_name}
    """
    # Create a temporary table with the h3 cell so we can perform spatial joins
    db_conn.execute(f"""CREATE TEMP TABLE tmp AS {add_h3_sql};""")
    # Create a spatial index on the temporary table
    if has_spatial_index:
        db_conn.execute(f"CREATE INDEX tmp_pv_idx ON tmp USING RTREE ({geom_column});")
    create_table_sql = f"""CREATE OR REPLACE TABLE {table_name}_dedup (
            unified_id VARCHAR PRIMARY KEY,
            area_m2 DOUBLE,
            spheroid_area DOUBLE,
            centroid_lon DOUBLE,
            centroid_lat DOUBLE,
            dataset VARCHAR,
            bbox STRUCT(xmin DOUBLE, ymin DOUBLE, xmax DOUBLE, ymax DOUBLE),
            geometry GEOMETRY
        );"""
    # create our table and add column constraints 
    db_conn.execute(create_table_sql)
    

    # Deduplicate using spatial functions
    dedup_query = f"""
    -- First identify all pairs of geometries that potentially overlap based on their H3 cells
    WITH potential_pairs AS (
        SELECT 
            a.unified_id AS unified_id_a,
            b.unified_id AS unified_id_b,
            a.area_m2 AS area_m2_a,
            b.area_m2 AS area_m2_b,
            a.geometry AS geom_a,
            b.geometry AS geom_b,
            ST_Equals(a.geometry, b.geometry) AS has_equiv_geom,
            ST_Overlaps(geom_a, geom_b) AS has_overlap_geom
        FROM tmp a
        JOIN tmp b USING (h3_cell)
        WHERE a.unified_id != b.unified_id
    ),
    -- Calculate overlap metrics for the potential pairs
    spatial_dedup AS (
        SELECT 
            unified_id_a,
            unified_id_b,
            area_m2_a,
            area_m2_b,
            -- Calculate overlap area for significant overlaps
            CASE 
                WHEN has_equiv_geom or has_overlap_geom
                THEN ST_Area(ST_Intersection(geom_a, geom_b)) / LEAST(ST_Area(geom_a), ST_Area(geom_b))
                ELSE 0 
            END AS overlap_ratio
        FROM potential_pairs
    ),
    -- Identify which geometries should be removed (larger one in overlapping pairs)
    geometries_to_remove AS (
        SELECT 
            CASE 
                WHEN area_m2_a > area_m2_b THEN unified_id_a
                ELSE unified_id_b
            END AS unified_id
        FROM spatial_dedup
        WHERE overlap_ratio > {overlap_thresh}
    )
    -- Create the final table with duplicates removed, using a suffix to avoid overwriting
    SELECT t.* EXCLUDE (h3_cell)
    FROM tmp t
    LEFT JOIN geometries_to_remove r
    ON t.unified_id = r.unified_id
    WHERE r.unified_id IS NULL;
    """
    # print duckdb performance profiling for query
    print(db_conn.execute(f"EXPLAIN ANALYZE {dedup_query};").fetchall())
    # insert the results of our deduplication into the table created above
    db_conn.execute(f"INSERT INTO {table_name}_dedup {dedup_query};")

    # Get counts before and after deduplication
    count_raw = db_conn.execute("SELECT COUNT(*) FROM tmp").fetchone()[0]
    db_conn.execute(dedup_query)
    count_final = db_conn.execute(f"SELECT COUNT(*) FROM {table_name}_dedup").fetchone()[0]
    print(f"Deduplication complete. Removed {count_raw - count_final} overlapping geometries ({(count_raw - count_final)/count_raw*100:.1f}%).")
    db_conn.close()

    if count_final > 0:
        print(f"Deduplicated table '{table_name}_dedup' created with {count_final} records.")
        return True
    elif count_final == count_raw:
        print(f"Deduplication complete. No geometries removed. {count_final} records remain.")
        return True
    else:
        print(f"Deduplication failed. No records remain in '{table_name}_dedup'.")
        return False

def geom_db_consolidate_dataset(
    parquet_files,
    table_name="global_consolidated_pv",
    geom_column="geometry",
    keep_geoms=["POLYGON", "MULTIPOLYGON"], 
    spatial_index=True,
    out_parquet=None,
    printout=False
):
    """
    Read a list of GeoParquet files into DuckDB, keeps only a consolidated list 
    of columns, union these geoparquets into a single table, and uses coarse-resolution h3 grid to 
    aggregate labels and naively remove duplicates with ST_EQUAL and ST_OVERLAPS.
    This function also creates a spatial index on the geometry column if specified.
    
    Args:
        parquet_files (list): List of paths to input parquet files
        out_db_file (str): Path to DuckDB database file
        table_name (str): Name of the consolidated table in DuckDB
        geom_column (str): Name of the geometry column in the parquet files
        spatial_index (bool): If True, create a spatial index on the geometry column
        out_parquet (str): Optional path to write consolidated table out as a GeoParquet file
        printout (bool): If True, prints statistics
        
    Returns:
        out_db_file (str): Path to the DuckDB database file
    """
    global DB_PATH
    # make sure path exists
    os.makedirs(DB_PATH, exist_ok=True)
    out_db_file = os.path.join(DB_PATH, f"{table_name}.duckdb")
    # remove any existing lock file so we can load existing db file
    if os.path.exists(out_db_file + ".wal"):
        print(f"Removing existing lock file: {out_db_file}.wal")
        os.remove(out_db_file + ".wal")

    print(f"Connecting to DuckDB database: {out_db_file}")
    conn = duckdb.connect(database=out_db_file, read_only=False)
    conn.install_extension("spatial")
    conn.load_extension("spatial")
    
    # Drop any existing table
    conn.execute(f"DROP TABLE IF EXISTS {table_name};")
    
    scans = []
    excluded_geoms_count = 0
    print(f"Building temp UNION query from {len(parquet_files)} parquet files...")
    
    for path in parquet_files:
        print(f"  Processing file: {os.path.basename(path)}")
        # Derive a dataset name from the parquet file's basename
        dataset_name = os.path.basename(str(path)).split('.')[0]
        
        try:
            # Read the parquet schema to get available column names
            schema = pq.read_schema(str(path))
            cols = set(schema.names)
            required_cols = {'area_m2', 'centroid_lon', 'centroid_lat', 'bbox', geom_column}
            
            # Check for required columns
            missing_cols = required_cols - cols
            if missing_cols:
                print(f"    WARNING: Skipping file {path}. Missing required columns: {missing_cols}")
                continue
            
            uid_candidates = ['fid', 'sol_id', 'GID_0', 'polygon_id', 'unique_id']
            present_uids = [f"CAST({uid} AS VARCHAR)" for uid in uid_candidates if uid in cols]

            # If none of the uid columns are present, use NULL
            if not present_uids:
                present_uids = ["NULL"]

            uid_expr = "COALESCE(" + ", ".join(present_uids) + ", 'NO_ID_' || ROW_NUMBER() OVER ())"  # Add fallback

            # Build SQL query for this file
            # filter out Point and LineString by default until we have heuristic for extracting meaningful PV polygon label from point
            scan_query = f"""
                SELECT 
                  sha256(concat_ws('_', {uid_expr}, '{dataset_name}')) AS unified_id,
                  area_m2,
                  ST_Area_Spheroid(geometry) AS spheroid_area,
                  centroid_lon,
                  centroid_lat,
                  '{dataset_name}' AS dataset,
                  bbox, 
                  ST_GeomFromWKB(TRY_CAST({geom_column} AS WKB_BLOB)) AS geometry
                FROM read_parquet('{str(path)}')
                WHERE ST_GeometryType(geometry) in ({', '.join([f"'{g}'" for g in keep_geoms])})
            """
            cnt_query = f"""SELECT COUNT(*) FROM read_parquet('{str(path)}') WHERE ST_GeometryType(geometry) not in ({', '.join([f"'{g}'" for g in keep_geoms])});"""
            excluded_geoms_count += conn.execute(cnt_query).fetchone()[0]
            
            scans.append(scan_query)
        except Exception as e:
            print(f"    ERROR processing file {path}: {e}")

    if not scans:
        print("No valid parquet files found. Exiting.")
        conn.close()
        return None
    
    # Create the UNION ALL query
    union_sql = "\nUNION ALL BY NAME\n".join(scans)
    
    try:
        # First create the table with explicit column types and constraints
        conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} (
            unified_id VARCHAR PRIMARY KEY,
            area_m2 DOUBLE,
            spheroid_area DOUBLE,
            centroid_lon DOUBLE,
            centroid_lat DOUBLE,
            dataset VARCHAR,
            bbox STRUCT(xmin DOUBLE, ymin DOUBLE, xmax DOUBLE, ymax DOUBLE),
            geometry GEOMETRY
        );
        """)
        
        # Then populate it with our existing query
        conn.execute(f"INSERT INTO {table_name} {union_sql};")
        # replace NaN or NULL values in numeric columns with -1 (to indicate invalid)
        conn.execute(f"UPDATE {table_name} SET spheroid_area = -1.0 WHERE spheroid_area IS NULL OR isnan(spheroid_area);")
        count_tmp = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Excluded {excluded_geoms_count} geometries that are not of type {', '.join(keep_geoms)}")
        
        if count_tmp == 0:
            print("ERROR: Temporary table is empty after UNION. Check input files and queries.")
            conn.close()
            return None
            
    except Exception as e:
        print(f"ERROR creating temporary table: {e}")
        print("UNION SQL used:")
        print(union_sql)
        conn.close()
        return None

    # Optionally create a spatial index on the temporary table
    if spatial_index:
        t1 = time.time()
        conn.execute(f"CREATE INDEX pv_idx ON {table_name} USING RTREE ({geom_column});")
        t2 = time.time()
        print(f"Spatial index created on {table_name} (used for dedup) in {t2 - t1:.2f} seconds.")
    
    # Print statistics if requested
    if printout:
        # Area distribution (min, max, avg)
        area_stats = conn.execute(
            f"SELECT MIN(area_m2), MAX(area_m2), AVG(area_m2) FROM {table_name}"
        ).fetchone()
        # Bounding box based on centroids
        bbox = conn.execute(
            f"SELECT MIN(centroid_lon), MAX(centroid_lon), MIN(centroid_lat), MAX(centroid_lat) FROM {table_name}"
        ).fetchone()
        
        print(f"Table '{table_name}' created with raw count (before dedup) of {count_tmp} records.")
        print(f"Area (m²): min = {area_stats[0]}, max = {area_stats[1]}, avg = {area_stats[2]}")
        print(f"Centroid extent: lon [{bbox[0]}, {bbox[1]}], lat [{bbox[2]}, {bbox[3]}]")
    
    # Optionally export consolidated table to geoparquet
    if out_parquet:
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        # DuckDB can export tables or query results to parquet using COPY
        conn.execute(f"COPY {table_name} TO '{out_parquet}' (FORMAT 'parquet', COMPRESSION 'zstd', ROW_GROUP_SIZE 10000);")
        print(f"Exported consolidated table to {out_parquet}")

    conn.close()
    print(f"Consolidated {len(parquet_files)} files into {out_db_file} → table '{table_name}'")
    
    if printout:
        print(f"Exported to: {out_parquet}")

    return out_db_file