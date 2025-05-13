"""
Visualization utilities for PV installation datasets.
This module provides functions for creating interactive maps and 
visualizations of PV installation data using Folium and PyDeck.
"""

import random
from typing import List, Optional, Dict, Union, Tuple
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='folium')
warnings.filterwarnings("ignore", category=UserWarning, module='pydeck')
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import pydeck as pdk
import ipywidgets as widgets
from IPython.display import display

# ================= DATASET METADATA =================

with open('dataset_metadata.json', 'r') as f:
    dataset_metadata = json.load(f)

# ================= ipywidgets formatting and helpers =================
def format_dataset_info(dataset):
    """Create a formatted HTML table for dataset metadata"""
    global dataset_metadata
    metadata = dataset_metadata[dataset]
    
    # Create table with metadata
    html = f"""
    <style>
    .dataset-table {{
        border-collapse: collapse;
        width: 30%;
        margin: 20px auto;
        font-family: Arial, sans-serif;
    }}
    .dataset-table th, .dataset-table td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }}
    .dataset-table th {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    </style>
    <table class="dataset-table">
        <tr><th>Metadata</th><th>Value</th></tr>
        <tr><td>DOI/URL</td><td>{metadata['doi']}</td></tr>
        <tr><td>Repository</td><td>{metadata['repo']}</td></tr>
        <tr><td>Compression</td><td>{metadata['compression'] or 'None'}</td></tr>
        <tr><td>Label Format</td><td>{metadata['label_fmt']}</td></tr>
        <tr><td>Has Images</td><td>{metadata['has_imgs']}</td></tr>
        <tr><td>Label Count</td><td>{metadata.get('label_count', 'Unknown')}</td></tr>
    </table>
    """
    return html

# helper function for creating a "Show Map" and "Clear Output" button with dedicated output area
def create_map_buttons(map_object):
    # Create taller buttons with custom styling
    button_layout = widgets.Layout(
        height='40px',
        width='150px',
        margin_left='2em',
        margin_right='2em',
        display='flex',
        align_items='center',
        justify_content='center'
    )
    
    # Create an output widget for the map
    output_area = widgets.Output(
        layout=widgets.Layout(
            height='500px',
            border='2px solid #ddd',
            overflow='auto',
            align_content='center'
        )
    )
    
    show_map_button = widgets.Button(description="Show Map", button_style='info', layout=button_layout)
    clear_output_button = widgets.Button(description="Clear Output", button_style='warning', layout=button_layout)

    def on_show_map_clicked(b):
        output_area.clear_output(wait=True)
        with output_area:
            display(map_object)
    
    def on_clear_output_clicked(b):
        output_area.clear_output(wait=True)
    
    show_map_button.on_click(on_show_map_clicked)
    clear_output_button.on_click(on_clear_output_clicked)
    
    # Return buttons and output area in a horizontal layout
    controls = widgets.HBox([show_map_button, clear_output_button])
    return widgets.VBox([controls, output_area])


# ================= FOLIUM VISUALIZATION FUNCTIONS =================

def create_folium_cluster_map(gdf, zoom_start=5, title="PV Installation Clusters"):
    """
    Create a cluster map of PV installations using Folium.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing PV installation data with geometry column
    zoom_start : int
        Initial zoom level for the map
    title : str
        Title for the map
        
    Returns:
    --------
    folium.Map
        Interactive Folium map with clustered markers
    """
    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) for Folium compatibility
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get centroid of all points to center the map
    center_lat = gdf.geometry.centroid.y.mean()
    center_lon = gdf.geometry.centroid.x.mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start,
                  tiles='CartoDB positron')
    
    # Add title
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each PV installation
    for idx, row in gdf.iterrows():
        # Get the centroid if the geometry is a polygon
        if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
            centroid = row.geometry.centroid
            popup_text = f"ID: {idx}"
            
            # Add additional information if available in the dataframe
            for col in ['capacity_mw', 'area_m2', 'installation_date', 'dataset', 'cen']:
                if col in gdf.columns:
                    popup_text += f"<br>{col}: {row[col]}"
            
            folium.Marker(
                location=[centroid.y, centroid.x],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='green', icon='solar-panel', prefix='fa')
            ).add_to(marker_cluster)
    
    return m


def create_folium_choropleth(gdf, column, bins=8, cmap='YlOrRd', 
                             title="PV Installation Density"):
    """
    Create a choropleth map of PV installations using Folium.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing PV installation data w2ith geometry column
    column : str
        Column name to use for choropleth coloring
    bins : int
        Number of bins for choropleth map
    cmap : str
        Matplotlib colormap name
    title : str
        Title for the map
        
    Returns:
    --------
    folium.Map
        Interactive Folium choropleth map
    """
    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get centroid of all points to center the map
    center_lat = gdf.geometry.centroid.y.mean()
    center_lon = gdf.geometry.centroid.x.mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3,
                  tiles='CartoDB positron')
    
    # Add title
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create choropleth layer
    folium.Choropleth(
        geo_data=gdf,
        name='choropleth',
        data=gdf,
        columns=[gdf.index.name if gdf.index.name else 'index', column],
        key_on='feature.id',
        fill_color=cmap,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=column,
        bins=bins
    ).add_to(m)
    
    # Add hover functionality
    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color': '#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color': '#000000', 
                                    'fillOpacity': 0.5, 
                                    'weight': 0.1}
    
    # Add tooltips
    folium.GeoJson(
        gdf,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[column],
            aliases=[column.replace('_', ' ').title()],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def create_folium_heatmap(gdf, intensity_column=None, radius=15, 
                          title="PV Installation Heatmap"):
    """
    Create a heatmap of PV installations using Folium.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing PV installation data with geometry column
    intensity_column : str, optional
        Column name to use for heatmap intensity; if None, all points have equal weight
    radius : int
        Radius for heatmap points (in pixels)
    title : str
        Title for the map
        
    Returns:
    --------
    folium.Map
        Interactive Folium heatmap
    """
    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get centroids for all geometries
    if any(gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])):
        centroids = gdf.geometry.centroid
    else:
        centroids = gdf.geometry
    
    # Get coordinates for heatmap
    heat_data = [[point.y, point.x] for point in centroids]
    
    # Add intensity if specified
    if intensity_column and intensity_column in gdf.columns:
        heat_data = [[point.y, point.x, float(intensity)] 
                    for point, intensity in zip(centroids, gdf[intensity_column])]
    
    # Get centroid of all points to center the map
    center_lat = sum(point[0] for point in heat_data) / len(heat_data)
    center_lon = sum(point[1] for point in heat_data) / len(heat_data)
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4,
                  tiles='CartoDB positron')
    
    # Add title
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        radius=radius,
        blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)
    
    return m


# ================= PYDECK VISUALIZATION FUNCTIONS =================

def get_pydeck_color(df, color_column=None, cmap='viridis', default_color=[0, 128, 0, 180]):
    """
    Generates color mapping instructions for Pydeck layers.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        color_column (str, optional): Column to base color on. Defaults to None.
        cmap (str, optional): Matplotlib colormap name for numeric data. Defaults to 'viridis'.
        default_color (list, optional): Default RGBA color if no column specified. Defaults to green.

    Returns:
        tuple: (Updated DataFrame with color columns if needed, Pydeck color accessor string/list)
    """
    if not color_column or color_column not in df.columns:
        return df, default_color # Return default color list

    # Ensure color column NaNs are handled
    if df[color_column].isnull().any():
        print(f"Warning: NaN values found in color column '{color_column}'. Filling with placeholder/mean.")
        if pd.api.types.is_numeric_dtype(df[color_column]):
            fill_val = df[color_column].mean() # Or median, or 0
            df[color_column] = df[color_column].fillna(fill_val)
        else:
            fill_val = 'Unknown'
            df[color_column] = df[color_column].astype(str).fillna(fill_val)

    if pd.api.types.is_numeric_dtype(df[color_column]):
        # --- Numeric Column: Use a colormap ---
        try:
            from matplotlib import colormaps
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=df[color_column].min(), vmax=df[color_column].max())
            colormap = colormaps[cmap]

            # Apply colormap and convert to RGBA 0-255 format
            colors = colormap(norm(df[color_column].values)) * 255
            df['__color_r'] = colors[:, 0].astype(int)
            df['__color_g'] = colors[:, 1].astype(int)
            df['__color_b'] = colors[:, 2].astype(int)
            # Use a fixed alpha or make it dynamic if needed
            df['__color_a'] = default_color[3] if len(default_color) == 4 else 180

            return df, '[__color_r, __color_g, __color_b, __color_a]'
        except ImportError:
            print("Matplotlib not found. Falling back to default color for numeric column.")
            return df, default_color
        except Exception as e:
            print(f"Error applying numeric colormap: {e}. Falling back to default color.")
            return df, default_color

    else:
        # --- Categorical Column: Assign random colors ---
        unique_cats = df[color_column].astype(str).unique()
        # Generate pseudo-random but consistent colors for categories
        color_map = {
            cat: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), default_color[3] if len(default_color) == 4 else 180]
            for cat in unique_cats
        }
        # Add 'Unknown' category if needed
        if 'Unknown' not in color_map:
             color_map['Unknown'] = [128, 128, 128, default_color[3] if len(default_color) == 4 else 180] # Grey for unknowns

        df['__color_r'] = df[color_column].astype(str).map(lambda x: color_map.get(x, color_map['Unknown'])[0])
        df['__color_g'] = df[color_column].astype(str).map(lambda x: color_map.get(x, color_map['Unknown'])[1])
        df['__color_b'] = df[color_column].astype(str).map(lambda x: color_map.get(x, color_map['Unknown'])[2])
        df['__color_a'] = df[color_column].astype(str).map(lambda x: color_map.get(x, color_map['Unknown'])[3])

        return df, '[__color_r, __color_g, __color_b, __color_a]'


def create_pydeck_scatterplot(gdf: gpd.GeoDataFrame,
                              color_column: Optional[str] = None,
                              size_column: Optional[str] = None,
                              size_scale: float = 50.0,
                              tooltip_cols: Optional[List[str]] = None,
                              map_style: str = 'light',
                              initial_zoom: int = 3):
    """
    Creates an interactive scatterplot map using PyDeck, plotting centroids.

    Args:
        gdf: Input GeoDataFrame. Assumed to have a 'geometry' column.
        color_column: Column name to use for point color.
        size_column: Column name to use for point size (radius).
                     Uses square root scaling for better visual perception.
        size_scale: Scaling factor for point radius.
        tooltip_cols: List of column names to include in the tooltip.
                      Defaults to common useful columns if None.
        map_style: Pydeck map style (e.g., 'light', 'dark', 'satellite').
        initial_zoom: Initial zoom level for the map.

    Returns:
        pydeck.Deck: A PyDeck map object, or None if input is invalid.
    """
    if gdf is None or gdf.empty:
        print("Input GeoDataFrame is empty or None.")
        return None
    if 'geometry' not in gdf.columns:
        print("GeoDataFrame must have a 'geometry' column.")
        return None

    print(f"Creating Pydeck scatterplot for {len(gdf)} features...")

    # Ensure correct CRS
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting GDF from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    # Prepare DataFrame for Pydeck
    df = pd.DataFrame()
    # Use centroid for plotting points
    df['lon'] = gdf['centroid_lon']
    df['lat'] = gdf['centroid_lat']

    # Copy relevant attribute columns
    potential_tooltip_cols = ['unified_id', 'dataset', 'area_m2', 'capacity_mw', 'install_date']
    if tooltip_cols is None:
        tooltip_cols = [col for col in potential_tooltip_cols if col in gdf.columns]
    else:
        # Ensure requested tooltip columns exist
        tooltip_cols = [col for col in tooltip_cols if col in gdf.columns]

    # Include color and size columns in the DataFrame if they exist and are needed
    cols_to_copy = set(tooltip_cols)
    if color_column and color_column in gdf.columns:
        cols_to_copy.add(color_column)
    if size_column and size_column in gdf.columns:
        cols_to_copy.add(size_column)

    for col in cols_to_copy:
        df[col] = gdf[col]

    # add slight increment to size_column to avoid zero size
    if size_column and size_column in gdf.columns and pd.api.types.is_numeric_dtype(df[size_column]):
        df[size_column] = df[size_column].fillna(0) + 1e-1 # Avoid zero size for sqrt scaling

    # --- Color Mapping ---
    df, color_accessor = get_pydeck_color(df, color_column)

    # --- Size Mapping ---
    if size_column and size_column in gdf.columns and pd.api.types.is_numeric_dtype(df[size_column]):
        # Use sqrt scaling for better visual perception of area/capacity
        # Handle potential negative values if necessary before sqrt
        min_val = df[size_column].min()
        if min_val < 0:
             print(f"Warning: Negative values found in size column '{size_column}'. Clamping to 0 for sqrt.")
             df['__size_val'] = np.sqrt(df[size_column].clip(lower=0).fillna(0)) * size_scale
        else:
             df['__size_val'] = np.sqrt(df[size_column].fillna(0)) * size_scale
        size_accessor = '__size_val'
        print(f"Using column '{size_column}' for point size (sqrt scaled).")
    else:
        if size_column:
             print(f"Warning: Size column '{size_column}' not found or not numeric. Using fixed size.")
        size_accessor = size_scale # Fixed radius if no valid column
        print(f"Using fixed point size: {size_scale}")


    # --- Tooltip ---
    tooltip_html = ""
    for col in tooltip_cols:
        # Format column name nicely for display
        col_display = col.replace('_', ' ').title()
        tooltip_html += f"<b>{col_display}:</b> {{{col}}}<br>"
    tooltip = {"html": tooltip_html.strip('<br>')} if tooltip_html else None


    # --- Layer ---
    layer = pdk.Layer(
        'ScatterplotLayer',
        df,
        get_position=['lon', 'lat'],
        get_radius=size_accessor,
        get_fill_color=color_accessor,
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_min_pixels=1,
        radius_max_pixels=100,
        get_line_color=[0, 0, 0, 100], # Faint black outline
        line_width_min_pixels=0.5
    )

    # --- View State ---
    view_state = pdk.ViewState(
        longitude=df['lon'].mean(),
        latitude=df['lat'].mean(),
        zoom=initial_zoom,
        pitch=0,
        bearing=0
    )

    # --- Deck ---
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=map_style
    )
    print("Pydeck scatterplot created successfully.")
    return deck


def create_pydeck_polygons(gdf: gpd.GeoDataFrame,
                           color_column: Optional[str] = None,
                           extrusion_column: Optional[str] = None,
                           extrusion_scale: float = 1.0,
                           tooltip_cols: Optional[List[str]] = None,
                           map_style: str = 'light',
                           initial_zoom: int = 10,
                           sample_frac: float = 1.0,
                           where_clause: Optional[str] = None):
    """
    Creates an interactive 3D polygon map using PyDeck.
    NOTE: Rendering large numbers of complex polygons can be slow.
          Consider using sample_frac or where_clause for large datasets.

    Args:
        gdf: Input GeoDataFrame. Must contain Polygon/MultiPolygon geometries.
        color_column: Column name to use for polygon color.
        extrusion_column: Column name to use for polygon height extrusion.
                          Uses linear scaling.
        extrusion_scale: Scaling factor for extrusion height. Applied AFTER normalization if numeric.
        tooltip_cols: List of column names to include in the tooltip.
        map_style: Pydeck map style.
        initial_zoom: Initial zoom level.
        sample_frac: Fraction of data to sample (0.0 to 1.0). 1.0 means no sampling.
        where_clause: A Pandas query string to filter the GDF before visualization.

    Returns:
        pydeck.Deck: A PyDeck map object, or None if input is invalid.
    """
    if gdf is None or gdf.empty:
        print("Input GeoDataFrame is empty or None.")
        return None
    if 'geometry' not in gdf.columns:
        print("GeoDataFrame must have a 'geometry' column.")
        return None

    # --- Filtering and Sampling ---
    original_count = len(gdf)
    if where_clause:
        try:
            gdf = gdf.query(where_clause).copy()
            print(f"Filtered GDF using '{where_clause}'. Count: {len(gdf)} (from {original_count})")
        except Exception as e:
            print(f"Error applying where_clause '{where_clause}': {e}. Using original GDF.")
            gdf = gdf.copy() # Ensure we have a copy
    else:
         gdf = gdf.copy() # Ensure we have a copy

    if sample_frac < 1.0 and len(gdf) > 0:
        try:
            gdf = gdf.sample(frac=sample_frac, random_state=42) # Use random_state for reproducibility
            print(f"Sampled {sample_frac*100:.1f}% of data. Count: {len(gdf)} (from {original_count if not where_clause else 'filtered'})")
        except Exception as e:
             print(f"Error sampling data: {e}. Using full (or filtered) GDF.")
    # --- End Filtering and Sampling ---


    # Ensure correct CRS
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting GDF from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    # Filter for valid Polygon/MultiPolygon geometries
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon']) & gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    if gdf.empty:
        print("No valid Polygon or MultiPolygon geometries found after filtering.")
        return None

    print(f"Creating Pydeck polygon map for {len(gdf)} features...")

    # Prepare data for Pydeck PolygonLayer
    # Using __geo_interface__ is generally more efficient than manual iteration
    gdf['__geojson_feature'] = gdf.apply(lambda row: row.geometry.__geo_interface__, axis=1)

    # Extract properties into the DataFrame for easier access by Pydeck accessors
    potential_tooltip_cols = ['unified_id', 'dataset', 'area_m2', 'capacity_mw', 'install_date']
    if tooltip_cols is None:
        tooltip_cols = [col for col in potential_tooltip_cols if col in gdf.columns]
    else:
        tooltip_cols = [col for col in tooltip_cols if col in gdf.columns] # Ensure they exist

    cols_to_keep = set(tooltip_cols)
    if color_column and color_column in gdf.columns:
        cols_to_keep.add(color_column)
    if extrusion_column and extrusion_column in gdf.columns:
        cols_to_keep.add(extrusion_column)
    cols_to_keep.add('__geojson_feature') # Add the geometry representation

    df_pydeck = gdf[list(cols_to_keep)].copy()


    # --- Color Mapping ---
    df_pydeck, color_accessor = get_pydeck_color(df_pydeck, color_column, default_color=[0, 128, 0, 150]) # Slightly transparent default

    # --- Extrusion Mapping ---
    if extrusion_column and extrusion_column in gdf.columns and pd.api.types.is_numeric_dtype(df_pydeck[extrusion_column]):
        # Normalize extrusion height for better scaling, handle NaNs
        min_val = df_pydeck[extrusion_column].min()
        max_val = df_pydeck[extrusion_column].max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
             print(f"Warning: Cannot normalize extrusion column '{extrusion_column}'. Using fixed extrusion.")
             df_pydeck['__elevation_val'] = 1.0 # Use a base value
        else:
             df_pydeck['__elevation_val'] = (df_pydeck[extrusion_column].fillna(min_val) - min_val) / (max_val - min_val)

        # Apply overall scale factor
        elevation_accessor = f'__elevation_val * {extrusion_scale * 1000}' # Multiply scale for visibility
        print(f"Using column '{extrusion_column}' for extrusion (normalized & scaled by {extrusion_scale*1000}).")
        extruded = True
    else:
        if extrusion_column:
            print(f"Warning: Extrusion column '{extrusion_column}' not found or not numeric. No extrusion applied.")
        elevation_accessor = 0 # No height
        extruded = False
        print("No extrusion applied.")


    # --- Tooltip ---
    tooltip_html = ""
    for col in tooltip_cols:
        col_display = col.replace('_', ' ').title()
        tooltip_html += f"<b>{col_display}:</b> {{{col}}}<br>"
    tooltip = {"html": tooltip_html.strip('<br>')} if tooltip_html else None

    # --- Layer ---
    layer = pdk.Layer(
        'GeoJsonLayer', # Use GeoJsonLayer which handles Polygon/MultiPolygon via __geo_interface__
        df_pydeck,
        opacity=0.7,
        stroked=True, # Wireframe
        filled=True,
        extruded=extruded,
        wireframe=True,
        get_polygon='__geojson_feature', # Access the geojson geometry
        get_fill_color=color_accessor,
        get_line_color=[0, 0, 0, 100],
        get_line_width=10, # Meters
        line_width_min_pixels=0.5,
        get_elevation=elevation_accessor if extruded else 0,
        pickable=True,
        auto_highlight=True
    )

    # --- View State ---
    # Calculate bounds more efficiently
    bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    view_state = pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=initial_zoom,
        pitch=45 if extruded else 0, # Pitch only if extruded
        bearing=0
    )

    # --- Deck ---
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=map_style
    )
    print("Pydeck polygon map created successfully.")
    return deck


def create_pydeck_heatmap(gdf: gpd.GeoDataFrame,
                          weight_column: Optional[str] = None,
                          radius_pixels: int = 50,
                          intensity: float = 1.0,
                          threshold: float = 0.05,
                          aggregation: str = 'SUM',
                          tooltip_cols: Optional[List[str]] = None,
                          map_style: str = 'light',
                          initial_zoom: int = 3):
    """
    Creates an interactive heatmap using PyDeck.

    Args:
        gdf: Input GeoDataFrame.
        weight_column: Column name to use for heatmap weighting. If None, uses count.
        radius_pixels: Radius of influence for each point in pixels.
        intensity: Multiplier factor for heatmap intensity.
        threshold: Minimum threshold for color rendering (0 to 1).
        aggregation: 'SUM' or 'MEAN' - how weights are accumulated.
        tooltip_cols: (Currently ignored by HeatmapLayer) List of columns for tooltip.
        map_style: Pydeck map style.
        initial_zoom: Initial zoom level.

    Returns:
        pydeck.Deck: A PyDeck map object, or None if input is invalid.
    """
    if gdf is None or gdf.empty:
        print("Input GeoDataFrame is empty or None.")
        return None
    if 'geometry' not in gdf.columns:
        print("GeoDataFrame must have a 'geometry' column.")
        return None

    print(f"Creating Pydeck heatmap for {len(gdf)} features...")

    # Ensure correct CRS
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting GDF from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    # Prepare DataFrame for Pydeck
    df = pd.DataFrame()
    df['lon'] = gdf['centroid_lon']
    df['lat'] = gdf['centroid_lat']

    # --- Weight Mapping ---
    if weight_column and weight_column in gdf.columns and pd.api.types.is_numeric_dtype(gdf[weight_column]):
         # Handle NaNs - fill with 0 or mean/median depending on desired behavior
        if gdf[weight_column].isnull().any():
            print(f"Warning: NaN values found in weight column '{weight_column}'. Filling with 0.")
            df['__weight'] = gdf[weight_column].fillna(0)
        else:
            df['__weight'] = gdf[weight_column]
        weight_accessor = '__weight'
        print(f"Using column '{weight_column}' for heatmap weights (Aggregation: {aggregation}).")
    else:
        if weight_column:
            print(f"Warning: Weight column '{weight_column}' not found or not numeric. Using count (weight=1).")
        weight_accessor = 1 # Use count if no valid weight column
        print(f"Using count (weight=1) for heatmap.")

    # --- Layer ---
    # Viridis color range (adjust alpha as needed)
    color_range = [
        [68, 1, 84, 255],
        [72, 40, 120, 255],
        [62, 74, 137, 255],
        [49, 104, 142, 255],
        [38, 130, 142, 255],
        [31, 158, 137, 255],
        [53, 183, 121, 255],
        [109, 205, 89, 255],
        [180, 222, 44, 255],
        [253, 231, 37, 255]
    ]

    layer = pdk.Layer(
        'HeatmapLayer',
        df,
        get_position=['lon', 'lat'],
        get_weight=weight_accessor,
        opacity=0.8,
        radius_pixels=radius_pixels,
        intensity=intensity,
        threshold=threshold,
        aggregation=aggregation.upper(), # Ensure uppercase
        color_range=color_range # Use Viridis colormap
    )

    # --- View State ---
    view_state = pdk.ViewState(
        longitude=df['lon'].mean(),
        latitude=df['lat'].mean(),
        zoom=initial_zoom,
        pitch=0,
        bearing=0
    )

    # --- Deck ---
    # Tooltip is generally not useful for HeatmapLayer
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=False # Disable tooltip for heatmap
    )
    print("Pydeck heatmap created successfully.")
    return deck