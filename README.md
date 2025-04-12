# Remote Sensing Assessment and Monitoring of Distributed Rooftop Solar Panel Arrays

Here we present the source, technical report, and artifacts/deliverables on our
initial exploration on Computer Vision and Remote Sensing methods for the
detection of Photovoltaic (PV) solar panels (with a particular focus on rooftop
installations) and the forecasting and modeling of their short-term energy
production using Multispectral satellites with very high spatial-resolution
(< 1m/pixel), and geostationary satellites with very high temporal-resolution
(scans every 5-15 minutes), respectively.

This repo, report, and several of the notebooks also serve as deliverables
where we explore two constituent subproblems for the project as part of the
following courses:

## CCOM6102: Computer Vision:

### Detection and assessment of distributed rooftop PV Systems using 

The first fundamental challenge lies in the automated, accurate, and scalable identification and geometric characterization of distributed rooftop PV panels
using very-high-resolution (VHR) multispectral satellite imagery.

Generating reliable, up-to-date inventories of these small, dispersed assets is
crucial for granular PV potential assessments, infrastructure planning,
monitoring deployment rates against policy goals, and producing the
georeferenced geometry data required for accurate \textit{site-level} solar irradiance
forecasting and energy generation estimates of distributed PV systems of all scales.

## CCOM6050: Design & Analysis of Algorithms

### Spatio-temporal Data Fusion for Solar Irradiance Time Series Forecasting

## Tools
- dbt core
- duckdb
- pytorch lightning
- GDAL
- rasterio
- geopandas
- fiona
- shapely
- open data cube
- xarray
- cubo