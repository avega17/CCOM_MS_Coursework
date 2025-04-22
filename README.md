# Remote Sensing Assessment and Monitoring of Distributed Rooftop Solar Panel Arrays

Here we present the source, technical report, and artifacts/deliverables for our
initial investigation and review of Computer Vision and Remote Sensing methods for:
1. detecting of Photovoltaic (PV) solar panels (particularly rooftop
installations) using Multispectral Satellite Imagery (MSI) with very high spatial-resolution
(< 1m/pixel in the case of [Maxar's entire constellation](https://www.maxar.com/maxar-intelligence/constellation))
2. using Data Products from MSI from Geostationary sensors with very high temporal-resolution
(e.g. scans every 5-15 minutes for NOAA GOES-R series)

This repo, included report [synced from Overleaf](https://www.overleaf.com/learn/how-to/GitHub_Synchronization), and several of the notebooks also serve as deliverables for the courses *CCOM6102: Computer Vision* and *CCOM6050: Design & Analysis of Algorithms*
where we explore two constituent subproblems for the project as part of the two courses:

## Computer Vision:

### Detection and assessment of distributed rooftop PV Systems using 

The first fundamental challenge lies in the automated, accurate, and scalable identification and geometric characterization
of distributed rooftop PV panels using very-high-resolution (VHR) multispectral satellite imagery.

Generating reliable, up-to-date inventories of these small, dispersed assets is
crucial for granular assessments of distributed PV installed capacity, infrastructure planning,
monitoring deployment rates against policy goals, and producing the georeferenced geometry data required for accurate
\textit{site-level} solar irradiance forecasting and energy generation estimates of distributed PV systems of all scales.

## Design & Analysis of Algorithms

### Data Fusion of EO imagery and spatio-temporal context applied to Solar Irradiance Time Series Forecasting 

Details TBD. See main reference, ["SolarCube: An Integrative Benchmark Dataset Harnessing Satellite and In-situ Observations for Large-scale Solar Energy Forecasting"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/06477eb61ea6b85c6608d42a222462df-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024 and their [corresponding repo](https://github.com/Ruohan-Li/SolarCube).

Coursework will likely be limited to testing clear sky and similar solar irradiance estimation techniques, and comparing their running time, and space and time complexity to 
the authors proposed Deep Learning based approach with much of the data and code taken as-is due to time constraints.

See their figure 1 below for a high-level overview of the dataset and tasks:

<figure style="text-align: center">
<img src="https://raw.githubusercontent.com/Ruohan-Li/SolarCube/master/images/final2.png" style="width:80%; height:auto;">
<figcaption align = "center"> SolarCube dataset composition, study areas, and baseline tasks  </figcaption>
</figure>

## Setup and Installation
### Conda 
Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [mamba](https://github.com/conda-forge/miniforge) (["a drop-in replacement for Conda that is generally faster and better at resolving dependencies"](https://statistics.berkeley.edu/computing/conda)) and create a new environment with the following command:
```bash
conda env create -f environment.yml
```
This environment has been tested on two different machines, but both are macOS ARM64 machines. Other OS and architectures are untested with the frozen versions listed in the environment.yml file.
Feel free to loosen or remove the version constraints in the environment.yml file if you encounter any issues with package installation and dependency resolution.

### Environment Variables and (future) API Keys
For the notebooks to run *as-is* you also need to create a .env file with the variables that are *not* commented out in the `env-template.txt` included in this repo for convenience.
Getting the notebook running with the default paths and variables is as simple as renaming the included env template file:
```bash
mv env-template.txt .env
```
Then, fill in the variables with your own values as needed. The variables are used to set up where data will be stored locally, the database connection file, and other environment variables.
See usage of `python-dotenv` [here](https://www.geeksforgeeks.org/using-python-environment-variables-with-python-dotenv/).

## Tools
- jupyter notebook/lab
- ipywidgets
- [torchgeo](https://www.osgeo.org/projects/torchgeo/) for datasets, geospatial data loaders, and transforms
- [torchvision](https://pytorch.org/vision/stable/index.html) for datasets, models, and transforms
- - [pytorch lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) for agile development and iteration, and enabling scaling 
- IBM's [terratorch](https://ibm.github.io/terratorch/architecture/) for use of Geospatial Foundation Models (GFMs) as baselines AND models to be fine-tuned
- [overture maps](https://docs.overturemaps.org/guides/buildings/#14/32.58453/-117.05154/0/60) for use in inferencing over rooftops 
- dbt core
- duckdb
- GDAL
- rasterio
- geopandas
- fiona
- shapely
- open data cube
- xarray
- cubo
- visualizations using one or more of: ipyleaflet, folium, lonboard, or pydeck for visualization
