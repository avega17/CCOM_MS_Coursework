# Remote Sensing Assessment and Monitoring of Distributed Rooftop Solar Panel Arrays

Here we present the source, technical report, and artifacts/deliverables for our
initial investigation and review of Computer Vision and Remote Sensing methods for the
detection of Photovoltaic (PV) solar panels (with a particular focus on rooftop
installations) and the forecasting and modeling of their short-term energy
production using Multispectral satellite imagery with very high spatial-resolution
(< 1m/pixel), and geostationary satellite sensors with very high temporal-resolution
(scans every 5-15 minutes), respectively.
This repo, report, and several of the notebooks also serve as deliverables
for the courses CCOM6102: Computer Vision and CCOM6050: Design & Analysis of Algorithms
where we explore two constituent subproblems for the project as part of the
following courses:

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

Coursework will likely be limited to testing clear sky and similar solar irradiance estimation baseline techniques, and comparing their running time, and space and time complexity to 
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
- [pystac-client](https://pystac-client.readthedocs.io/en/latest/usage.html#itemsearch) for searching and filtering STAC catalog items
- [maxar-platform](https://developers.maxar.com/docs/developer-tools/python-sdk/) 
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

## Datasets 

### PV Solar Panel Inventory/Locations (with and without imagery):

- "Distributed solar photovoltaic array location and extent dataset for remote sensing object identification" - K. Bradbury, 2016 | [paper DOI](https://doi.org/10.1038/sdata.2016.106) | [dataset DOI](https://doi.org/10.6084/m9.figshare.3385780.v4) | polygon annotations for 19,433 PV modules in 4 cities in California, USA
- "A solar panel dataset of very high resolution satellite imagery to support the Sustainable Development Goals" - C. Clark et al, 2023 | [paper DOI](https://doi.org/10.1038/s41597-023-02539-8) | [dataset DOI](https://doi.org/10.6084/m9.figshare.22081091.v3) | 2,542 object labels (per spatial resolution)
- "A harmonised, high-coverage, open dataset of solar photovoltaic installations in the UK" - D. Stowell et al, 2020 | [paper DOI](https://doi.org/10.1038/s41597-020-00739-0) | [dataset DOI](https://zenodo.org/records/4059881) | 265,418 data points (over 255,000 are stand-alone installations, 1067 solar farms, and rest are subcomponents within solar farms)
- "Georectified polygon database of ground-mounted large-scale solar photovoltaic sites in the United States" - K. Sydny, 2023 | [paper DOI](https://doi.org/10.1038/s41597-023-02644-8) | [dataset DOI](https://www.sciencebase.gov/catalog/item/6671c479d34e84915adb7536) | 4186 data points (Note: these correspond to PV _facilities_ rather than individual panel arrays or objects and need filtering of duplicates with other datasets and further processing to extract the PV arrays in the facility)
- "Vectorized solar photovoltaic installation dataset across China in 2015 and 2020" - J. Liu et al, 2024 | [paper DOI](https://doi.org/10.1038/s41597-024-04356-z) | [dataset link](https://github.com/qingfengxitu/ChinaPV) | 3,356 PV labels (inspect quality!)
- "Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery" - H. Jiang, 2021 | [paper DOI](https://doi.org/10.5194/essd-13-5389-2021) | [dataset DOI](https://doi.org/10.5281/zenodo.5171712) | 3,716 samples of PV data points
- "A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata" - G. Kasmi, 2023 | [paper DOI](https://doi.org/10.1038/s41597-023-01951-4) | [dataset DOI](https://doi.org/10.5281/zenodo.6865878) | > 28K points of PV installations; 13K+ segmentation masks for PV arrays; metadata for 8K+ installations
- "An Artificial Intelligence Dataset for Solar Energy Locations in India" - A. Ortiz, 2022 | [paper DOI](https://doi.org/10.1038/s41597-022-01499-9) | [dataset link 1](https://researchlabwuopendata.blob.core.windows.net/solar-farms/solar_farms_india_2021.geojson) or [dataset link 2](https://raw.githubusercontent.com/microsoft/solar-farms-mapping/refs/heads/main/data/solar_farms_india_2021_merged_simplified.geojson) | 117 geo-referenced points of solar installations across India
- "GloSoFarID: Global multispectral dataset for Solar Farm IDentification in satellite imagery" - Z. Yang, 2024 | [paper DOI](https://doi.org/10.48550/arXiv.2404.05180) | [dataset DOI](https://github.com/yzyly1992/GloSoFarID/tree/main/data_coordinates) | 6,793 PV samples across 3 years (double counting of samples)
- "A global inventory of photovoltaic solar energy generating units" - L. Kruitwagen et al, 2021 | [paper DOI](https://doi.org/10.1038/s41586-021-03957-7) | [dataset DOI](https://doi.org/10.5281/zenodo.5005867) | 50,426 for training, cross-validation, and testing; 68,661 predicted polygon labels 
- "Harmonised global datasets of wind and solar farm locations and power" - S. Dunnett et al, 2020 | [paper DOI](https://doi.org/10.1038/s41597-020-0469-8) | [dataset DOI](https://doi.org/10.6084/m9.figshare.11310269.v6) | 35272 PV installations

## EO Medium to High Imagery via STAC collections

STAC (SpatioTemporal Asset Catalog) is a standard for describing geospatial information in a way that is easy to search and filter based on time, location, and other metadata. 
There are several medium resolution and high resolution EO imagery collections, alongside relevant historical geospatial  available via STAC which we non-exhaustively list below:

- Maxar's Global [Catalog](https://stacindex.org/catalogs/maxar-open-data-catalog-ard-format#/) from their [Open Data Program](https://www.maxar.com/open-data)
    - GSD's: 0.3m, 0.5m
    - Bands: 4-band (RGB + NIR) and 8-band (RGB + NIR + SWIR) depending on specific Catalog used
    - see a a (limited) interactive web viewer [here](https://xpress.maxar.com/) (use side-bar to select "Open Data" and a specific event/collection)
- [Earthview Satellogic Dataset](https://satellogic-earthview.s3.us-west-2.amazonaws.com/index.html) STAC
    - GSD: 1.0m
    - Bands: 4-band (RGB + NIR)
    - Time coverage: H2 2022
    - Imagery: contains 7 million images.
- Sentinel-2 catalogs [hosted in AWS](https://registry.opendata.aws/sentinel-2/)
    - See COG (Cloud Optimized GeoTIFF) subset [here](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
    - See ESA's WorldCover land cover maps for 2020 and 2021 [here](https://registry.opendata.aws/esa-worldcover-vito/)
    - GSD's: 10m, 20m, 60m (varies by band)
    - Bands: up to 13 bands (RGB + NIR + SWIR) depending on specific Catalog used
    - see interactive web viewer [here](https://viewer.aws.element84.com/)!
- [Microsoft Planetary Computer STAC API](https://stacindex.org/catalogs/microsoft-pc)
- [Planet Labs Open Data](https://www.planet.com/data/stac/browser/?.language=en) static catalog
- [NAIP (National Agriculture Imagery Program)](https://radiantearth.github.io/stac-browser/#/external/earth-search.aws.element84.com/v1/collections/naip) Catalog
    - GSD's: 0.6m, 1.0m
    - Bands: 4-band (RGB + NIR)
- See list of STAC catalogs in [opengeos github repo](https://github.com/opengeos/stac-index-catalogs)
- Registry of [Open Data hosted on AWS](https://registry.opendata.aws/) where many are STAC compliant
    - See [Earth Search API](https://element84.com/earth-search/) for STAC search and discovery of (a subset of) this registry
<!-- - Copernicus 30m DEM: https://radiantearth.github.io/stac-browser/#/external/earth-search.aws.element84.com/v1/collections/cop-dem-glo-30 -->
<!-- - [European Space Agency (ESA) Open Science Catalog](https://stacindex.org/catalogs/osc#/) -->
<!-- - [NASA's Common Metadata Repository STAC](https://stacindex.org/catalogs/cmr-stac#/) -->

## Geostationary or Weather-satellites with high-temporal resolution EO imagery via STAC collections

- JMA Himawari 8/9 [collection on AWS](https://registry.opendata.aws/noaa-himawari/)
- NOAA Global Mosaic of Geostationary Satellite Imagery (GMGSI) [AWS STAC collection](https://registry.opendata.aws/noaa-gmgsi/)
    - "composited from data from several geostationary satellites orbiting the globe, including the GOES-East and GOES-West Satellites operated by U.S. NOAA/NESDIS, the Meteosat-10 and Meteosat-9 satellites from theMeteosat Second Generation (MSG) series of satellites operated by European Organization for the Exploitation of Meteorological Satellites (EUMETSAT), and the Himawari-9 satellite operated by the Japan Meteorological Agency (JMA)"
    - "GMGSI composite images have an approximate 8 km (5 mile) horizontal resolution and are **updated every hour**"
- NOAA's [AWS registry](https://registry.opendata.aws/noaa-goes/) for GOES 16, 17, 18, and (new!) 19
    - "GOES satellites provide continuous weather imagery and monitoring of meteorological and space environment data across North America. GOES satellites provide the kind of continuous monitoring necessary for intensive data analysis. They hover continuously over one position on the surface. The satellites orbit high enough to allow for a full-disc view of the Earth. Because they stay above a fixed spot on the surface, they provide a constant vigil" 

## Spatio-temporal context data
This covers solar irradiance, temperature, metereological data, administrative boundaries, building vector datasets (for urban inference) etc.

- NREL NSRDB (National Renewable Energy Laboratory's National Solar Radiation Database) via [AWS STAC collection](https://registry.opendata.aws/nrel-pds-nsrdb/)
    - "a serially complete collection of hourly and half-hourly values of the three most common measurements of solar radiation – global horizontal, direct normal, and diffuse horizontal irradiance — and meteorological data"
    - see interactive web viewer [here](https://nsrdb.nrel.gov/data-viewer)!
    - *for Puerto Rico*: every 30/60 mins, 4km **from 1998-2019**!!
- NASA Prediction of Worldwide Energy Resources (POWER) [registry on AWS](https://registry.opendata.aws/nasa-power/)
    - "The POWER project contains over 380 satellite-derived meteorology and **solar energy Analysis Ready Data (ARD) at four temporal levels: hourly, daily, monthly, and climatology**. The POWER data archive provides data at the native resolution of the source products. The data is updated nightly to maintain **near real time availability** (2-3 days for meteorological parameters and **5-7 days for solar**). The POWER services catalog consists of a series of RESTful Application Programming Interfaces, geospatial enabled image services, and web mapping Data Access Viewer. These three service offerings support data discovery, access, and distribution to the project’s user base as ARD and as direct application inputs to decision support tools."
- Department of Energy's Open Energy Data Initiative (OEDI) [Data Lake registry on AWS](https://registry.opendata.aws/oedi-data-lake/)
- NSF NCAR Curated ECMWF Reanalysis 5 (ERA5) [registry on AWS](https://registry.opendata.aws/nsf-ncar-era5/)
