# Senate Polarization Analysis

This repository supports the CCOM Data Analysis project on quantifying U.S. Senate polarization from the 87th through the 119th Congress. We iterate on the core logic in `asvn_project_political_polarization.py` and mirror milestone results into the notebook for visualization and presentation.

## Environment Setup

We prescribe [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management while retaining a `requirements.txt` for legacy `pip` workflows. uv ships with a Rust-powered, multi-threaded dependency resolver and cache-aware installer that typically outperforms `pip` by an order of magnitude on cold or warm installs.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add `~/.local/bin` (macOS/Linux) to your `PATH` if it is not already present.

### Using uv (recommended)

```bash
# create a Python 3.11 virtual environment for the project
uv venv --python 3.11
source .venv/bin/activate

# install and sync all dependencies declared in pyproject.toml
uv sync

# alternatively, install without creating lockfiles using pip compatibility mode
uv pip install .
```

uv caches wheels and resolution metadata, so subsequent `uv sync` runs are incremental and parallelised across CPU cores.

### Using the environment with Jupyter / VS Code

- **VS Code:** run the Command Palette → `Python: Select Interpreter` and choose `.venv` created by uv.
- **Classic Jupyter:**

	```bash
	source .venv/bin/activate
	python -m ipykernel install --user --name senate-polarization --display-name "ccom6994-polarization"
	```

	After registering the kernel, pick `Python (senate-polarization)` inside Notebook UIs.

### Using pip (fallback)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Roadmap

Each exercise builds on the DuckDB-powered ingestion pipeline to produce reproducible metrics and visualizations:

- **Exercise 1** — Parallel ingestion of vote files into DuckDB and creation of processed Senate vote tables feeding a vectorized analysis loop.
- **Exercise 2** — Dynamic fetching of newer Congress vote files, reintegration into our tables, and enhanced silhouette trend highlighting.
- **Exercise 3** — Cluster-party alignment metrics plus interactive PCA scatter exploration.
- **Exercise 4** — Additional polarization indices (Dunn, Davies–Bouldin, Calinski–Harabasz) and party separation percentages.
- **Exercise 5** — Party cohesion via cosine similarity, persisting senator-level scores in DuckDB with aggregate trends.
- **Exercise 6** — Dash dashboard delivering a grid of Plotly figures derived from the prior exercises.

## Approach Highlights

- **DuckDB-first ingestion** keeps the pipeline scalable while remaining interoperable with Pandas DataFrames when needed.
- **Function-driven plotting** ensures reproducibility, lets us surface identical visuals in both notebook and Dash contexts, and keeps heavy figure payloads out of version control.
- **Modern tooling compatibility** with uv, Plotly, and Dash accelerates local iteration yet remains accessible to standard Python environments via `requirements.txt`.

We will implement and validate each exercise incrementally in the Python module before syncing the notebook. Testing and large downloads stay out of automation to keep feedback loops tight.
