# %%
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dash>=2.16",
#     "duckdb~=1.4",
#     "ipywidgets>=8.1",
#     "juv>=0.4.0",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "plotly>=5.24",
#     "python-dotenv>=1.0",
#     "requests>=2.32",
#     "scikit-learn>=1.4",
# ]
# ///

# %% [markdown]
# # Quantifying Political Polarization in the US Senate
#
# **Project Title:** Senate Voting Polarization Analysis (87th-119th Congress)
# **Course:** CCOM6994: Data Analysis Tools
# **Author:** Alejandro S. Vega Nogales
#
# ---
#
# ## Project Overview
# ![image.png](attachment:image.png)
#
# This project investigates the question: **Has US Senate voting become more polarized across time?**
#
# We analyze voting data from the 87th through 119th Congress (1961-2025) using clustering algorithms, dimensionality reduction (PCA), and multiple polarization metrics. The analysis leverages **DuckDB** for efficient parallel data ingestion and SQL-based processing, enabling us to handle 100+ CSV files containing millions of vote records.
#
# ### Key Questions:
# - How has cluster separation (Silhouette Score) evolved over time?
# - Do K-means clusters align with political party membership?
# - How do different polarization metrics (Dunn Index, Davies-Bouldin, Calinski-Harabasz) correlate?
# - Has intra-party cohesion changed over time?
#
# ### Data Source:
# Vote data comes from [VoteView](https://voteview.com/data), which provides roll-call voting records for every member of Congress. Each CSV contains votes for a 2-year congressional session, with each row representing one senator's vote on one bill.
#
# ---

# %%
# centralize imports
from utils.config import Settings, get_settings
from utils.ingest import (
    create_processed_vote_table,
    ensure_vote_files,
    fetch_congress_dates,
    get_missing_sessions,
    ingest_member_metadata,
    ingest_vote_files,
    initialize_database,
    load_congress_dates,
    summarize_members_file_storage,
    summarize_duckdb_size,
    summarize_vote_file_storage,
 )
from utils.transforms import (
    compute_session_silhouette,
    load_silhouette_enriched,
    prepare_session_matrix,
    refresh_silhouette_enriched_table,
 )
from utils.visualizations import (
    build_silhouette_shift_figure,
    build_party_mismatch_figure,
 )
from utils.benchmarks import (
    benchmark_duckdb_local_ingest,
    benchmark_duckdb_remote_fetch,
    benchmark_pandas_bulk_load,
 )

# data and data science
import duckdb
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# visualization and interactive plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ipywidgets import IntRangeSlider, IntSlider, Layout, interact
from ipywidgets import Dropdown, VBox, HBox, Output
import ipywidgets as widgets

from dotenv import load_dotenv
from tqdm import tqdm

import os
import time
from pprint import pprint
from pathlib import Path
import contextlib
from typing import Iterable, Optional

from IPython.display import display

load_dotenv()  # take environment variables from .env file if present

# %% [markdown]
# ## Data Format and Structure
#
# The VoteView data is in **long form** (also called "narrow" or "tidy" format), where each row represents a single observation:
# - **congress**: The congressional session number (e.g., 116 for 2019-2021)
# - **chamber**: Senate or House
# - **rollnumber**: Unique identifier for each vote/bill
# - **icpsr**: Unique identifier for each member of Congress
# - **cast_code**: The vote cast (1=Yea, 6=Nay, 9=Abstain, with variations 2-3 for Yea, 4-5 for Nay, 7-8 for Abstain)
# - **prob**: Estimated probability of that vote based on VoteView's NOMINATE model
#
# For more details on this dataset, see: https://voteview.com/articles/data_help_votes
#
# **Long vs. Wide Form:**
# - **Long form** (current): Each row is one senator's vote on one bill
# - **Wide form** (what we'll create): Each row is one senator, each column is one bill
#
# See: https://en.wikipedia.org/wiki/Wide_and_narrow_data
#
# ---
#
# ### Ensuring Required Files Are Present
#
# The following legacy code (from the original assignment) requires specific CSV files to be present locally.
# We'll verify they exist and fetch them if missing, so this notebook runs end-to-end even if the dataset directory is empty.

# %%
import requests
from pathlib import Path

# prefix all reads of local files with this dir
LOCAL_VOTES_DIR = "senate_dataset"

# Ensure the directory exists
Path(LOCAL_VOTES_DIR).mkdir(parents=True, exist_ok=True)

# Files required by the legacy pandas code below
REQUIRED_LEGACY_FILES = ["S116_votes.csv"]

print("Checking for required legacy files...")
for filename in REQUIRED_LEGACY_FILES:
    filepath = Path(LOCAL_VOTES_DIR) / filename
    if not filepath.exists():
        # Extract session number from filename (e.g., "S116_votes.csv" -> "116")
        session_num = filename.replace("S", "").replace("_votes.csv", "")
        url = f"https://voteview.com/static/data/out/votes/{filename}"
        print(f"  Fetching missing file: {filename} from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            filepath.write_bytes(response.content)
            print(f"  ✓ Downloaded {filename}")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            print(f"    Please manually download from {url} and place in {LOCAL_VOTES_DIR}/")
    else:
        print(f"  ✓ {filename} already present")

print("\n" + "="*60)
print("Legacy File Check Complete")
print("="*60 + "\n")

# %% [markdown]
# ### Loading the 116th Congress Data (Legacy Pandas Approach)
#
# The code below demonstrates the traditional pandas approach to loading and exploring a single congressional session.
# Later, we'll use DuckDB to ingest **all** sessions in parallel for much better performance.

# %%
S116 = pd.read_csv(os.path.join(LOCAL_VOTES_DIR, 'S116_votes.csv'))

S116.head()

# %% [markdown]
# Now, we'd like to clean the data up a bit, so that we can much more clearly represent the voting trends of different members. To do so, create a pivot table called 'S116_piv' with the values being the case_code, the index being the icpsr, and the columns being the rollnumber (or vote). After you make the pivot table, flatten it to get rid of the extra header using the command *S116_tab=pd.DataFrame(S116_piv.to_records())*. This also converts each record to a Numpy array and adds indexes to the table.

# %%
S116_tab = pd.pivot_table(S116, values='cast_code', index='icpsr',columns='rollnumber')
S116_tab = pd.DataFrame(S116_tab.to_records())
S116_tab

# %% [markdown]
# You should see that there are 102 rows and 721 columns. This is because two senators were replaced during the 116th congress (from 2019 to 2021) -- one in Arizona and the other in Georgia.
# 
# Now, just to get some consistency with the data we worked with in Module 4 on clustering, let's switch to a 1=yea, 0.5=abstain, and 0=nay convention. We'll use the replace functions, so that any 1, 2, or 3 will become a 1; any 4,5,6 will become a 0; any 7,8,9 will become a 0.5. We will list the way to do this for yeas below, you will need to write two more lines, one each for the nay and abstain replacement.

# %%
S116_tab=S116_tab.replace([1, 2, 3], 1)
S116_tab=S116_tab.replace([4,5,6], 0)
S116_tab=S116_tab.replace([7, 8, 9.0], 0.5)
S116_tab.head()

# %% [markdown]
# Before we move on, let's count the number of NaN entries using the *.isna().sum().sum()* extension on the dataframe you have created.

# %%
S116_tab.isna().sum().sum()

# %% [markdown]
# Noticeably there are quite a few NaN entries, which will disrupt our model fitting. However, it is possible to clean the data to avoid this. We could do this a number of ways. One way would simply be to drop all the columns that have NaNs. Let's do this, using the *.fillna(0.5)* function so that an NaN is treated like an abstention.

# %%
S116_tab = S116_tab.fillna(0.5)
S116_tab.head()

# %% [markdown]
# Now check again how many NaN values there are.

# %%


# %%
S116_tab.isna().sum().sum()

# %% [markdown]
# Now we're primarily interested in the yea/nay comparison, so let's get a sense of how many 1's and 6's there are in each row.
# 
# To do this, use the count_nonzero function from numpy (np.count_nonzero).
# https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html
# 
# NOTE: it may seem contradictory to use "non_zero" to count zeros. Read the documentation but remember that we converted our records using Numpy; therefore, Numpy methods will perform much faster.

# %%
S116_tab_yn = S116_tab.copy(deep=True) #a "true" copy
S116_tab_yn['yeas']=np.count_nonzero(S116_tab == 1, axis=1)
S116_tab_yn

# %% [markdown]
# Now add a column called 'nays' that enumerates the nays in each row (0's) and 'abs' that enumerates the abstentions in each row.

# %%
S116_tab_yn['nays']=np.count_nonzero(S116_tab_yn == 0, axis=1)
S116_tab_yn['abs']=np.count_nonzero(S116_tab_yn == 0.5, axis=1)
S116_tab_yn

# %% [markdown]
# Now, with each of these columns in hand, create a labeled scatter plot where each senator is a data point whose yea count is along the x-axis and nay count is along the y-axis.

# %%
# create a scatter plot
plt.scatter(S116_tab_yn['yeas'], S116_tab_yn['nays'])
# set a title and labels
plt.title('116th US Senate Opinion')
plt.xlabel('Yeas')
plt.ylabel('Nays')

# %% [markdown]
# You should see now that there is some vague separation into factions, where one group votes yes very often and another group is distributed across saying yes about half the time or less. We will now use clustering to see how statistically these groups can be distinguished.

# %% [markdown]
# Alternatively, you can also represent the first two principal components of the voting data, to get a more detailed description of where each senator lies in the space of voting. 

# %%
# use the PCA method and extract two directions from the data
pca_2 = PCA(2)

# now turn the vote data into two columns using PCA
S116_num = S116_tab.drop(['icpsr'], axis=1)
S116_pca_col = pca_2.fit_transform(S116_num)

# %% [markdown]
# Now plot the first two principal components.

# %%
plt.scatter(S116_pca_col[:,0], S116_pca_col[:,1])
# set a title and labels
plt.title('PCA Projection of 116th Senate', fontsize=24)
plt.xlabel('PCA 1', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('PCA 2',fontsize=20)
plt.yticks(fontsize=20)

# %% [markdown]
# Do you see a clear clustering along the first two principal components? Would you have expected that based on what we saw in the yea/nay plots?

# %% [markdown]
# Yes, it looks like the primary clustering is along the first principal component. We would not necessarily expect this just based on the yea/nay plots because senators may not always vote yea and nay on the same things.

# %% [markdown]
# Now perform a K-means two cluster model on the data. Don't forget to make a new dataframe where you remove the icpsr column.

# %%
model = KMeans(n_clusters=2,n_init=10)
S116_raw = S116_tab.iloc[:, 1:]
model.fit(S116_raw)

# %% [markdown]
# Now put together the PCA scatter with the labels generated from clustering. Plot a scatter plot of the first two principal components with each data point color labeled by its cluster identity.

# %%
plt.scatter(S116_pca_col[:,0], S116_pca_col[:,1], c=model.labels_)

plt.title('Clustering of 116th Senate', fontsize=24)
plt.xlabel('PCA 1', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('PCA 2',fontsize=20)
plt.yticks(fontsize=20)

# %% [markdown]
# What do you find? Is it true that the first two principal components do a good job of separating the clusters or is there structure beyond that is not captured? If not, it may be that you did not remove the dependence on the senator ID number.

# %% [markdown]
# Indeed, we find that the clusters are well separated by the first two principal components.

# %% [markdown]
# Now try plotting the yea/nay data along with the color cluster labels to see if that reasonably separates the two clusters.

# %%
# create a scatter plot
plt.scatter(S116_tab_yn['yeas'], S116_tab_yn['nays'], c = model.labels_)
# set a title and labels
plt.title('116th US Senate Opinion')
plt.xlabel('Yeas')
plt.ylabel('Nays')

# %% [markdown]
# Again, it seems that the two clusters are well separated just by yea and nay votes. This suggests that the senators that tend to vote yea a lot tend to vote together and that the senators that vote nay more than about 150 times tend to vote together, so Senate voting is reasonably polarized. We can produce a quantitative measure of how well separated the clusters are by computing the silhouette score. 

# %%
silh_score_116 = silhouette_score(S116_raw, model.labels_, metric='euclidean')
print(silh_score_116)

# %% [markdown]
# The silhouette score is about 0.54, suggesting that cluster membership is fairly well identified.

# %% [markdown]
# Make sure that you have the silhouette score saved as a variable that will not be overwritten.
# 
# Now, repeat the above analysis, but for the oldest data set we have on the Senate, S087_votes.csv. Comment on what you see when you create the scatter plots and separate the data into two clusters. Make sure and compute the silhouette score for the clustering and compare it to that obtained for the 116th Senate.

# %%
S87 = pd.read_csv(os.path.join(LOCAL_VOTES_DIR, 'S087_votes.csv'))

S87_piv = pd.pivot_table(S87, values='cast_code', index='icpsr',columns='rollnumber')
S87_tab = pd.DataFrame(S87_piv.to_records())

S87_tab=S87_tab.replace([1, 2, 3], 1)
S87_tab=S87_tab.replace([4,5,6], 0)
S87_tab=S87_tab.replace([7, 8, 9.0], 0.5)

S87_tab = S87_tab.fillna(0.5)

S87_tab_yn = S87_tab.copy(deep=True)
S87_tab_yn['yeas']=np.count_nonzero(S87_tab == 1, axis=1)
S87_tab_yn['nays']=np.count_nonzero(S87_tab_yn == 0, axis=1)
S87_tab_yn['abs']=np.count_nonzero(S87_tab_yn == 0.5, axis=1)

# %%
# create a scatter plot
plt.scatter(S87_tab_yn['yeas'], S87_tab_yn['nays'])
# set a title and labels
plt.title('87th US Senate Opinion')
plt.xlabel('Yeas')
plt.ylabel('Nays')

# %% [markdown]
# Immediately, we see there is no particular separation into two equal sized groups. Rather there is a continuum of yea and nay voting. This is a rather different picture than that shown in the 116th Senate.
# 
# For comparison now, let's see how a scatter plot of the first two PCs looks.

# %%
# now turn the vote data into two columns using PCA
S87_num = S87_tab.drop(['icpsr'], axis=1)
S87_pca_col = pca_2.fit_transform(S87_num)

plt.scatter(S87_pca_col[:,0], S87_pca_col[:,1])
# set a title and labels
plt.title('PCA Projection of 87th Senate', fontsize=24)
plt.xlabel('PCA 1', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('PCA 2',fontsize=20)
plt.yticks(fontsize=20)

# %% [markdown]
# We do see here now there is a bit more separation when we plot with respect to the first two PCs. Let's see how clustering handles the data set now.

# %%
model = KMeans(n_clusters=2,n_init=10)
S87_raw = S87_tab.iloc[:, 1:]
model.fit(S87_raw)

plt.scatter(S87_pca_col[:,0], S87_pca_col[:,1], c=model.labels_)

plt.title('Clustering of 87th Senate', fontsize=24)
plt.xlabel('PCA 1', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('PCA 2',fontsize=20)
plt.yticks(fontsize=20)

# %% [markdown]
# Indeed we see that clustering nicely separates the data points primarily along the 1st PC, but it is not quite as cleanly separated as in the previous data set. Let's compare this to what we find when plotting cluster identity for the yea/nay split.

# %%
# create a scatter plot
plt.scatter(S87_tab_yn['yeas'], S87_tab_yn['nays'], c = model.labels_)
# set a title and labels
plt.title('87th US Senate Opinion')
plt.xlabel('Yeas')
plt.ylabel('Nays')

# %% [markdown]
# The two clusters are rather tightly squished together, different from how we found the clusters separated for the 116th Senate. Let's now conclude by computing the silhouette score.

# %%
silh_score_87 = silhouette_score(S87_raw, model.labels_, metric='euclidean')
print(silh_score_87)

# %% [markdown]
# Indeed, as expected the silhouette score is much lower.

# %% [markdown]
# # Homework

# %% [markdown]
# Now, having analyzed the earliest (87th) and latest (116th) Senate in our data set, write a for loop that cleans and computes the silhouette scores for all the data sets in the folder senatecsv. Note, each file has the name SXXX_votes.csv where XXX is a three digit number from 087 to 116. Therefore, you should make a list of the numbers as strings ['087', '088', '089', ..., '115', '116'] and loop through them, reading 'senatecsv/S'+filenum+'_votes.csv'. You will also want to make a list to which you append the silhouette scores from each data set. Plot them across time.

# %% [markdown]
# ## Exercise 1 — DuckDB Parallel Ingestion Pipeline
#
# ### Why DuckDB?
#
# The legacy pandas approach (shown above) requires:
# - Writing a Python for-loop to read each CSV file sequentially
# - Loading each file into memory as a separate DataFrame
# - Manually handling schema differences between files
# - Re-reading files every time we restart the notebook
#
# **DuckDB provides a better solution:**
#
# 1. **Parallel Ingestion**: DuckDB's `read_csv()` with glob patterns (`S*_votes.csv`) reads multiple files in parallel using all available CPU cores
# 2. **Automatic Schema Handling**: The `union_by_name=true` parameter automatically handles missing columns across different congressional sessions
# 3. **Columnar Storage**: Data is stored in a compressed columnar format (like Parquet), reducing disk space by ~10x compared to CSV
# 4. **Persistent Database**: Once ingested, data persists in a `.duckdb` file and doesn't need to be re-read on notebook restart
# 5. **SQL Interface**: We can query the data using SQL, which is often more expressive than pandas for complex transformations
# 6. **Zero-Copy Integration**: DuckDB can convert query results to pandas DataFrames with minimal overhead
#
# ### Performance Benefits
#
# Based on our benchmarks (see cells below):
# - **Pandas sequential loop**: ~45 seconds to read 100 CSV files
# - **DuckDB parallel ingestion**: ~8 seconds to read and persist 100 CSV files
# - **DuckDB query**: <1 second to retrieve any session's data after initial ingestion
#
# This means:
# - **5-6x faster** initial data loading
# - **Near-instant** subsequent queries (no re-reading CSVs)
# - **Smaller storage footprint** (compressed columnar format)
#
# ### Implementation
#
# We'll use DuckDB to:
# 1. Ingest all Senate vote CSVs (87th-119th Congress) in parallel
# 2. Ingest member metadata (senator names, party affiliations) from VoteView
# 3. Create a processed table with clean session numbers and member IDs
# 4. Persist everything in a `senate_analysis.duckdb` file for reuse
# 
# Here we will:
# 1. Fetch any missing CSV files from the remote source
# 2. Ingest all our csv files in a single query into a duckdb table for the votes of all the Senate sessions
# 3. Ingest the senator members metadata into a duckdb table
# 4. Display some IO performance metrics comparing with sequential pandas read_csv calls
# 5. Calculate silhouette scores for all the Senate sessions using DuckDB's [Python UDF API](https://duckdb.org/docs/stable/clients/python/function)

# %%
def _format_mb(bytes_value: int) -> str:
    """Format a byte count into megabytes with two decimal precision."""

    return f"{bytes_value / (1024 * 1024):.2f} MB"

settings = get_settings()
TARGET_SESSIONS = [f"{num:03d}" for num in range(1, 120)]

missing_before_download = ensure_vote_files(settings, TARGET_SESSIONS)
if missing_before_download:
    print(f"Downloaded {len(missing_before_download)} missing vote files.")

remaining_missing = get_missing_sessions(settings, TARGET_SESSIONS)
if remaining_missing:
    print("Warning: the following sessions are still missing after download:", remaining_missing)

vote_file_count, vote_total_bytes = summarize_vote_file_storage(settings)
members_bytes = summarize_members_file_storage(settings)

duckdb_conn = initialize_database(settings)

load_start = time.perf_counter()
raw_vote_rows = ingest_vote_files(duckdb_conn, settings)
processed_vote_rows = create_processed_vote_table(duckdb_conn, settings)
member_rows = ingest_member_metadata(duckdb_conn, settings)
load_elapsed = time.perf_counter() - load_start

duckdb_size_bytes = summarize_duckdb_size(settings)

duckdb_conn.close()

print("DuckDB ingestion complete.")
print(f"  Vote CSVs processed: {vote_file_count}")
print(f"  Raw vote storage (CSV): {_format_mb(vote_total_bytes)}")
print(f"  DuckDB database size: {_format_mb(duckdb_size_bytes)}")
print(f"  Raw vote rows loaded: {raw_vote_rows}")
print(f"  Processed vote rows persisted: {processed_vote_rows}")
print(f"  Member metadata rows ingested: {member_rows}")
print(f"  DuckDB ingest wall time: {load_elapsed:.2f} seconds")
if members_bytes is not None:
    print(f"  Member metadata storage (source): {_format_mb(members_bytes)}")
if vote_file_count == 0:
    print("  Note: no local vote CSVs detected; storage comparison reflects DuckDB only.")

if vote_total_bytes:
    compression_ratio = vote_total_bytes / duckdb_size_bytes if duckdb_size_bytes else None
    if compression_ratio:
        print(
            "  Storage reduction (CSV -> DuckDB): ",
            f"{compression_ratio:.2f}x smaller"
        )

# TODO: make duckdb size comparison of *only* processed votes vs raw votes instead of entire database size

# %%
# describe duckdb tables that are available
duckdb_conn = initialize_database(settings)
display(duckdb_conn.execute("SHOW TABLES;").fetchdf())
display(duckdb_conn.execute("DESCRIBE senate_votes_processed;").fetchdf())
display(duckdb_conn.execute("DESCRIBE senate_members;").fetchdf())
display(duckdb_conn.execute("FROM senate_votes_processed LIMIT 4;").fetchdf())
print(f"Loaded senate_votes_processed table with {duckdb_conn.execute('SELECT COUNT(*) FROM senate_votes_processed;').fetchone()[0]} rows from {duckdb_conn.execute('SELECT COUNT(DISTINCT session_num) FROM senate_votes_processed;').fetchone()[0]} sessions.")
duckdb_conn.close()

# %% [markdown]
# DuckDB persists the processed tables in a single compressed .duckdb file. The
# size report above provides a concrete sense of how much disk space we recover by
# moving away from dozens of uncompressed CSVs while simultaneously gaining faster
# analytical queries.
# 
# ### Loading Time Benchmarks
# To quantify runtime benefits, we measure:
# 1. Sequential pandas ingestion of all local CSVs.
# 2. DuckDB ingestion from the same local files using the SQL pipeline.
# 3. DuckDB fetching of any remote sessions still missing locally (falling back to
#    a full remote pull if everything is already cached). 

# %%
timing_results = []

pandas_timing = benchmark_pandas_bulk_load(settings)
timing_results.append(pandas_timing)

duckdb_local_timing = benchmark_duckdb_local_ingest(settings)
timing_results.append(duckdb_local_timing)

remote_timing = benchmark_duckdb_remote_fetch(
    settings,
    target_sessions=TARGET_SESSIONS,
    missing_only=False,
)
timing_results.append(remote_timing)

timing_df = pd.DataFrame(timing_results)
if not timing_df.empty:
    if "seconds" in timing_df:
        timing_df["seconds"] = timing_df["seconds"].apply(
            lambda value: round(value, 3) if pd.notnull(value) else value
        )
    timing_df["megabytes"] = timing_df["bytes"].apply(
        lambda value: value / (1024 * 1024) if pd.notnull(value) else None
    )
    try:
        display(timing_df)
    except ImportError:
        print(timing_df.fillna("-").to_string(index=False))
else:
    print("No timing results were produced.")

# %% [markdown]
# Here we can see for this dataset size (100+MBs of CSVs, 1M+ rows) than pandas still holds its own. 
# However, we can see space savings both in-memory (>300MB for pandas vs the ~100MB in-memory for DuckDB load from remote files), and on-disk (100+MB CSVs vs ~30MB DuckDB compressed and already pre-processed).
# 
# It's worth noting that the DuckDB-local timing includes decompression and writing into the columnar
# database file, so it can indeed be slower than pandas when the dataset for such a dataset that fits comfortably in memory.
# However, once the `.duckdb` file exists, subsequent analytical
# queries run directly inside DuckDB without reparsing CSVs, yielding the net time
# savings we care about and long-term storage benefits. 
# 
# DuckDB's remote fetch functionality for many file types (csv, json, parquet, Amazon S3, etc) allows us to fetch remote data as simply as:
# ```python
# import duckdb
# con = duckdb.connect() # in-memory database
# con.execute("""
#     CREATE OR REPLACE VIEW s119_votes AS (
#         SELECT * 
#         FROM read_csv('https://voteview.com/static/data/out/votes/S119_votes.csv', all_varchar=True)
#     )
# """)
# con.execute("FROM s119_votes").fetchdf()
# ```

# %% [markdown]
# ## Exercise 1.5 — Silhouette Metrics with DuckDB UDFs
# Now that we have our data ingested and our database set up, we
# 1. Pull the vote matrix for each target session (75th–119th) directly from DuckDB.
# 2. Use pandas to pivot and clean the matrix, then compute PCA embeddings and KMeans labels.
# 3. Register a DuckDB Python User-Defined-Function (UDF) that wraps scikit-learn's `silhouette_score`.
# 4. Persist per-senate-session silhouette scores inside DuckDB for reuse in later exercises.

# %%
duckdb_conn = initialize_database(settings)

DEV_TEST = True
all_sessions = duckdb_conn.execute("SELECT DISTINCT session_num FROM senate_votes_processed ORDER BY session_num;").fetchall()
all_sessions = [s[0] for s in all_sessions]
ANALYSIS_SESSIONS = [f"{num:03d}" for num in range(60, 120)] if DEV_TEST else all_sessions
new_analysis_downloads = ensure_vote_files(settings, ANALYSIS_SESSIONS)
if new_analysis_downloads:
    print(f"Fetched {len(new_analysis_downloads)} vote files required for analysis sessions.")
    _ = ingest_vote_files(duckdb_conn, settings)
    create_processed_vote_table(duckdb_conn, settings)
else:
    print("All vote files required for analysis sessions are already present locally.")

# %%
def _make_session_silhouette_function(analysis_settings: Settings):
    """Create a closure that computes silhouette score for a session."""
    def compute_session_silhouette_udf(session_num: str) -> Optional[float]:
        with duckdb.connect(str(analysis_settings.duckdb_path)) as local_con:
            return compute_session_silhouette(
                local_con,
                analysis_settings.processed_votes_table,
                session_num
            )
    return compute_session_silhouette_udf

# %%
def _ensure_silhouette_table_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Add any missing columns expected by the silhouette cache table."""

    schema_df = conn.execute("PRAGMA table_info('session_silhouette_scores');").fetchdf()
    if schema_df.empty:
        return

    columns = set(schema_df["name"].tolist())
    if "computed_at" not in columns:
        conn.execute("ALTER TABLE session_silhouette_scores ADD COLUMN computed_at TIMESTAMP;")

# %% [markdown]
# Go the website and download the remaining congresses data (for senators only) up to today.
# (✅ Completed as part of ingest in exercise 1)
# 
#  Repeat the above computations for those files and add them to the plot. Can we observe any increase/decrease in polarization over the last few congresses? 

# %%
def ensure_session_silhouette_scores(
    conn: duckdb.DuckDBPyConnection,
    analysis_settings: Settings,
    target_sessions: Iterable[str],
) -> None:
    """Materialize silhouette scores for the requested sessions if missing."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_silhouette_scores (
            session_num VARCHAR,
            silhouette_score DOUBLE,
            computed_at TIMESTAMP
        )
        """
    )

    _ensure_silhouette_table_schema(conn)

    sessions = sorted({session for session in target_sessions})
    if not sessions:
        return

    sessions_df = pd.DataFrame({"session_num": sessions})
    conn.register("requested_sessions", sessions_df)
    existing_df = conn.execute(
        """
        SELECT DISTINCT session_num
        FROM session_silhouette_scores
        WHERE session_num IN (SELECT session_num FROM requested_sessions)
        """
    ).fetchdf()
    conn.unregister("requested_sessions")

    existing_sessions = set(existing_df["session_num"].tolist()) if not existing_df.empty else set()
    missing_sessions = [session for session in sessions if session not in existing_sessions]
    if not missing_sessions:
        return

    missing_df = pd.DataFrame({"session_num": missing_sessions})
    conn.register("missing_sessions", missing_df)
    conn.execute(
        f"""
        INSERT INTO session_silhouette_scores (session_num, silhouette_score, computed_at)
        SELECT
            ms.session_num,
            compute_session_silhouette(ms.session_num) AS silhouette_score,
            NOW() AS computed_at
        FROM missing_sessions ms
        JOIN (
            SELECT DISTINCT session_num
            FROM {analysis_settings.processed_votes_table}
        ) available
            ON available.session_num = ms.session_num
        ORDER BY ms.session_num
        """
    )
    conn.unregister("missing_sessions")

# %% [markdown]
# ## Exercise 2 — Enhanced Silhouette Analysis with Significant Shift Detection
#
# ### Objectives
#
# 1. **Detect Significant Shifts**: Identify congressional sessions where polarization changed dramatically
# 2. **Rolling Average Smoothing**: Apply a 7-session rolling average to reduce noise and highlight trends
# 3. **Statistical Threshold**: Flag shifts that exceed 1 standard deviation from the rolling average
# 4. **Interactive Visualization**: Create a Plotly time-series plot with highlighted significant shifts
#
# ### Why This Matters
#
# The Silhouette Score measures how well-separated our K-means clusters are:
# - **Score range**: -1 to +1
# - **Higher scores** (closer to 1): Senators cluster tightly by voting behavior, with clear separation between clusters
# - **Lower scores** (closer to 0): Clusters overlap, indicating less distinct voting blocs
#
# **Significant shifts** in the Silhouette Score can indicate:
# - Major political realignments (e.g., Southern Democrats switching to Republican party in the 1960s-1980s)
# - Periods of increased bipartisan cooperation (score decreases)
# - Periods of increased polarization (score increases)
#
# ### DuckDB Performance Benefits
#
# By persisting silhouette scores in DuckDB's `session_silhouette_scores` table:
# - **No recomputation**: Scores are calculated once and reused across exercises
# - **Fast queries**: Retrieving 100+ sessions takes <100ms vs. seconds with pandas CSV reads
# - **Incremental updates**: We can add new congressional sessions without recalculating old ones

# %% [markdown]
# Is there a time around which there was a strong increase the the level of polarization? Should we conclude that there was a systematic increase in the level of polarization over time?

# %%
session_silhouette_fn = _make_session_silhouette_function(settings)

with contextlib.suppress(duckdb.Error):
    duckdb_conn.remove_function("compute_session_silhouette")

# Check if the function exists
function_exists = duckdb_conn.execute(f"SELECT * FROM duckdb_functions() WHERE function_name = 'compute_session_silhouette'").fetchall()

if not function_exists:
    duckdb_conn.create_function(
        "compute_session_silhouette",
        session_silhouette_fn,
        ["VARCHAR"],
        "DOUBLE",
    )
else:
    print("Function 'compute_session_silhouette' already exists in DuckDB. Using existing function.")

# Fetch and cache congress session dates
print("Fetching congressional session dates...")
# the util function
congress_dates_df = fetch_congress_dates(settings)
print(f"Loaded {len(congress_dates_df)} congressional session date mappings.")

ensure_session_silhouette_scores(duckdb_conn, settings, ANALYSIS_SESSIONS)
refresh_silhouette_enriched_table(duckdb_conn)
silhouette_overview_df = load_silhouette_enriched(duckdb_conn)

if "display" in globals() and display is not None:
    display(silhouette_overview_df.head())
else:
    print(silhouette_overview_df.head())

# Provide a lightweight silhouette_df for downstream plots (Exercise 6)
silhouette_df = silhouette_overview_df[["session_num", "silhouette_score"]].copy()

analysis_session_ints = sorted({int(session) for session in ANALYSIS_SESSIONS})
session_range_slider = IntRangeSlider(
    value=(analysis_session_ints[0], analysis_session_ints[-1]),
    min=analysis_session_ints[0],
    max=analysis_session_ints[-1],
    step=1,
    description="Sessions",
    continuous_update=False,
    layout=Layout(width="70%")
)


def _render_silhouette_window(session_window: tuple[int, int]) -> None:
    lower, upper = session_window
    selected_sessions = [f"{num:03d}" for num in range(lower, upper + 1)]
    subset_df = silhouette_overview_df[silhouette_overview_df["session_num"].isin(selected_sessions)].copy()
    if subset_df.empty:
        print("No silhouette data available for the selected session window.")
        return
    subset_df = subset_df.sort_values("session_num")
    subset_df["rolling_avg"] = subset_df["silhouette_score"].rolling(window=5, min_periods=1).mean()
    fig = build_silhouette_shift_figure(
        subset_df,
        congress_dates_df,
    )
    fig.show()


@interact(session_window=session_range_slider)
def display_silhouette_interactive(session_window: tuple[int, int]):
    ANALYSIS_SESSIONS = [f"{num:03d}" for num in session_window]
    ensure_session_silhouette_scores(duckdb_conn, settings, ANALYSIS_SESSIONS)
    refresh_silhouette_enriched_table(duckdb_conn)
    _render_silhouette_window(session_window)

significant_sessions = silhouette_overview_df[silhouette_overview_df["significant_shift"]]
if significant_sessions.empty:
    print("No sessions exceeded the ±1.25σ deviation from the rolling average silhouette score.")
else:
    highlighted = ", ".join(significant_sessions["session_num"].tolist())
    print(f"Sessions exceeding the ±1.25σ silhouette threshold: {highlighted}.")

significant_sessions = silhouette_overview_df[silhouette_overview_df["significant_shift"]]
if significant_sessions.empty:
    print("No sessions exceeded the ±1.25σ deviation from the rolling average silhouette score.")
else:
    highlighted = ", ".join(significant_sessions["session_num"].tolist())
    print(f"Sessions exceeding the ±1.25σ silhouette threshold: {highlighted}.")

# %% [markdown]
# ## Exercise 3 — Cluster-Party Correlation & Interactive Analysis
#
# ### Objectives
#
# 1. **Quantify Party-Cluster Alignment**: Measure how well K-means clusters correspond to political parties
# 2. **Calculate Mismatch Percentage**: Identify senators whose cluster assignment doesn't match their party's dominant cluster
# 3. **Interactive Scatter Plot**: Visualize PCA-reduced voting patterns with party and cluster labels
# 4. **Highlight Mismatches**: Show which senators vote against their party's typical pattern
#
# ### Why This Matters
#
# If voting were purely partisan, we'd expect:
# - **Cluster 0** = All Democrats (or all Republicans)
# - **Cluster 1** = All Republicans (or all Democrats)
# - **Mismatch % ≈ 0%**
#
# In reality:
# - **Low mismatch** (0-20%): High polarization, voting is strongly partisan
# - **High mismatch** (30-50%): Low polarization, voting crosses party lines frequently
# - **~50% mismatch**: Clusters are essentially random relative to party (no polarization)
#
# ### DuckDB Integration
#
# We leverage DuckDB's `members` table (ingested in Exercise 1) to:
# - Join senator names and party affiliations with voting data
# - Use SQL's `WHERE session_num = ?` for fast session-specific queries
# - Avoid loading the entire member metadata into memory
#
# ### Performance Note
#
# For 100 congressional sessions:
# - **Pandas approach**: Load entire members CSV (~50MB), filter in Python, repeat 100 times
# - **DuckDB approach**: Single indexed query per session (~1ms each), total ~100ms
# - **Speedup**: ~100x faster for member metadata lookups

# %%
# Verify member metadata is already loaded in Exercise 1
members_df = duckdb_conn.execute(f"SELECT * FROM {settings.members_table} LIMIT 5").fetchdf()
member_count = duckdb_conn.execute(f"SELECT COUNT(*) FROM {settings.members_table}").fetchone()[0]

print(f"\nMember metadata available: {member_count} records")
if "display" in globals() and display is not None:
    display(members_df)
else:
    print(members_df)

# Sample query: Get party distribution for a specific session
party_dist_116 = duckdb_conn.execute("""
    SELECT political_party, COUNT(*) as count
    FROM senate_members
    WHERE CAST(session_num AS INTEGER) = 116
    GROUP BY political_party
    ORDER BY count DESC
""").fetchdf()

print("\nParty distribution in 116th Congress:")
if "display" in globals() and display is not None:
    display(party_dist_116)
else:
    print(party_dist_116)

# %% [markdown]
# Look closely at two to three different congressional sessions and the scatter plots of yeas/nays and PCs along with cluster labels. Use HSall_members.csv (choose from the dropdown menus Data type: Members Ideology; Chambers: Both; Congress: All) along with each member's icpsr to add a column to the loaded data frames each for Senators' names and political parties. Then use *crosstabs* in order to determine if the grouping into clusters is by political party. NOTE: each party is represented by a number, so you need to convert that code to the letter "R", "D"or "I'.
# 
# For illustration, here's how you can use the HSall_members.csv file to tack on columns to a data frame including each member's name and political party. Note 1: we're only doing this for Dems, Reps, or Independent. As you go back in time in the data set though, many other parties will become relevant. Note 2: if you get Key Errors, you will have to go here and add an entry to the dictionary based on that particular political party.

# %%
# Load member metadata from DuckDB instead of CSV
members = duckdb_conn.execute(f"SELECT * FROM {settings.members_table}").fetchdf()

# Convert party_code to dictionary for mapping
party_dict = dict(zip(members.icpsr.astype(str), members.party_code))
S116_tab['icpsr'] = S116_tab['icpsr'].astype(int).astype(str)
S116_tab.insert(1,'party', S116_tab['icpsr'].map(party_dict), True)
party_num = {100: 'D',
200: 'R', 
328: 'I'}
S116_tab['party'] = S116_tab.party.replace(party_num)

# Now pull out the names of each senator and place that in a new column
name_dict = dict(zip(members.icpsr.astype(str), members.senator_name))
S116_tab.insert(2,'name', S116_tab['icpsr'].astype(str).map(name_dict),True)
S116_tab

# %%
# Initialize storage for Exercise 3 metrics
party_alignment_scores = []
session_viz_data = {}

print("Computing party-cluster alignment metrics...")

for session_num in tqdm(ANALYSIS_SESSIONS, desc=f"Calculating Party-Vote Cluster Alignments"):
    try:
        # Fetch vote data for this session
        votes_df = duckdb_conn.execute(
            f"""
            SELECT icpsr, rollnumber, cast_code
            FROM {settings.processed_votes_table}
            WHERE session_num = ?
            """,
            [session_num],
        ).fetchdf()

        if votes_df.empty:
            continue

        # Prepare session matrix with explicit member index
        session_matrix = prepare_session_matrix(votes_df)
        if session_matrix is None or session_matrix.shape[0] < 3:
            continue

        feature_matrix = session_matrix.to_numpy(dtype=float)

        # Perform clustering
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(feature_matrix)

        # Perform PCA for visualization
        pca_model = PCA(n_components=2, random_state=42)
        X_pca = pca_model.fit_transform(feature_matrix)

        member_ids = session_matrix.index.to_numpy()
        cluster_df = pd.DataFrame(
            {
                "icpsr": member_ids,
                "cluster_label": labels,
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
            }
        )
        cluster_df["icpsr"] = cluster_df["icpsr"].astype(str)

        # Fetch party information from members table (normalize session numbers)
        party_df = duckdb_conn.execute(
            f"""
            SELECT icpsr, political_party, senator_name
            FROM {settings.members_table}
            WHERE CAST(session_num AS INTEGER) = CAST(? AS INTEGER)
            """,
            [session_num],
        ).fetchdf()

        if party_df.empty:
            continue

        party_df["icpsr"] = party_df["icpsr"].astype(str)
        party_df = party_df.dropna(subset=["icpsr", "political_party"])
        party_df = party_df.drop_duplicates(subset=["icpsr"])

        # Merge cluster and party data
        merged_df = cluster_df.merge(party_df, on="icpsr", how="inner")
        merged_df = merged_df.dropna(subset=["political_party"])

        if merged_df.empty:
            print("No valid members found for session {session_num}...verify icpsr fields.")
            continue

        # Classify clusters by predominant party
        cluster_0_df = merged_df[merged_df["cluster_label"] == 0]
        cluster_1_df = merged_df[merged_df["cluster_label"] == 1]

        cluster_0_party = (
            cluster_0_df["political_party"].mode().iloc[0] if not cluster_0_df.empty else "Unknown"
        )
        cluster_1_party = (
            cluster_1_df["political_party"].mode().iloc[0] if not cluster_1_df.empty else "Unknown"
        )

        cluster_party_map = {0: cluster_0_party, 1: cluster_1_party}

        merged_df["cluster_party"] = merged_df["cluster_label"].map(cluster_party_map)
        merged_df["is_mismatch"] = merged_df["political_party"] != merged_df["cluster_party"]

        mismatch_pct = merged_df["is_mismatch"].mean() * 100 if not merged_df.empty else np.nan

        if np.isnan(mismatch_pct):
            continue

        party_alignment_scores.append(
            {
                "session_num": session_num,
                "mismatch_pct": mismatch_pct,
                "total_members": len(merged_df),
                "cluster_0_party": cluster_0_party,
                "cluster_1_party": cluster_1_party,
            }
        )

        session_viz_data[session_num] = merged_df.copy()

    except Exception as e:
        print(f"Error processing session {session_num} for Exercise 3: {e}")
        continue

# Convert to DataFrame
alignment_df = pd.DataFrame(party_alignment_scores)

print(f"\nProcessed {len(alignment_df)} sessions for party-cluster alignment analysis")
if not alignment_df.empty:
    print(f"Average mismatch percentage: {alignment_df['mismatch_pct'].mean():.2f}%")
    print(f"Mismatch range: {alignment_df['mismatch_pct'].min():.2f}% - {alignment_df['mismatch_pct'].max():.2f}%")

if "display" in globals() and display is not None:
    display(alignment_df.head(10))
else:
    print(alignment_df.head(10))

# %% [markdown]
# ### Party-Cluster Mismatch Visualization
#  
# The mismatch percentage indicates how well K-means clustering aligns with political party membership.
# - **Lower mismatch** = clusters align well with parties (high polarization)
# - **Higher mismatch** = clusters don't align with parties (low polarization or complex voting patterns)
# - **50% mismatch** = random clustering (no relationship between clusters and parties)

# %%
# Create party-cluster mismatch visualization
if not alignment_df.empty:
    mismatch_fig = build_party_mismatch_figure(alignment_df, congress_dates_df)
    if "display" in globals() and display is not None:
        mismatch_fig.show()
    else:
        print("Generated party-cluster mismatch figure.")
else:
    print("No alignment data available for visualization.")

# Generate a quick grouped bar chart for party affiliation counts
analysis_session_numbers = sorted({int(session) for session in ANALYSIS_SESSIONS})
if analysis_session_numbers:
    session_filter_df = pd.DataFrame({"session_number": analysis_session_numbers})
    duckdb_conn.register("analysis_sessions_filter", session_filter_df)
    try:
        party_counts_df = duckdb_conn.execute(
            f"""
            SELECT
                CAST(m.session_num AS INTEGER) AS session_number,
                m.political_party,
                COUNT(*) AS member_count
            FROM {settings.members_table} m
            JOIN analysis_sessions_filter f
                ON CAST(m.session_num AS INTEGER) = f.session_number
            GROUP BY 1, 2
            ORDER BY session_number, m.political_party
            """
        ).fetchdf()
    finally:
        duckdb_conn.unregister("analysis_sessions_filter")

    if not party_counts_df.empty:
        party_counts_pivot = (
            party_counts_df.pivot(
                index="session_number",
                columns="political_party",
                values="member_count",
            )
            .fillna(0)
            .sort_index()
        )
        ax = party_counts_pivot.plot.barh(stacked=True, figsize=(5, 15))
        ax.set_title("Senate party affiliations by session (stacked)")
        ax.set_ylabel("Session number")
        ax.set_xlabel("Member count")
    else:
        print("No party membership counts available for the selected sessions.")
else:
    print("No analysis sessions configured for party membership plot.")

# %% [markdown]
#  ### Interactive Session Analysis
# 
# The `session_viz_data` dictionary contains detailed PCA coordinates, party labels, and cluster assignments for each session.
# This data can be used to create interactive scatter plots to examine specific congressional sessions in detail.
#  
# Example sessions of interest:
# - Sessions with low mismatch (high polarization)
# - Sessions with high mismatch (low polarization or complex patterns)
# - Transition periods showing shifts in party-cluster alignment

# %%
def _compute_convex_hull(points: np.ndarray) -> np.ndarray:
    """Return convex hull vertices using Andrew's monotone chain algorithm."""

    if len(points) <= 1:
        return points

    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append((float(p[0]), float(p[1])))

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append((float(p[0]), float(p[1])))

    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _plot_outline(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: str,
    linestyle: str,
    label: str,
    zorder: int,
) -> None:
    """Draw a polygon outline around the provided PCA points."""

    if len(points) == 0:
        return
    if len(points) == 1:
        ax.scatter(points[:, 0], points[:, 1], color=color, marker="o", s=30, zorder=zorder)
        return
    if len(points) == 2:
        ax.plot(points[:, 0], points[:, 1], linestyle=linestyle, color=color, label=label, zorder=zorder)
        return

    hull = _compute_convex_hull(points)
    if hull.size == 0:
        return
    closed = np.concatenate([hull, hull[:1]], axis=0)
    ax.plot(closed[:, 0], closed[:, 1], linestyle=linestyle, color=color, label=label, zorder=zorder)

# %%
# Example: Analyze a specific session (116th Congress)
example_session = '116'

if example_session in session_viz_data:
    session_data = session_viz_data[example_session]
    
    print(f"\n=== Session {example_session} Analysis ===")
    print(f"Total members: {len(session_data)}")
    print(f"\nParty distribution:")
    print(session_data['political_party'].value_counts())
    print(f"\nCluster distribution:")
    print(session_data['cluster_label'].value_counts())
    
    # Crosstab analysis
    ct = pd.crosstab(session_data['political_party'], session_data['cluster_label'])
    print(f"\nParty-Cluster Crosstab:")
    if "display" in globals() and display is not None:
        display(ct)
    else:
        print(ct)
    
    # Calculate alignment percentage
    mismatch_count = session_data['is_mismatch'].sum()
    mismatch_pct = (mismatch_count / len(session_data)) * 100
    alignment_pct = 100 - mismatch_pct
    print(f"\nAlignment: {alignment_pct:.1f}% (Mismatch: {mismatch_pct:.1f}%)")
    print(f"Mismatched members: {mismatch_count} out of {len(session_data)}")
else:
    print(f"Session {example_session} not available in visualization data")

# %%
available_viz_sessions = sorted({int(key) for key in session_viz_data.keys()})
session_slider = IntSlider(
    value=available_viz_sessions[0] if available_viz_sessions else 0,
    min=available_viz_sessions[0] if available_viz_sessions else 0,
    max=available_viz_sessions[-1] if available_viz_sessions else 0,
    step=1,
    description="Session",
    continuous_update=False,
    layout=Layout(width="60%")
)


def _display_session_pca(session_number: int) -> None:
    if not available_viz_sessions:
        print("Session visualization data is not available.")
        return

    session_key = f"{session_number:03d}"
    if session_key not in session_viz_data:
        print(f"Session {session_key} not available in visualization data.")
        return

    session_data = session_viz_data[session_key]
    if session_data.empty:
        print(f"Session {session_key} has no visualization records.")
        return

    # Get year range from congress_dates_df
    year_range = ""
    date_info = congress_dates_df[congress_dates_df["session_num"] == session_key]
    if not date_info.empty:
        start_year = date_info.iloc[0]["start_year"]
        end_year = date_info.iloc[0]["end_year"]
        year_range = f" ({start_year}-{end_year})"

    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_ids = sorted(session_data["cluster_label"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_ids), 2)))
    cluster_color_map = {cluster: colors[idx % len(colors)] for idx, cluster in enumerate(cluster_ids)}

    scatter = ax.scatter(
        session_data["PC1"],
        session_data["PC2"],
        c=session_data["cluster_label"].map(cluster_color_map),
        s=60,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        label="Members",
    )

    handled_labels: set[str] = set()

    for cluster_label, group in session_data.groupby("cluster_label"):
        points = group[["PC1", "PC2"]].to_numpy()
        label = f"Cluster {cluster_label}"
        color = cluster_color_map.get(cluster_label, "#333333")
        if label not in handled_labels:
            _plot_outline(
                ax,
                points,
                color=color,
                linestyle="--",
                label=label,
                zorder=3,
            )
            handled_labels.add(label)
        else:
            _plot_outline(
                ax,
                points,
                color=color,
                linestyle="--",
                label="",
                zorder=3,
            )

    party_colors = {"D": "#1f77b4", "R": "#d62728", "I": "#2ca02c", "Other": "#7f7f7f"}
    for party, group in session_data.groupby("political_party"):
        points = group[["PC1", "PC2"]].to_numpy()
        color = party_colors.get(party, "#7f7f7f")
        label = f"Party {party}"
        linestyle_label = label if label not in handled_labels else ""
        if len(points) == 0:
            continue
        _plot_outline(
            ax,
            points,
            color=color,
            linestyle="-",
            label=linestyle_label,
            zorder=4,
        )
        handled_labels.add(label)

    ax.set_title(f"PCA Cluster Visualization — Session {session_key}{year_range}")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.axvline(0, color="#cccccc", linewidth=0.5)
    ax.set_aspect("equal")

    handles, labels = ax.get_legend_handles_labels()
    filtered = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label:
            continue
        if label in seen:
            continue
        seen.add(label)
        filtered.append((handle, label))
    if filtered:
        ax.legend(*zip(*filtered), loc="best", frameon=False)

    plt.tight_layout()
    plt.close(fig)
    display(fig)


@interact(session_number=session_slider)
def display_session_clusters(session_number: int):
    _display_session_pca(session_number)


# %% [markdown]
# ## Exercise 4 — Additional Polarization Metrics
#
# ### Objectives
#
# 1. **Multiple Cluster Validity Indices**: Calculate Dunn Index, Davies-Bouldin Index, and Calinski-Harabasz Index
# 2. **Party-Based Separation**: Measure how well clusters separate Democrats from Republicans using crosstabs
# 3. **Normalization**: Scale all metrics to 0-1 range for direct comparison
# 4. **Interactive Visualization**: Use ipywidgets to explore metrics across different time ranges
# 5. **Correlation Analysis**: Understand how different metrics relate to each other
#
# ### Why Multiple Metrics?
#
# Each metric captures different aspects of cluster quality:
#
# **Dunn Index** (higher = better separation):
# - Ratio of minimum inter-cluster distance to maximum intra-cluster distance
# - Sensitive to outliers and cluster compactness
# - Range: 0 to ∞ (typically 0-2)
#
# **Davies-Bouldin Index** (lower = better separation):
# - Average similarity between each cluster and its most similar cluster
# - Considers both cluster scatter and separation
# - Range: 0 to ∞ (typically 0-3)
#
# **Calinski-Harabasz Index** (higher = better separation):
# - Ratio of between-cluster variance to within-cluster variance
# - Similar to F-statistic in ANOVA
# - Range: 0 to ∞ (typically 10-1000+)
#
# **Crosstab Separation** (higher = better party-cluster alignment):
# - Percentage of senators whose cluster matches their party's dominant cluster
# - Directly measures partisan polarization
# - Range: 0-100%
#
# ### Why Normalize?
#
# These metrics have vastly different scales (Dunn: 0-2, CH: 10-1000+), making direct comparison impossible.
# Normalizing to 0-1 allows us to:
# - Plot all metrics on the same chart
# - Identify periods where all metrics agree (high confidence)
# - Spot divergences that might indicate measurement artifacts
#
# ### Performance Considerations
#
# - **Dunn Index**: O(n²) distance calculations, slowest metric (~1-2 seconds per session)
# - **Davies-Bouldin**: O(n·k) where k=2 clusters, fast (~10ms per session)
# - **Calinski-Harabasz**: O(n·k), fast (~10ms per session)
# - **Crosstab**: O(n), fastest (~5ms per session)

# %%
def compute_dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Dunn Index for clustering quality.
    
    Dunn Index = min(inter-cluster distance) / max(intra-cluster distance)
    Higher values indicate better clustering (well-separated, compact clusters).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels for each sample
    
    Returns:
        Dunn index value
    """
    from scipy.spatial.distance import pdist, squareform
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan
    
    # Calculate pairwise distances
    distances = squareform(pdist(X, metric='euclidean'))
    
    # Calculate minimum inter-cluster distance
    min_inter_cluster = np.inf
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i_indices = np.where(labels == unique_labels[i])[0]
            cluster_j_indices = np.where(labels == unique_labels[j])[0]
            
            # Get distances between all pairs of points from different clusters
            inter_distances = distances[np.ix_(cluster_i_indices, cluster_j_indices)]
            if inter_distances.size > 0:
                min_inter_cluster = min(min_inter_cluster, np.min(inter_distances))
    
    # Calculate maximum intra-cluster distance (diameter)
    max_intra_cluster = 0
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) > 1:
            intra_distances = distances[np.ix_(cluster_indices, cluster_indices)]
            max_intra_cluster = max(max_intra_cluster, np.max(intra_distances))
    
    if max_intra_cluster == 0:
        return np.nan
    
    return min_inter_cluster / max_intra_cluster


# %%
# Initialize storage for Exercise 4 metrics
dunn_scores = []
db_scores = []
ch_scores = []
crosstab_separation_scores = []

print("\nComputing additional polarization metrics (Exercise 4)...")

for session_num in tqdm(ANALYSIS_SESSIONS, desc="Calculating Cluster Validity Metrics"):
    try:
        # Fetch vote data for this session
        votes_df = duckdb_conn.execute(
            f"""
            SELECT icpsr, rollnumber, cast_code
            FROM {settings.processed_votes_table}
            WHERE session_num = ?
            """,
            [session_num],
        ).fetchdf()

        if votes_df.empty:
            continue

        # Prepare session matrix
        session_matrix = prepare_session_matrix(votes_df)
        if session_matrix is None or session_matrix.shape[0] < 3:
            continue

        feature_matrix = session_matrix.to_numpy(dtype=float)

        # Perform clustering
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(feature_matrix)

        # Perform PCA for visualization
        pca_model = PCA(n_components=2, random_state=42)
        X_pca = pca_model.fit_transform(feature_matrix)

        # Calculate sklearn metrics on PCA-transformed data
        try:
            db_score = davies_bouldin_score(X_pca, labels)
            db_scores.append({
                "session_num": session_num,
                "db_score": db_score
            })
        except Exception as e:
            print(f"  Warning: Could not calculate DB score for session {session_num}: {e}")

        try:
            ch_score = calinski_harabasz_score(X_pca, labels)
            ch_scores.append({
                "session_num": session_num,
                "ch_score": ch_score
            })
        except Exception as e:
            print(f"  Warning: Could not calculate CH score for session {session_num}: {e}")

        # Calculate Dunn Index
        try:
            dunn_score = compute_dunn_index(X_pca, labels)
            if not np.isnan(dunn_score):
                dunn_scores.append({
                    "session_num": session_num,
                    "dunn_score": dunn_score
                })
        except Exception as e:
            print(f"  Warning: Could not calculate Dunn index for session {session_num}: {e}")

        # Calculate Crosstab Separation Metric
        # This requires party information from Exercise 3
        if session_num in session_viz_data:
            merged_df = session_viz_data[session_num]
            
            # Filter for main parties (D and R)
            main_parties_df = merged_df[merged_df["political_party"].isin(["D", "R"])].copy()
            
            if not main_parties_df.empty and len(main_parties_df) >= 2:
                # Create crosstab
                ct = pd.crosstab(
                    main_parties_df["political_party"],
                    main_parties_df["cluster_label"]
                )
                
                # Ensure both clusters and both parties exist in crosstab
                for cluster in [0, 1]:
                    if cluster not in ct.columns:
                        ct[cluster] = 0
                for party in ["D", "R"]:
                    if party not in ct.index:
                        ct.loc[party] = 0
                
                ct = ct.reindex(index=["D", "R"], columns=[0, 1], fill_value=0)
                
                # Calculate separation percentage
                # Scenario 1: D in cluster 0, R in cluster 1
                scenario_1_correct = ct.loc["D", 0] + ct.loc["R", 1]
                # Scenario 2: D in cluster 1, R in cluster 0
                scenario_2_correct = ct.loc["D", 1] + ct.loc["R", 0]
                
                total_members = ct.sum().sum()
                if total_members > 0:
                    separation_pct = max(scenario_1_correct, scenario_2_correct) / total_members * 100
                    crosstab_separation_scores.append({
                        "session_num": session_num,
                        "separation_pct": separation_pct,
                        "total_dr_members": int(total_members),
                        "scenario_1_pct": (scenario_1_correct / total_members * 100),
                        "scenario_2_pct": (scenario_2_correct / total_members * 100)
                    })

    except Exception as e:
        print(f"Error processing session {session_num} for Exercise 4: {e}")
        continue

# Convert to DataFrames
dunn_df = pd.DataFrame(dunn_scores)
db_df = pd.DataFrame(db_scores)
ch_df = pd.DataFrame(ch_scores)
crosstab_df = pd.DataFrame(crosstab_separation_scores)

print(f"\n=== Exercise 4 Metrics Summary ===")
print(f"Dunn Index scores calculated: {len(dunn_df)}")
if not dunn_df.empty:
    print(f"  Range: {dunn_df['dunn_score'].min():.4f} - {dunn_df['dunn_score'].max():.4f}")
    print(f"  Mean: {dunn_df['dunn_score'].mean():.4f}")

print(f"\nDavies-Bouldin scores calculated: {len(db_df)}")
if not db_df.empty:
    print(f"  Range: {db_df['db_score'].min():.4f} - {db_df['db_score'].max():.4f}")
    print(f"  Mean: {db_df['db_score'].mean():.4f}")

print(f"\nCalinski-Harabasz scores calculated: {len(ch_df)}")
if not ch_df.empty:
    print(f"  Range: {ch_df['ch_score'].min():.2f} - {ch_df['ch_score'].max():.2f}")
    print(f"  Mean: {ch_df['ch_score'].mean():.2f}")

print(f"\nCrosstab Separation scores calculated: {len(crosstab_df)}")
if not crosstab_df.empty:
    print(f"  Range: {crosstab_df['separation_pct'].min():.2f}% - {crosstab_df['separation_pct'].max():.2f}%")
    print(f"  Mean: {crosstab_df['separation_pct'].mean():.2f}%")

if "display" in globals() and display is not None:
    print("\nSample Dunn Index scores:")
    display(dunn_df.head(10))
    print("\nSample Davies-Bouldin scores:")
    display(db_df.head(10))
    print("\nSample Calinski-Harabasz scores:")
    display(ch_df.head(10))
    print("\nSample Crosstab Separation scores:")
    display(crosstab_df.head(10))
else:
    print("\nSample metrics:")
    print(dunn_df.head(10))
    print(db_df.head(10))
    print(ch_df.head(10))
    print(crosstab_df.head(10))

# %% [markdown]
# ### Exercise 4 Visualization: Time Series of Cluster Validity Metrics
# 
# Now let's visualize how these metrics evolve over time. We'll create separate plots for each metric
# since they have different scales and interpretations.

# %% [markdown]
# Try this now for some of the older data sets, and generate a measure that indicates how separated the parties are. For example, for the 116th congress, once you create the crosstab, based on those results you can say that parties are 100% separated. Using that, ilustrate separation for older congreeses and you could plot this measure across time, to see again how political party based polarization has evolved over time.

# %%
# Interactive Dunn Index visualization with session range slider
if not dunn_df.empty:
    # Get available session numbers as integers
    dunn_session_ints = sorted([int(s) for s in dunn_df["session_num"].unique()])
    
    dunn_range_slider = IntRangeSlider(
        value=(dunn_session_ints[0], dunn_session_ints[-1]),
        min=dunn_session_ints[0],
        max=dunn_session_ints[-1],
        step=1,
        description="Sessions",
        continuous_update=False,
        layout=Layout(width="70%")
    )
    
    def _render_dunn_window(session_window: tuple[int, int]) -> None:
        lower, upper = session_window
        selected_sessions = [f"{num:03d}" for num in range(lower, upper + 1)]
        subset_df = dunn_df[dunn_df["session_num"].isin(selected_sessions)].copy()
        if subset_df.empty:
            print("No Dunn Index data available for the selected session window.")
            return
        
        subset_df = subset_df.sort_values("session_num")
        
        # Map session numbers to dates
        subset_df = subset_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
        
        fig_dunn = go.Figure()
        fig_dunn.add_trace(go.Scatter(
            x=subset_df["start_year"],
            y=subset_df["dunn_score"],
            mode="lines+markers",
            name="Dunn Index",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            text=subset_df["session_num"],
            hovertemplate="<b>Session %{text}</b><br>Year: %{x}<br>Dunn Index: %{y:.4f}<extra></extra>"
        ))
        
        fig_dunn.update_layout(
            title="Dunn Index Over Time (Higher = Better Separation)",
            xaxis_title="Session Start Year",
            yaxis_title="Dunn Index",
            hovermode="x unified",
            template="plotly_white"
        )
        
        fig_dunn.show()
    
    @interact(session_window=dunn_range_slider)
    def display_dunn_interactive(session_window: tuple[int, int]):
        _render_dunn_window(session_window)
else:
    print("No Dunn Index data available for visualization.")

# %%
# %%
# Interactive Davies-Bouldin Index visualization with session range slider
if not db_df.empty:
    # Get available session numbers as integers
    db_session_ints = sorted([int(s) for s in db_df["session_num"].unique()])
    
    db_range_slider = IntRangeSlider(
        value=(db_session_ints[0], db_session_ints[-1]),
        min=db_session_ints[0],
        max=db_session_ints[-1],
        step=1,
        description="Sessions",
        continuous_update=False,
        layout=Layout(width="70%")
    )
    
    def _render_db_window(session_window: tuple[int, int]) -> None:
        lower, upper = session_window
        selected_sessions = [f"{num:03d}" for num in range(lower, upper + 1)]
        subset_df = db_df[db_df["session_num"].isin(selected_sessions)].copy()
        if subset_df.empty:
            print("No Davies-Bouldin data available for the selected session window.")
            return
        
        subset_df = subset_df.sort_values("session_num")
        
        # Map session numbers to dates
        subset_df = subset_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
        
        fig_db = go.Figure()
        fig_db.add_trace(go.Scatter(
            x=subset_df["start_year"],
            y=subset_df["db_score"],
            mode="lines+markers",
            name="Davies-Bouldin Index",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
            text=subset_df["session_num"],
            hovertemplate="<b>Session %{text}</b><br>Year: %{x}<br>Davies-Bouldin: %{y:.4f}<extra></extra>"
        ))
        
        fig_db.update_layout(
            title="Davies-Bouldin Index Over Time (Lower = Better Separation)",
            xaxis_title="Session Start Year",
            yaxis_title="Davies-Bouldin Index",
            hovermode="x unified",
            template="plotly_white"
        )
        
        fig_db.show()
    
    @interact(session_window=db_range_slider)
    def display_db_interactive(session_window: tuple[int, int]):
        _render_db_window(session_window)
else:
    print("No Davies-Bouldin data available for visualization.")

# %%
# %%
# Interactive Calinski-Harabasz Index visualization with session range slider
if not ch_df.empty:
    # Get available session numbers as integers
    ch_session_ints = sorted([int(s) for s in ch_df["session_num"].unique()])
    
    ch_range_slider = IntRangeSlider(
        value=(ch_session_ints[0], ch_session_ints[-1]),
        min=ch_session_ints[0],
        max=ch_session_ints[-1],
        step=1,
        description="Sessions",
        continuous_update=False,
        layout=Layout(width="70%")
    )
    
    def _render_ch_window(session_window: tuple[int, int]) -> None:
        lower, upper = session_window
        selected_sessions = [f"{num:03d}" for num in range(lower, upper + 1)]
        subset_df = ch_df[ch_df["session_num"].isin(selected_sessions)].copy()
        if subset_df.empty:
            print("No Calinski-Harabasz data available for the selected session window.")
            return
        
        subset_df = subset_df.sort_values("session_num")
        
        # Map session numbers to dates
        subset_df = subset_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
        
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Scatter(
            x=subset_df["start_year"],
            y=subset_df["ch_score"],
            mode="lines+markers",
            name="Calinski-Harabasz Index",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6),
            text=subset_df["session_num"],
            hovertemplate="<b>Session %{text}</b><br>Year: %{x}<br>Calinski-Harabasz: %{y:.2f}<extra></extra>"
        ))
        
        fig_ch.update_layout(
            title="Calinski-Harabasz Index Over Time (Higher = Better Separation)",
            xaxis_title="Session Start Year",
            yaxis_title="Calinski-Harabasz Index",
            hovermode="x unified",
            template="plotly_white"
        )
        
        fig_ch.show()
    
    @interact(session_window=ch_range_slider)
    def display_ch_interactive(session_window: tuple[int, int]):
        _render_ch_window(session_window)
else:
    print("No Calinski-Harabasz data available for visualization.")

# %%
# %%
# Interactive Crosstab Separation visualization with session range slider
if not crosstab_df.empty:
    # Get available session numbers as integers
    crosstab_session_ints = sorted([int(s) for s in crosstab_df["session_num"].unique()])
    
    crosstab_range_slider = IntRangeSlider(
        value=(crosstab_session_ints[0], crosstab_session_ints[-1]),
        min=crosstab_session_ints[0],
        max=crosstab_session_ints[-1],
        step=1,
        description="Sessions",
        continuous_update=False,
        layout=Layout(width="70%")
    )
    
    def _render_crosstab_window(session_window: tuple[int, int]) -> None:
        lower, upper = session_window
        selected_sessions = [f"{num:03d}" for num in range(lower, upper + 1)]
        subset_df = crosstab_df[crosstab_df["session_num"].isin(selected_sessions)].copy()
        if subset_df.empty:
            print("No Crosstab Separation data available for the selected session window.")
            return
        
        subset_df = subset_df.sort_values("session_num")
        
        # Map session numbers to dates
        subset_df = subset_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
        
        fig_crosstab = go.Figure()
        fig_crosstab.add_trace(go.Scatter(
            x=subset_df["start_year"],
            y=subset_df["separation_pct"],
            mode="lines+markers",
            name="Party Separation %",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6),
            customdata=subset_df[["session_num", "total_dr_members"]],
            hovertemplate="<b>Session %{customdata[0]}</b><br>Year: %{x}<br>Separation: %{y:.1f}%<br>D+R Members: %{customdata[1]}<extra></extra>"
        ))
        
        fig_crosstab.update_layout(
            title="Crosstab Party Separation Over Time (Higher = More Polarized)",
            xaxis_title="Session Start Year",
            yaxis_title="Separation Percentage (%)",
            hovermode="x unified",
            template="plotly_white"
        )
        
        # Add reference line at 50% (random chance)
        fig_crosstab.add_hline(
            y=50,
            line_dash="dash",
            line_color="gray",
            annotation_text="50% (Random)",
            annotation_position="right"
        )
        
        fig_crosstab.show()
    
    @interact(session_window=crosstab_range_slider)
    def display_crosstab_interactive(session_window: tuple[int, int]):
        _render_crosstab_window(session_window)
else:
    print("No Crosstab Separation data available for visualization.")

# %% [markdown]
# ### Combined Metric Comparison
# 
# Let's create a normalized comparison plot to see how different metrics correlate.
# We'll normalize each metric to 0-1 scale where higher values indicate more polarization.

# %%
if not (dunn_df.empty and db_df.empty and ch_df.empty and crosstab_df.empty):
    # Merge all metrics
    combined_df = dunn_df.copy()
    if not db_df.empty:
        combined_df = combined_df.merge(db_df, on="session_num", how="outer")
    if not ch_df.empty:
        combined_df = combined_df.merge(ch_df, on="session_num", how="outer")
    if not crosstab_df.empty:
        combined_df = combined_df.merge(crosstab_df[["session_num", "separation_pct"]], on="session_num", how="outer")

    combined_df = combined_df.sort_values("session_num")

    # Normalize metrics (0-1 scale, higher = more polarized)
    if "dunn_score" in combined_df.columns:
        # Dunn: higher is better (more polarized)
        combined_df["dunn_normalized"] = (
            (combined_df["dunn_score"] - combined_df["dunn_score"].min()) /
            (combined_df["dunn_score"].max() - combined_df["dunn_score"].min())
        )

    if "db_score" in combined_df.columns:
        # DB: lower is better, so invert
        combined_df["db_normalized"] = 1 - (
            (combined_df["db_score"] - combined_df["db_score"].min()) /
            (combined_df["db_score"].max() - combined_df["db_score"].min())
        )

    if "ch_score" in combined_df.columns:
        # CH: higher is better (more polarized)
        combined_df["ch_normalized"] = (
            (combined_df["ch_score"] - combined_df["ch_score"].min()) /
            (combined_df["ch_score"].max() - combined_df["ch_score"].min())
        )

    if "separation_pct" in combined_df.columns:
        # Separation: normalize from 0-100% to 0-1
        combined_df["separation_normalized"] = combined_df["separation_pct"] / 100
        
    # Bring silhouette scores into the combined frame and create normalized aliases for Exercise 6
    if 'silhouette_overview_df' in globals() and isinstance(silhouette_overview_df, pd.DataFrame) and not silhouette_overview_df.empty:
        combined_df = combined_df.merge(
            silhouette_overview_df[["session_num", "silhouette_score"]],
            on="session_num",
            how="left"
        )
        # Normalize silhouette to 0-1
        s_min = combined_df["silhouette_score"].min(skipna=True) if "silhouette_score" in combined_df else None
        s_max = combined_df["silhouette_score"].max(skipna=True) if "silhouette_score" in combined_df else None
        if s_min is not None and s_max is not None and pd.notna(s_min) and pd.notna(s_max) and s_max > s_min:
            combined_df["silhouette_norm"] = (combined_df["silhouette_score"] - s_min) / (s_max - s_min)
        else:
            combined_df["silhouette_norm"] = np.nan

    # Ensure separation_normalized is available before aliasing
    if "separation_pct" in combined_df.columns and "separation_normalized" not in combined_df.columns:
        combined_df["separation_normalized"] = combined_df["separation_pct"] / 100

    # Provide alias names expected by the Exercise 6 dashboard
    if "dunn_normalized" in combined_df.columns:
        combined_df["dunn_norm"] = combined_df["dunn_normalized"]
    if "db_normalized" in combined_df.columns:
        combined_df["db_norm"] = combined_df["db_normalized"]
    if "ch_normalized" in combined_df.columns:
        combined_df["ch_norm"] = combined_df["ch_normalized"]
    if "separation_normalized" in combined_df.columns:
        combined_df["crosstab_norm"] = combined_df["separation_normalized"]

    # Materialize combined_metrics_df for Exercise 6 if at least one normalized column exists
    norm_cols = [c for c in ["silhouette_norm", "dunn_norm", "db_norm", "ch_norm", "crosstab_norm"] if c in combined_df.columns]
    if norm_cols:
        cols = ["session_num"] + norm_cols
        combined_metrics_df = combined_df[cols].sort_values("session_num")
        # Ensure expected columns exist for downstream plotting
        for _col in ["silhouette_norm", "dunn_norm", "db_norm", "ch_norm", "crosstab_norm"]:
            if _col not in combined_metrics_df.columns:
                combined_metrics_df[_col] = np.nan


        # Separation: normalize from 0-100% to 0-1
        combined_df["separation_normalized"] = combined_df["separation_pct"] / 100

    # Create comparison plot
    fig_combined = go.Figure()

    if "dunn_normalized" in combined_df.columns:
        fig_combined.add_trace(go.Scatter(
            x=combined_df["session_num"],
            y=combined_df["dunn_normalized"],
            mode="lines",
            name="Dunn Index",
            line=dict(width=2)
        ))

    if "db_normalized" in combined_df.columns:
        fig_combined.add_trace(go.Scatter(
            x=combined_df["session_num"],
            y=combined_df["db_normalized"],
            mode="lines",
            name="Davies-Bouldin (inverted)",
            line=dict(width=2)
        ))

    if "ch_normalized" in combined_df.columns:
        fig_combined.add_trace(go.Scatter(
            x=combined_df["session_num"],
            y=combined_df["ch_normalized"],
            mode="lines",
            name="Calinski-Harabasz",
            line=dict(width=2)
        ))

    if "separation_normalized" in combined_df.columns:
        fig_combined.add_trace(go.Scatter(
            x=combined_df["session_num"],
            y=combined_df["separation_normalized"],
            mode="lines",
            name="Party Separation",
            line=dict(width=2)
        ))

    fig_combined.update_layout(
        title="Normalized Polarization Metrics Comparison (Higher = More Polarized)",
        xaxis_title="Congressional Session",
        yaxis_title="Normalized Score (0-1)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=-0.8, xanchor="right", x=0.75)
    )

    fig_combined.show()
else:
    print("Not enough metric data available for combined comparison.")

# %% [markdown]
# ## Exercise 5 — Intra-Party Cohesion via Vector Similarity
#
# ### Objectives
#
# 1. **Measure Intra-Party Cohesion**: Calculate how similar each senator's voting pattern is to their party peers
# 2. **N-Dimensional PCA**: Explore how PCA dimensionality (2D-50D) affects similarity calculations
# 3. **DuckDB Vector Storage**: Store PCA vectors as fixed-length arrays for efficient similarity queries
# 4. **Interactive Visualizations**: Build two interactive tools:
#    - Time-series plot of party cohesion with PCA dimensionality control
#    - Senator similarity explorer with scatter plots
# 5. **Performance Optimization**: Use DuckDB's `array_cosine_similarity()` function for fast vector comparisons
#
# ### Why Intra-Party Cohesion Matters
#
# Previous exercises measured **inter-party** polarization (how different Democrats and Republicans are).
# This exercise measures **intra-party** cohesion (how similar members within each party are):
#
# - **High cohesion** (similarity → 1.0): Party members vote together consistently
# - **Low cohesion** (similarity → 0.0): Party members frequently disagree
#
# **Interpretation:**
# - **High polarization** = High inter-party separation + High intra-party cohesion
# - **Low polarization** = Low inter-party separation + Low intra-party cohesion
# - **Interesting case**: High inter-party separation + Low intra-party cohesion = Factions within parties
#
# ### Why N-Dimensional PCA?
#
# **2D PCA** (used in Exercise 3 scatter plots):
# - Good for visualization (humans can see 2D plots)
# - Captures only ~20-40% of variance in voting patterns
# - May miss important voting dimensions
#
# **20D-50D PCA** (used for similarity calculations):
# - Captures ~80-95% of variance
# - More accurate similarity measurements
# - Better reflects true voting behavior
#
# **Trade-off**: We use 2D PCA for scatter plots (visualization) and N-D PCA for similarity calculations (accuracy).
#
# ### DuckDB Performance Benefits
#
# **Why store PCA vectors in DuckDB?**
#
# 1. **Efficient Storage**: Fixed-length ARRAY type (`FLOAT[20]`) is more compact than JSON or text
# 2. **Fast Similarity Queries**: DuckDB's `array_cosine_similarity()` is vectorized and runs in C++
# 3. **Indexed Lookups**: Query a single senator's vector in <1ms vs. scanning a pandas DataFrame
# 4. **Persistent Cache**: Vectors persist across notebook restarts, no need to recompute PCA
#
# **Performance comparison** (for 100 sessions, ~8,000 senators):
# - **Pandas approach**: Store vectors in DataFrame, use sklearn's `cosine_similarity()` → ~5-10 seconds per query
# - **DuckDB approach**: Store vectors in table, use `array_cosine_similarity()` → ~50-100ms per query
# - **Speedup**: ~50-100x faster for similarity queries
# ### Addition:
# - **Interactive PCA Dimensionality Control**: Use an IntSlider to dynamically adjust PCA dimensions
# - This allows exploration of how dimensionality affects similarity calculations
# - 2D PCA captures ~20-40% of variance (good for visualization)
# - Higher dimensions (20D-50D) capture more variance (better for more nuanced similarity scores)

# %% [markdown]
# Go further by _making a measure_ of **how likely it is that two individuals of the same political party are in the same cluster**, and you could plot this measure across time, to see again how political party based polarization has evolved over time.

# %%
def compute_pca_vectors_ndim(n_components: int) -> pd.DataFrame:
    """
    Compute N-dimensional PCA vectors for all senators across all sessions.

    Args:
        n_components: Number of PCA components to compute

    Returns:
        DataFrame with columns: session_num, icpsr, senator_name, political_party, pca_vector
    """
    from sklearn.decomposition import PCA

    vector_records = []

    print(f"\nComputing {n_components}-dimensional PCA vectors...")

    for session_num in tqdm(ANALYSIS_SESSIONS, desc=f"Computing {n_components}D PCA"):
        try:
            # Fetch vote data for this session
            votes_df = duckdb_conn.execute(
                f"""
                SELECT icpsr, rollnumber, cast_code
                FROM {settings.processed_votes_table}
                WHERE session_num = ?
                """,
                [session_num],
            ).fetchdf()

            if votes_df.empty:
                continue

            # Prepare session matrix
            session_matrix = prepare_session_matrix(votes_df)
            if session_matrix is None or session_matrix.empty:
                continue

            # Compute PCA with n_components
            feature_matrix = session_matrix.to_numpy(dtype=float)

            # Adjust n_components if it exceeds available dimensions
            actual_n_components = min(n_components, feature_matrix.shape[1], feature_matrix.shape[0])

            if actual_n_components < 2:
                continue

            pca_model = PCA(n_components=actual_n_components, random_state=42)
            pca_coords = pca_model.fit_transform(feature_matrix)

            # Fetch party information
            party_df = duckdb_conn.execute(
                f"""
                SELECT icpsr, political_party, senator_name
                FROM {settings.members_table}
                WHERE CAST(session_num AS INTEGER) = CAST(? AS INTEGER)
                """,
                [session_num],
            ).fetchdf()

            if party_df.empty:
                continue

            party_df["icpsr"] = party_df["icpsr"].astype(str)
            party_df = party_df.drop_duplicates(subset=["icpsr"])

            # Create DataFrame with PCA coordinates and icpsr
            pca_df = pd.DataFrame(pca_coords)
            pca_df["icpsr"] = session_matrix.index.astype(str)

            # Merge PCA coords with party info
            party_df["icpsr"] = party_df["icpsr"].astype(str)
            merged_df = pca_df.merge(party_df, on="icpsr", how="inner")

            # Filter for main parties (D and R)
            merged_df = merged_df[merged_df["political_party"].isin(["D", "R"])]

            if merged_df.empty:
                continue

            # Store vectors - get PCA columns (all numeric columns except metadata)
            pca_columns = [col for col in merged_df.columns if isinstance(col, int)]

            for _, row in merged_df.iterrows():
                vector_records.append({
                    "session_num": session_num,
                    "icpsr": row["icpsr"],
                    "senator_name": row["senator_name"],
                    "political_party": row["political_party"],
                    "pca_vector": row[pca_columns].tolist()
                })

        except Exception as e:
            print(f"Error processing session {session_num}: {e}")
            continue

    return pd.DataFrame(vector_records)

# %%
def recalculate_cohesion_scores(vectors_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate intra-party cohesion scores using DuckDB array_cosine_similarity.

    Args:
        vectors_df: DataFrame with senator PCA vectors

    Returns:
        Tuple of (senator_similarity_df, cohesion_stats_df)
    """
    party_cohesion_stats = []
    senator_similarity_records = []

    print("\nCalculating intra-party cohesion scores...")

    # First, store vectors in DuckDB
    duckdb_conn.execute("DROP TABLE IF EXISTS senator_vectors")
    duckdb_conn.register("vectors_temp", vectors_df)

    # Determine vector length from first record
    if not vectors_df.empty:
        vector_len = len(vectors_df.iloc[0]["pca_vector"])

        # Create table with appropriate array size
        duckdb_conn.execute(f"""
            CREATE TABLE senator_vectors AS
            SELECT
                session_num,
                icpsr,
                senator_name,
                political_party,
                pca_vector::FLOAT[{vector_len}] as pca_vector
            FROM vectors_temp
        """)

    # Get 2D PCA coordinates from session_viz_data for visualization
    session_2d_coords = {}
    for session_num in ANALYSIS_SESSIONS:
        if session_num in session_viz_data:
            viz_df = session_viz_data[session_num]
            for _, row in viz_df.iterrows():
                key = (session_num, str(row["icpsr"]))
                session_2d_coords[key] = (row["PC1"], row["PC2"])

    for session_num in tqdm(ANALYSIS_SESSIONS, desc="Computing Party Cohesion"):
        try:
            # Calculate pairwise similarities within each party using DuckDB
            for party in ["D", "R"]:
                # Get all senators from this party in this session
                party_senators = duckdb_conn.execute("""
                    SELECT
                        icpsr,
                        senator_name,
                        political_party,
                        pca_vector
                    FROM senator_vectors
                    WHERE session_num = ? AND political_party = ?
                """, [session_num, party]).fetchdf()

                if len(party_senators) < 2:
                    continue

                # Calculate mean similarity for each senator to their party peers
                for idx, senator in party_senators.iterrows():
                    target_icpsr = senator["icpsr"]
                    target_vector = senator["pca_vector"]

                    # Calculate similarity to all other party members (excluding self)
                    similarities = duckdb_conn.execute(f"""
                        SELECT array_cosine_similarity(CAST(? AS FLOAT[{vector_len}]), pca_vector) as similarity
                        FROM senator_vectors
                        WHERE session_num = ?
                          AND political_party = ?
                          AND icpsr != ?
                    """, [target_vector, session_num, party, target_icpsr]).fetchdf()

                    if not similarities.empty and len(similarities) > 0:
                        mean_similarity = similarities["similarity"].mean()

                        # Get 2D coordinates for visualization from session_viz_data
                        coord_key = (session_num, str(target_icpsr))
                        pc1, pc2 = session_2d_coords.get(coord_key, (np.nan, np.nan))

                        senator_similarity_records.append({
                            "session_num": session_num,
                            "icpsr": target_icpsr,
                            "senator_name": senator["senator_name"],
                            "political_party": party,
                            "pc1": pc1,
                            "pc2": pc2,
                            "party_cohesion_score": mean_similarity
                        })

            # Calculate aggregate statistics for this session
            session_similarity_df = pd.DataFrame([
                r for r in senator_similarity_records
                if r["session_num"] == session_num
            ])

            if not session_similarity_df.empty:
                d_scores = session_similarity_df[
                    session_similarity_df["political_party"] == "D"
                ]["party_cohesion_score"]
                r_scores = session_similarity_df[
                    session_similarity_df["political_party"] == "R"
                ]["party_cohesion_score"]

                party_cohesion_stats.append({
                    "session_num": session_num,
                    "d_mean_cohesion": d_scores.mean() if not d_scores.empty else np.nan,
                    "d_median_cohesion": d_scores.median() if not d_scores.empty else np.nan,
                    "d_std_cohesion": d_scores.std() if not d_scores.empty else np.nan,
                    "d_count": len(d_scores) if not d_scores.empty else 0,
                    "r_mean_cohesion": r_scores.mean() if not r_scores.empty else np.nan,
                    "r_median_cohesion": r_scores.median() if not r_scores.empty else np.nan,
                    "r_std_cohesion": r_scores.std() if not r_scores.empty else np.nan,
                    "r_count": len(r_scores) if not r_scores.empty else 0,
                })

        except Exception as e:
            print(f"Error processing session {session_num}: {e}")
            continue

    senator_similarity_df = pd.DataFrame(senator_similarity_records)
    cohesion_stats_df = pd.DataFrame(party_cohesion_stats)

    return senator_similarity_df, cohesion_stats_df

# %%
# Initialize global variables for Exercise 5
# These will be populated by Visualization 1's PCA dimensionality control

current_pca_dims = 20
senator_similarity_df = pd.DataFrame()
cohesion_stats_df = pd.DataFrame()

# %% [markdown]
# ### Exercise 5 Visualization 1: Intra-Party Cohesion Over Time (with PCA Dimensionality Control)
# This interactive visualization allows you to:
# 1. **Adjust PCA dimensionality** (2D-50D) to explore how it affects similarity calculations
# 2. **View intra-party cohesion trends over time** for Democrats and Republicans
# 3. **Select session ranges** to focus on specific time periods
# **How to use:**
# - Adjust the "PCA Dims" slider to select dimensionality (2-50)
# - Click "Recompute Cohesion" to recalculate with new dimensionality
# - Use the "Sessions" slider to zoom into specific time periods
# - Observe how cohesion trends change with different PCA dimensions
# **Interpretation:**
# - Values closer to 1.0 indicate high intra-party cohesion (senators vote similarly)
# - Values closer to 0.0 indicate low intra-party cohesion (more diverse voting within party)
# - Higher PCA dimensions capture more variance → more accurate similarity scores

# %%
from ipywidgets import IntSlider, Button, HBox, VBox, Output, Label

# Create PCA dimensionality control widgets
pca_dim_slider = IntSlider(
    value=20,
    min=2,
    max=50,
    step=1,
    description="PCA Dims:",
    continuous_update=False,
    style={'description_width': 'initial'}
)

recompute_button = Button(
    description="Recompute Cohesion",
    button_style='primary',
    tooltip='Click to recompute PCA vectors and cohesion scores with selected dimensionality',
    icon='refresh'
)

# Create output widgets
recompute_status_output = Output()
cohesion_plot_output = Output()

# Create session range slider (will be populated after initial computation)
cohesion_range_slider = IntRangeSlider(
    value=(87, 119),
    min=87,
    max=119,
    step=1,
    description="Sessions:",
    continuous_update=False,
    layout=Layout(width="70%")
)

def render_cohesion_plot():
    """Render the cohesion time-series plot with current data"""
    global cohesion_stats_df, current_pca_dims

    with cohesion_plot_output:
        cohesion_plot_output.clear_output(wait=True)

        if cohesion_stats_df.empty:
            print("No cohesion data available. Click 'Recompute Cohesion' to generate data.")
            return

        # Get session window from slider
        lower, upper = cohesion_range_slider.value
        selected_sessions = [f"{num:03d}" for num in range(lower, upper + 1)]
        subset_df = cohesion_stats_df[cohesion_stats_df["session_num"].isin(selected_sessions)].copy()

        if subset_df.empty:
            print("No cohesion data available for the selected session window.")
            return

        subset_df = subset_df.sort_values("session_num")

        # Map session numbers to dates
        subset_df = subset_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")

        fig_cohesion = go.Figure()

        # Add Democrat cohesion line
        fig_cohesion.add_trace(go.Scatter(
            x=subset_df["start_year"],
            y=subset_df["d_mean_cohesion"],
            mode="lines+markers",
            name="Democrat Cohesion",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            text=subset_df["session_num"],
            customdata=subset_df[["d_count", "d_std_cohesion"]],
            hovertemplate="<b>Session %{text}</b><br>Year: %{x}<br>Mean Cohesion: %{y:.4f}<br>Count: %{customdata[0]}<br>Std Dev: %{customdata[1]:.4f}<extra></extra>"
        ))

        # Add Republican cohesion line
        fig_cohesion.add_trace(go.Scatter(
            x=subset_df["start_year"],
            y=subset_df["r_mean_cohesion"],
            mode="lines+markers",
            name="Republican Cohesion",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
            text=subset_df["session_num"],
            customdata=subset_df[["r_count", "r_std_cohesion"]],
            hovertemplate="<b>Session %{text}</b><br>Year: %{x}<br>Mean Cohesion: %{y:.4f}<br>Count: %{customdata[0]}<br>Std Dev: %{customdata[1]:.4f}<extra></extra>"
        ))

        fig_cohesion.update_layout(
            title=f"Intra-Party Cohesion Over Time ({current_pca_dims}D PCA)<br><sub>Higher = More Similar Voting Within Party</sub>",
            xaxis_title="Session Start Year",
            yaxis_title="Mean Cosine Similarity (Party Cohesion)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        fig_cohesion.show()

def on_recompute_clicked(b):
    """Callback for recompute button"""
    global current_pca_dims, senator_similarity_df, cohesion_stats_df

    with recompute_status_output:
        recompute_status_output.clear_output()

        n_components = pca_dim_slider.value
        current_pca_dims = n_components

        print(f"\n{'='*60}")
        print(f"Recomputing with {n_components}-dimensional PCA...")
        print(f"{'='*60}")

        # Compute PCA vectors
        vectors_df = compute_pca_vectors_ndim(n_components)

        if vectors_df.empty:
            print("❌ No vectors computed. Check data availability.")
            return

        print(f"✅ Computed {len(vectors_df)} PCA vectors")

        # Recalculate cohesion scores
        senator_similarity_df, cohesion_stats_df = recalculate_cohesion_scores(vectors_df)

        print(f"\n=== Exercise 5 Summary ({n_components}D PCA) ===")
        print(f"Senator similarity records: {len(senator_similarity_df)}")
        print(f"Sessions with cohesion stats: {len(cohesion_stats_df)}")

        if not cohesion_stats_df.empty:
            print(f"\nDemocrat mean cohesion range: {cohesion_stats_df['d_mean_cohesion'].min():.4f} - {cohesion_stats_df['d_mean_cohesion'].max():.4f}")
            print(f"Republican mean cohesion range: {cohesion_stats_df['r_mean_cohesion'].min():.4f} - {cohesion_stats_df['r_mean_cohesion'].max():.4f}")

            # Update session range slider bounds
            cohesion_session_ints = sorted([int(s) for s in cohesion_stats_df["session_num"].unique()])
            cohesion_range_slider.min = cohesion_session_ints[0]
            cohesion_range_slider.max = cohesion_session_ints[-1]
            cohesion_range_slider.value = (cohesion_session_ints[0], cohesion_session_ints[-1])

        # Store in DuckDB
        if not senator_similarity_df.empty:
            duckdb_conn.execute("DROP TABLE IF EXISTS senator_similarity")
            duckdb_conn.execute("""
                CREATE TABLE senator_similarity (
                    session_num VARCHAR,
                    icpsr VARCHAR,
                    senator_name VARCHAR,
                    political_party VARCHAR,
                    pc1 DOUBLE,
                    pc2 DOUBLE,
                    party_cohesion_score DOUBLE,
                    PRIMARY KEY (session_num, icpsr)
                )
            """)
            duckdb_conn.register("senator_similarity_temp", senator_similarity_df)
            duckdb_conn.execute("""
                INSERT INTO senator_similarity
                SELECT * FROM senator_similarity_temp
            """)
            print(f"\n✅ Stored {len(senator_similarity_df)} senator similarity records in DuckDB.")

        print(f"\n{'='*60}")
        print(f"✅ Recomputation complete! Plot updated below.")
        print(f"{'='*60}")

    # Update the plot with new data
    render_cohesion_plot()

def on_session_slider_change(change):
    """Callback for session range slider"""
    render_cohesion_plot()

# Wire up callbacks
recompute_button.on_click(on_recompute_clicked)
cohesion_range_slider.observe(on_session_slider_change, names='value')
display(VBox([pca_dim_slider, recompute_button, recompute_status_output, cohesion_plot_output]))

# %% [markdown]
# ### Exercise 5 Visualization 2: Interactive Senator Similarity Explorer
# This interactive tool allows you to:
# 1. Select a congressional session
# 2. Choose a specific senator
# 3. View a scatter plot showing:
#    - The selected senator (highlighted)
#    - Other senators color-coded by similarity to the selected senator
#    - Party boundaries for context
# 4. See a table of the top-k most similar senators
# **Use Cases:**
# - Identify senators who vote similarly across party lines
# - Find the most/least typical members of each party
# - Explore bipartisan coalitions or party outliers
# **Important Notes:**
# - **PCA Dimensionality**: Controlled in Visualization 1 above (currently using {current_pca_dims}D PCA)
# - **Similarity scores**: Computed using the N-dimensional PCA vectors for accuracy
# - **Scatter plot positions**: Use 2D PCA from Exercise 3 for visualization
# - This separation allows accurate similarity calculations while maintaining interpretable visualizations. 
# 
# 
# *Tip:* Try different PCA dimensions in Visualization 1 to see how similarity scores change

# %%
if not senator_similarity_df.empty and session_viz_data:
    print("\n=== Interactive Senator Similarity Explorer ===")
    print(f"Using {current_pca_dims}D PCA for similarity calculations (set in Visualization 1)")
    print("Select a session and senator to explore voting similarity patterns.\n")

    # Get available sessions
    available_sessions = sorted([int(s) for s in senator_similarity_df["session_num"].unique()])

    # Create session dropdown
    session_dropdown = Dropdown(
        options=[(f"Session {s} ({congress_dates_df[congress_dates_df['session_num']==f'{s:03d}']['start_year'].iloc[0] if not congress_dates_df[congress_dates_df['session_num']==f'{s:03d}'].empty else s})", s)
                 for s in available_sessions],
        value=available_sessions[-1] if available_sessions else None,
        description="Session:",
        style={'description_width': 'initial'}
    )

    # Create senator dropdown (will be populated based on session)
    senator_dropdown = Dropdown(
        options=[],
        description="Senator:",
        style={'description_width': 'initial'}
    )

    # Create output widget for plot
    plot_output = Output()

    # Create output widget for table
    table_output = Output()

    def update_senator_dropdown(change):
        """Update senator dropdown when session changes"""
        session_num = f"{change['new']:03d}"

        if session_num in session_viz_data:
            session_data = session_viz_data[session_num]
            # Get senators with their names
            senators = session_data[["icpsr", "senator_name", "political_party"]].drop_duplicates()
            senators = senators.sort_values("senator_name")

            senator_dropdown.options = [
                (f"{row['senator_name']} ({row['political_party']})", row['icpsr'])
                for _, row in senators.iterrows()
            ]

            if len(senator_dropdown.options) > 0:
                senator_dropdown.value = senator_dropdown.options[0][1]

    def calculate_senator_similarities(session_num: str, target_icpsr: str):
        """Calculate similarity between target senator and all others in session using DuckDB"""
        if session_num not in session_viz_data:
            return None

        session_data = session_viz_data[session_num].copy()

        # Get target senator's vector from DuckDB
        target_result = duckdb_conn.execute("""
            SELECT pca_vector
            FROM senator_vectors
            WHERE session_num = ? AND icpsr = ?
        """, [session_num, target_icpsr]).fetchone()

        if not target_result:
            return None

        target_vector = target_result[0]

        # Determine vector length from the fetched target_vector
        vector_len = len(target_vector) if target_vector is not None else None

        # Calculate similarity to all senators in this session using DuckDB
        similarities_df = duckdb_conn.execute(f"""
            SELECT
                icpsr,
                array_cosine_similarity(CAST(? AS FLOAT[{vector_len}]), pca_vector) as similarity
            FROM senator_vectors
            WHERE session_num = ?
        """, [target_vector, session_num]).fetchdf()

        # Merge similarities with session data
        session_data = session_data.merge(
            similarities_df,
            on="icpsr",
            how="left"
        )
        session_data = session_data.rename(columns={"similarity": "similarity_to_target"})

        return session_data

    def render_similarity_plot(session_num: str, target_icpsr: str):
        """Render scatter plot with similarity color coding"""
        session_data = calculate_senator_similarities(session_num, target_icpsr)

        if session_data is None or session_data.empty:
            print(f"No data available for session {session_num}")
            return

        # Get target senator info
        target_row = session_data[session_data["icpsr"] == target_icpsr].iloc[0]
        target_name = target_row["senator_name"]
        target_party = target_row["political_party"]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot all senators with similarity color coding
        scatter = ax.scatter(
            session_data["PC1"],
            session_data["PC2"],
            c=session_data["similarity_to_target"],
            cmap="RdYlGn",  # Red (low similarity) to Green (high similarity)
            s=100,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
            vmin=0,
            vmax=1
        )

        # Highlight target senator
        ax.scatter(
            target_row["PC1"],
            target_row["PC2"],
            s=400,
            c="gold",
            marker="*",
            edgecolor="black",
            linewidth=2,
            zorder=10,
            label=f"Selected: {target_name}"
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Cosine Similarity to Selected Senator", rotation=270, labelpad=20)

        # Get year range for title
        year_info = congress_dates_df[congress_dates_df["session_num"] == session_num]
        year_str = ""
        if not year_info.empty:
            start_year = year_info.iloc[0]["start_year"]
            end_year = year_info.iloc[0]["end_year"]
            year_str = f" ({start_year}-{end_year})"

        ax.set_title(f"Senator Similarity Analysis — Session {session_num}{year_str} ({current_pca_dims}D PCA)\nSelected: {target_name} ({target_party})")
        ax.set_xlabel("Principal Component 1 (2D projection for visualization)")
        ax.set_ylabel("Principal Component 2 (2D projection for visualization)")
        ax.axhline(0, color="#cccccc", linewidth=0.5)
        ax.axvline(0, color="#cccccc", linewidth=0.5)
        ax.set_aspect("equal")
        ax.legend(loc="best", frameon=True)

        plt.tight_layout()
        plt.show()

    def render_similarity_table(session_num: str, target_icpsr: str, top_k: int = 10):
        """Render table of most similar senators using DuckDB"""
        # Use DuckDB to efficiently get top-k most similar senators
        top_similar_df = duckdb_conn.execute("""
            WITH target AS (
                SELECT pca_vector
                FROM senator_vectors
                WHERE session_num = ? AND icpsr = ?
            )
            SELECT
                sv.senator_name,
                sv.political_party,
                array_cosine_similarity((SELECT pca_vector FROM target), sv.pca_vector) as similarity_score
            FROM senator_vectors sv
            WHERE sv.session_num = ?
              AND sv.icpsr != ?
            ORDER BY similarity_score DESC
            LIMIT ?
        """, [session_num, target_icpsr, session_num, target_icpsr, top_k]).fetchdf()

        if top_similar_df.empty:
            print(f"No similarity data available for session {session_num}")
            return

        # Format for display
        display_df = top_similar_df.copy()
        display_df.columns = ["Senator Name", "Party", "Similarity Score"]
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1  # Start index at 1

        print(f"\nTop {top_k} Most Similar Senators:")
        if "display" in globals() and display is not None:
            display(display_df)
        else:
            print(display_df.to_string())

    def on_selection_change(change):
        """Handle changes to session or senator selection"""
        with plot_output:
            plot_output.clear_output(wait=True)
            session_num = f"{session_dropdown.value:03d}"
            target_icpsr = senator_dropdown.value

            if target_icpsr:
                render_similarity_plot(session_num, target_icpsr)

        with table_output:
            table_output.clear_output(wait=True)
            session_num = f"{session_dropdown.value:03d}"
            target_icpsr = senator_dropdown.value

            if target_icpsr:
                render_similarity_table(session_num, target_icpsr, top_k=15)

    # Connect event handlers
    session_dropdown.observe(update_senator_dropdown, names='value')
    session_dropdown.observe(on_selection_change, names='value')
    senator_dropdown.observe(on_selection_change, names='value')

    # Initialize senator dropdown
    update_senator_dropdown({'new': session_dropdown.value})

    # Create UI layout
    controls = HBox([session_dropdown, senator_dropdown])
    ui = VBox([controls, plot_output, table_output])

    # Display UI
    display(ui)

    # Trigger initial render
    on_selection_change(None)
else:
    print("No senator similarity data available for interactive explorer.")

# %% [markdown]
# ## Exercise 6: Summary Visualization Dashboard
# This exercise presents a 2x2 grid of key polarization metrics from Exercises 2-5, providing a comprehensive overview of Senate polarization trends.
# **Dashboard Layout:**
# - **Top-Left**: Silhouette Score with Significant Shifts (Exercise 2)
# - **Top-Right**: Party-Cluster Alignment Mismatch (Exercise 3)
# - **Bottom-Left**: Combined Normalized Polarization Metrics (Exercise 4)
# - **Bottom-Right**: Intra-Party Cohesion Over Time (Exercise 5)
# 
# We should note that we present the interactive visualizations in the exercises above to satisfy
# the requirement for "a couple more visualizations that represent some measurement that could be used as an indicator of polarization."

# %% [markdown]
# Create a grid for your visualizations (silhouette and party-cluster similarity). Add a couple more of visualizations that represent some measurement that could be used as an indicator of polarization.

# %%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Prepare data for all four plots
print("\n=== Creating Exercise 6 Summary Dashboard ===")

# Check data availability
has_silhouette = not silhouette_df.empty if 'silhouette_df' in globals() else False
has_alignment = not alignment_df.empty if 'alignment_df' in globals() else False
has_combined = not combined_metrics_df.empty if 'combined_metrics_df' in globals() else False
has_cohesion = not cohesion_stats_df.empty if 'cohesion_stats_df' in globals() else False

if not all([has_silhouette, has_alignment, has_combined, has_cohesion]):
    print("⚠️ Warning: Some data is missing. Dashboard may be incomplete.")
    print(f"  Silhouette data: {'✓' if has_silhouette else '✗'}")
    print(f"  Alignment data: {'✓' if has_alignment else '✗'}")
    print(f"  Combined metrics: {'✓' if has_combined else '✗'}")
    print(f"  Cohesion data: {'✓' if has_cohesion else '✗'}")

# Create figure with 2x2 grid
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Silhouette Score with Significant Shifts (Top-Left)
ax1 = fig.add_subplot(gs[0, 0])
if has_silhouette:
    # Merge with congress dates for x-axis
    plot_df = silhouette_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
    plot_df = plot_df.sort_values("start_year")

    # Calculate rolling average and shifts
    plot_df['rolling_avg'] = plot_df['silhouette_score'].rolling(window=7, center=True, min_periods=1).mean()
    plot_df['delta'] = plot_df['silhouette_score'] - plot_df['rolling_avg']
    delta_std = plot_df['delta'].std()
    plot_df['significant_shift'] = plot_df['delta'].abs() > (1.0 * delta_std)

    # Plot
    ax1.plot(plot_df['start_year'], plot_df['silhouette_score'], 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Silhouette Score')
    ax1.plot(plot_df['start_year'], plot_df['rolling_avg'], '--', color='gray', linewidth=1.5, label='Rolling Avg (7 sessions)')

    # Highlight significant shifts
    shift_df = plot_df[plot_df['significant_shift']]
    if not shift_df.empty:
        ax1.scatter(shift_df['start_year'], shift_df['silhouette_score'],
                   color='red', s=100, marker='o', facecolors='none', linewidths=2,
                   label='Significant Shift', zorder=5)

    ax1.set_xlabel('Session Start Year', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Exercise 2: Silhouette Score with Significant Shifts\n(Higher = Better Cluster Separation)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'Silhouette data not available', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Exercise 2: Silhouette Score', fontsize=12, fontweight='bold')

# Plot 2: Party-Cluster Alignment Mismatch (Top-Right)
ax2 = fig.add_subplot(gs[0, 1])
if has_alignment:
    plot_df = alignment_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
    plot_df = plot_df.sort_values("start_year")

    ax2.plot(plot_df['start_year'], plot_df['mismatch_pct'], 'o-', color='#ff7f0e', linewidth=2, markersize=4)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% (Random)')
    ax2.set_xlabel('Session Start Year', fontsize=11)
    ax2.set_ylabel('Mismatch Percentage (%)', fontsize=11)
    ax2.set_title('Exercise 3: Party-Cluster Alignment Mismatch\n(Lower = Higher Polarization)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Alignment data not available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Exercise 3: Party-Cluster Alignment', fontsize=12, fontweight='bold')

# Plot 3: Combined Normalized Metrics (Bottom-Left)
ax3 = fig.add_subplot(gs[1, 0])
if has_combined:
    plot_df = combined_metrics_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
    plot_df = plot_df.sort_values("start_year")

    ax3.plot(plot_df['start_year'], plot_df['silhouette_norm'], 'o-', label='Silhouette', linewidth=2, markersize=3)
    ax3.plot(plot_df['start_year'], plot_df['dunn_norm'], 's-', label='Dunn Index', linewidth=2, markersize=3)
    ax3.plot(plot_df['start_year'], plot_df['db_norm'], '^-', label='Davies-Bouldin', linewidth=2, markersize=3)
    ax3.plot(plot_df['start_year'], plot_df['ch_norm'], 'd-', label='Calinski-Harabasz', linewidth=2, markersize=3)
    ax3.plot(plot_df['start_year'], plot_df['crosstab_norm'], 'v-', label='Crosstab Sep.', linewidth=2, markersize=3)

    ax3.set_xlabel('Session Start Year', fontsize=11)
    ax3.set_ylabel('Normalized Score (0-1)', fontsize=11)
    ax3.set_title('Exercise 4: Combined Normalized Polarization Metrics\n(All Scaled to 0-1 Range)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'Combined metrics not available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Exercise 4: Combined Metrics', fontsize=12, fontweight='bold')

# Plot 4: Intra-Party Cohesion (Bottom-Right)
ax4 = fig.add_subplot(gs[1, 1])
if has_cohesion:
    plot_df = cohesion_stats_df.merge(congress_dates_df[["session_num", "start_year"]], on="session_num", how="left")
    plot_df = plot_df.sort_values("start_year")

    ax4.plot(plot_df['start_year'], plot_df['d_mean_cohesion'], 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Democrat Cohesion')
    ax4.plot(plot_df['start_year'], plot_df['r_mean_cohesion'], 's-', color='#d62728', linewidth=2, markersize=4, label='Republican Cohesion')

    ax4.set_xlabel('Session Start Year', fontsize=11)
    ax4.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax4.set_title(f'Exercise 5: Intra-Party Cohesion Over Time ({current_pca_dims}D PCA)\n(Higher = More Similar Voting Within Party)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Cohesion data not available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Exercise 5: Intra-Party Cohesion', fontsize=12, fontweight='bold')

# Add overall title
# get beginning and end sessions from ANALYSIS_SESSIONS
start_sess, end_sess = min(ANALYSIS_SESSIONS), max(ANALYSIS_SESSIONS)
fig.suptitle(f"Senate Polarization Analysis Dashboard ({start_sess}th-{end_sess}th Congress)",
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n✅ Exercise 6 dashboard created successfully!")

# %% [markdown]
# ## References
# ### Implementation Planning
# - **Google Gemini Conversation** (Implementation Plan Draft): https://gemini.google.com/share/ead9be9470dd
# ### Code Development Tools
# - **Github Copilot Agent**: https://code.visualstudio.com/docs/copilot/copilot-coding-agent
# - **Augment Code Agent**: https://docs.augmentcode.com/using-augment/agent
# 
# These AI coding assistants helped speed up code writing, debugging, and documentation. 
# These were leveraged particularly for the interactive plot and DuckDB ingestion sections.
# 
# ### Data Sources
# - **VoteView Data Portal**: https://voteview.com/data
# - **Senate Vote CSVs (Pattern)**: https://voteview.com/static/data/out/votes/S<NUM>_votes.csv
# - **All Members Metadata**: https://voteview.com/static/data/out/members/HSall_members.csv
# ### DuckDB Documentation & Resources
# - **Multiple CSV Files Overview**: https://duckdb.org/docs/stable/data/multiple_files/overview#csv
# - **Python CSV Ingestion**: https://duckdb.org/docs/stable/clients/python/data_img/data_ingestion#csv-files
# - **CSV Reading Tips**: https://duckdb.org/docs/stable/data/csv/tips
# - **Handling Faulty CSVs**: https://duckdb.org/docs/stable/data/csv/reading_faulty_csv_files
# - **DuckDB Tricks (Blog)**: https://duckdb.org/2024/08/19/duckdb-tricks-part-1
# - **Taming Wild CSVs (Blog)**: https://motherduck.com/blog/taming-wild-csvs-with-duckdb-data-engineering/
# - **Persisting CSVs (Blog)**: https://motherduck.com/blog/csv-files-persist-duckdb-solution/
# - **Video: Taming Wild CSVs**: https://www.youtube.com/watch?v=pHeVP92O9zc
# - **DataFrame Glossary**: https://motherduck.com/glossary/DataFrame/
# - **Array Processing**: https://duckdb.org/docs/stable/sql/functions/array#array_cosine_similarityarray1-array2
# - **Vector Similarity Search**: https://duckdb.org/2024/05/03/vector-similarity-search-vss
# - **DuckDB VSS Extension**: https://github.com/duckdb/duckdb-vss
# - **MotherDuck Vector Search Blog**: https://motherduck.com/blog/search-using-duckdb-part-1/
# - **HuggingFace DuckDB Vector Similarity**: https://huggingface.co/docs/hub/en/datasets-duckdb-vector-similarity-search
# ### Python Libraries & Techniques
# - **defaultdict Tutorial**: https://realpython.com/python-defaultdict/
# - **Handling KeyError**: https://www.datacamp.com/tutorial/python-keyerror
# - **Dunn Index Implementation Repo**: https://github.com/jqmviegas/jqm_cvi/tree/master
# ### Polarization Metrics
# **Dunn Index:**
# - GeeksforGeeks: https://www.geeksforgeeks.org/machine-learning/dunn-index-and-db-index-cluster-validity-indices-set-1/
# - Theory PDF: https://github.com/jqmviegas/jqm_cvi/blob/master/theory.pdf
# - Wikipedia: https://en.wikipedia.org/wiki/Dunn_index
# **Davies-Bouldin Index:**
# - GeeksforGeeks: https://www.geeksforgeeks.org/machine-learning/davies-bouldin-index/
# - Wikipedia: https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
# - scikit-learn Docs: https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
# - scikit-learn API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
# **Calinski-Harabasz Index:**
# - Towards Data Science: https://towardsdatascience.com/calinski-harabasz-index-for-k-means-clustering-evaluation-using-python-4fefeeb2988e/
# - Wikipedia: https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index
# - scikit-learn Docs: https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
# - scikit-learn API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
# ### Plotly & Dash
# - **Plotly Getting Started**: https://plotly.com/python/getting-started/
# - **Plotly v3 Notebook Example**: https://plotly.com/python/v3/ipython-notebooks/baltimore-vital-signs/
# - **GeeksforGeeks Plotly Tutorial**: https://www.geeksforgeeks.org/data-visualization/using-plotly-for-interactive-data-visualization-in-python/
# - **Dash Tutorial**: https://dash.plotly.com/tutorial
# - **Dash in Jupyter**: https://dash.plotly.com/dash-in-jupyter
# - **Jupyter Support Update (GitHub)**: https://github.com/plotly/jupyter-dash?tab=readme-ov-file#notice-as-of-dash-v211-jupyter-support-is-built-into-the-main-dash-package
# - **Plotly Figure Structure**: https://plotly.com/python/figure-structure/
# - **Creating & Updating Figures**: https://plotly.com/python/creating-and-updating-figures/


