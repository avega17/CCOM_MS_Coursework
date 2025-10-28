"""Data transformation helpers for senate session analysis."""

from __future__ import annotations

from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

__all__ = [
    "prepare_session_matrix",
    "compute_session_silhouette",
    "load_silhouette_enriched",
    "refresh_silhouette_enriched_table",
]


def prepare_session_matrix(votes_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Prepare a session vote matrix for clustering analysis.

    Args:
        votes_df: DataFrame with icpsr, rollnumber, and cast_code columns.

    Returns:
        DataFrame of vote features indexed by member ``icpsr`` or ``None`` if
        the session cannot be analysed.
    """
    if votes_df.empty:
        return None

    session_df = votes_df.copy()
    session_df["icpsr"] = session_df["icpsr"].astype(str)
    session_pivot = session_df.pivot_table(
        values="cast_code",
        index="icpsr",
        columns="rollnumber",
        aggfunc="first",
    )
    session_pivot = session_pivot.replace([1, 2, 3], 1)
    session_pivot = session_pivot.replace([4, 5, 6], 0)
    session_pivot = session_pivot.replace([7, 8, 9, 9.0], 0.5)
    session_pivot = session_pivot.fillna(0.5)
    session_pivot = session_pivot.sort_index()

    if session_pivot.shape[0] < 3:
        return None

    return session_pivot


def compute_session_silhouette(
    connection: duckdb.DuckDBPyConnection,
    processed_votes_table: str,
    session_num: str,
) -> Optional[float]:
    """Compute silhouette score for a single session.
    
    Args:
        connection: DuckDB connection
        processed_votes_table: Name of the processed votes table
        session_num: Session number to analyze
        
    Returns:
        Silhouette score, or None if computation fails
    """
    votes_df = connection.execute(
        f"""
        SELECT icpsr, rollnumber, cast_code
        FROM {processed_votes_table}
        WHERE session_num = ?
        """,
        [session_num],
    ).fetchdf()

    feature_frame = prepare_session_matrix(votes_df)
    if feature_frame is None:
        return None

    feature_matrix = feature_frame.to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster_assignments = kmeans.fit_predict(feature_matrix)
    if len(np.unique(cluster_assignments)) < 2:
        return None

    pca_model = PCA(n_components=2, random_state=42)
    components = pca_model.fit_transform(feature_matrix)
    return float(silhouette_score(components, cluster_assignments))


def refresh_silhouette_enriched_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Compute rolling statistics and significant-shift flags for stored silhouettes.
    
    Args:
        conn: DuckDB connection with session_silhouette_scores table
    """
    conn.execute(
        """
        CREATE OR REPLACE TABLE session_silhouette_enriched AS
        WITH ordered AS (
            SELECT
                session_num,
                silhouette_score,
                CAST(session_num AS INTEGER) AS session_number
            FROM session_silhouette_scores
            WHERE silhouette_score IS NOT NULL
        ),
        rolling AS (
            SELECT
                session_num,
                session_number,
                silhouette_score,
                AVG(silhouette_score) OVER (
                    ORDER BY session_number
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS rolling_avg
            FROM ordered
        ),
        delta AS (
            SELECT
                session_num,
                session_number,
                silhouette_score,
                rolling_avg,
                silhouette_score - rolling_avg AS delta
            FROM rolling
        ),
        stats AS (
            SELECT
                session_num,
                session_number,
                silhouette_score,
                rolling_avg,
                delta,
                COALESCE(STDDEV_POP(delta) OVER (), 0) AS delta_std
            FROM delta
        )
        SELECT
            session_num,
            silhouette_score,
            rolling_avg,
            delta,
            delta_std,
            CASE
                WHEN delta_std > 0 AND ABS(delta) >= 1.0 * delta_std THEN TRUE
                ELSE FALSE
            END AS significant_shift
        FROM stats
        ORDER BY session_number
        """
    )


def load_silhouette_enriched(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return the silhouette summary table as a pandas DataFrame.
    
    Args:
        conn: DuckDB connection with session_silhouette_enriched table
        
    Returns:
        DataFrame with enriched silhouette scores
    """
    return conn.execute(
        """
        SELECT session_num,
               silhouette_score,
               rolling_avg,
               delta,
               delta_std,
               significant_shift
        FROM session_silhouette_enriched
        ORDER BY CAST(session_num AS INTEGER)
        """
    ).fetchdf()
