"""Benchmark helpers for comparing ingestion strategies."""

from __future__ import annotations

import gc
import time
from typing import Dict, Iterable, List, Optional

import duckdb
import pandas as pd

from .config import Settings
from .ingest import (
    build_vote_uris,
    initialize_database,
    ingest_vote_files,
    list_local_vote_files,
    list_local_vote_sessions,
    summarize_duckdb_size,
    summarize_remote_vote_size,
)


def _format_result(
    label: str,
    seconds: Optional[float],
    *,
    rows: Optional[int] = None,
    files: Optional[int] = None,
    bytes_processed: Optional[int] = None,
    note: str | None = None,
) -> Dict[str, Optional[float | int | str]]:
    return {
        "method": label,
        "seconds": seconds,
        "rows": rows,
        "files": files,
        "bytes": bytes_processed,
        "note": note,
    }


def _sql_quote(value: str) -> str:
    """Return a SQL-safe single-quoted literal string."""

    return "'" + value.replace("'", "''") + "'"


def benchmark_pandas_bulk_load(settings: Settings) -> Dict[str, Optional[float | int | str]]:
    """Load all local vote CSVs with pandas and report elapsed time and memory."""

    files = list_local_vote_files(settings)
    if not files:
        return _format_result(
            "pandas-local",
            None,
            files=0,
            note="No local vote CSV files detected",
        )

    start = time.perf_counter()
    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    elapsed = time.perf_counter() - start
    memory_bytes = int(combined.memory_usage(deep=True).sum())
    row_count = len(combined)

    del frames
    del combined
    gc.collect()

    return _format_result(
        "pandas-local",
        elapsed,
        rows=row_count,
        files=len(files),
        bytes_processed=memory_bytes,
    )


def benchmark_duckdb_local_ingest(settings: Settings) -> Dict[str, Optional[float | int | str]]:
    """Time DuckDB ingestion from local CSVs using the project pipeline."""

    files = list_local_vote_files(settings)
    if not files:
        return _format_result(
            "duckdb-local",
            None,
            files=0,
            note="No local vote CSV files detected",
        )

    connection = initialize_database(settings)
    start = time.perf_counter()
    row_count = ingest_vote_files(connection, settings)
    elapsed = time.perf_counter() - start
    connection.close()

    db_bytes = summarize_duckdb_size(settings)

    return _format_result(
        "duckdb-local",
        elapsed,
        rows=row_count,
        files=len(files),
        bytes_processed=db_bytes,
    )


def benchmark_duckdb_remote_fetch(
    settings: Settings,
    *,
    target_sessions: Optional[Iterable[str | int]] = None,
    missing_only: bool = True,
) -> Dict[str, Optional[float | int | str]]:
    """Time DuckDB ingestion from remote VoteView CSVs without persisting."""

    if target_sessions is None:
        target_sessions = [f"{num:03d}" for num in range(40, 120)]

    sessions: List[str] = [str(session).zfill(3) for session in target_sessions]

    if missing_only:
        local_sessions = set(list_local_vote_sessions(settings))
        sessions = [session for session in sessions if session not in local_sessions]

    if not sessions:
        return _format_result(
            "duckdb-remote",
            None,
            files=0,
            note="No remote sessions selected (all already cached locally)",
        )

    uris = build_vote_uris(settings, sessions)
    connection = initialize_database(settings)

    uri_list_sql = ", ".join(_sql_quote(uri) for uri in uris)
    query = f"""
        SELECT COUNT(*)
        FROM read_csv([
            {uri_list_sql}
        ],
        auto_detect=TRUE,
        header=TRUE,
        union_by_name=TRUE,
        ignore_errors=TRUE,
        sample_size=-1,
        parallel=TRUE)
    """

    try:
        start = time.perf_counter()
        row_count = connection.execute(query).fetchone()[0]
        elapsed = time.perf_counter() - start
    except duckdb.Error as exc:  # pragma: no cover - diagnostic path only
        connection.close()
        return _format_result(
            "duckdb-remote",
            None,
            files=len(uris),
            note=f"DuckDB error: {exc}",
        )
    connection.close()

    remote_bytes = summarize_remote_vote_size(settings, sessions)

    return _format_result(
        "duckdb-remote",
        elapsed,
        rows=row_count,
        files=len(uris),
        bytes_processed=remote_bytes,
        note="Fetched via HTTP" if remote_bytes else None,
    )