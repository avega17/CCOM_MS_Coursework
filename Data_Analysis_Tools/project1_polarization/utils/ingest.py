"""Ingestion utilities for Senate vote and member datasets."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import requests

from .config import Settings

logger = logging.getLogger(__name__)


_SESSION_PATTERN = re.compile(r"S(\d{3})_votes\.csv", re.IGNORECASE)


def _escape_literal(value: str) -> str:
    """Escape single quotes for safe SQL string literals."""
    return value.replace("'", "''")


def _sql_quote(value: str) -> str:
    """Return a SQL-safe single-quoted literal."""

    return "'" + value.replace("'", "''") + "'"


def initialize_database(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for the project."""

    settings.ensure_runtime_directories()
    cache_directory = settings.votes_dir / ".duckdb-quackstore-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(settings.duckdb_path))
    connection.execute(
        f"SET temp_directory='{_escape_literal(str(settings.temp_directory))}'"
    )
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    try:
        connection.execute("INSTALL quackstore FROM community")
        connection.execute("LOAD quackstore")
        cache_path = _escape_literal(cache_directory.as_posix())
        connection.execute(
            f"SET GLOBAL quackstore_cache_path='{cache_path}'"
        )
        with contextlib.suppress(duckdb.Error):
            connection.execute(
                f"SET quackstore_cache_path='{cache_path}'"
            )
        connection.execute("SET GLOBAL enable_http_metadata_cache=TRUE")
        with contextlib.suppress(duckdb.Error):
            connection.execute("SET enable_http_metadata_cache=TRUE")
        connection.execute("SET GLOBAL enable_object_cache=TRUE")
        with contextlib.suppress(duckdb.Error):
            connection.execute("SET enable_object_cache=TRUE")
    except duckdb.Error as exc:  # pragma: no cover - depends on extension availability
        logger.warning(
            "QuackStore extension not available or configuration failed: %s",
            exc,
        )
    logger.info("DuckDB initialized at %s", settings.duckdb_path)
    return connection


def ingest_vote_files(
    connection: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    source_glob: Optional[str] = None,
) -> int:
    """Load Senate vote CSVs into a temporary DuckDB table.

    Parameters
    ----------
    connection: duckdb.DuckDBPyConnection
        Active connection configured via :func:`initialize_database`.
    settings: Settings
        Global project settings.
    source_glob: Optional[str]
        Override path or URI glob passed to DuckDB ``read_csv``.

    Returns
    -------
    int
        Row count stored in the temporary raw votes table.
    """

    glob = source_glob or settings.votes_glob
    query = f"""
        CREATE OR REPLACE TEMP TABLE {settings.raw_votes_table} AS
        SELECT *
        FROM read_csv(
            '{_escape_literal(glob)}',
            auto_detect=TRUE,
            header=TRUE,
            union_by_name=TRUE,
            ignore_errors=TRUE,
            filename=TRUE,
            sample_size=-1,
            parallel=TRUE
        );
    """
    logger.info("Reading vote files via glob: %s", glob)
    connection.execute(query)
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {settings.raw_votes_table}"
    ).fetchone()[0]
    logger.info("Loaded %s rows into %s", row_count, settings.raw_votes_table)
    return int(row_count)


def create_processed_vote_table(
    connection: duckdb.DuckDBPyConnection,
    settings: Settings,
) -> int:
    """Persist the cleaned Senate votes table with extracted session numbers."""

    query = f"""
        CREATE OR REPLACE TABLE {settings.processed_votes_table} AS
        SELECT
            regexp_extract(filename, 'S(\d{{3}})_votes\\.csv', 1) AS session_num,
            COALESCE(
                CAST(try_cast(icpsr AS BIGINT) AS VARCHAR),
                regexp_replace(CAST(icpsr AS VARCHAR), '\\.(0)+$', ''),
                CAST(icpsr AS VARCHAR)
            ) AS icpsr,
            * EXCLUDE (filename, icpsr)
        FROM {settings.raw_votes_table};
    """
    connection.execute(query)
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {settings.processed_votes_table}"
    ).fetchone()[0]
    logger.info(
        "Persisted %s rows into %s",
        row_count,
        settings.processed_votes_table,
    )
    return int(row_count)


def ingest_member_metadata(
    connection: duckdb.DuckDBPyConnection,
    settings: Settings,
    export=True,
    *,
    persist_local: bool = False,
) -> int:
    """Ingest the VoteView member metadata file into DuckDB.

    Parameters
    ----------
    persist_local: bool
        When ``True``, download the CSV to ``settings.members_local_path`` prior
        to ingestion to showcase on-disk caching.
    """

    source = settings.members_uri
    if persist_local:
        target_path = settings.members_local_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        source = target_path.as_posix()
        logger.info("Cached members CSV at %s", target_path)

    query = f"""
        CREATE OR REPLACE TABLE {settings.members_table} AS
        SELECT
            CAST(icpsr AS VARCHAR) AS icpsr,
            LPAD(CAST(congress AS VARCHAR), 3, '0') AS session_num,
            bioname AS senator_name,
            party_code,
            CASE
                WHEN party_code = 100 THEN 'D'
                WHEN party_code = 200 THEN 'R'
                WHEN party_code = 328 THEN 'I'
                ELSE 'Other'
            END AS political_party
        FROM read_csv(
            '{_escape_literal(source)}',
            auto_detect=TRUE,
            header=TRUE,
            union_by_name=TRUE,
            sample_size=-1,
            ignore_errors=TRUE
        )
        WHERE UPPER(chamber) = 'SENATE';
    """
    logger.info("Ingesting member metadata from %s", source)
    connection.execute(query)
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {settings.members_table}"
    ).fetchone()[0]
    logger.info("Loaded %s rows into %s", row_count, settings.members_table)

    if export:
        uri_filename = source.split("/")[-1]
        export_path = os.path.join(settings.votes_dir, uri_filename)
        export_query = """
            COPY (
                SELECT * FROM {settings.members_table}
            )
            TO '{_escape_literal(export_path)}'
            (FORMAT 'csv', HEADER TRUE, OVERWRITE_OR_IGNORE TRUE);
        """
        connection.execute(export_query)

    return int(row_count)


def summarize_vote_file_storage(settings: Settings) -> Tuple[int, int]:
    """Return the count and total size (bytes) of locally stored vote CSV files."""

    files = list_local_vote_files(settings)
    total_size = 0
    for file_path in files:
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return len(files), total_size


def summarize_duckdb_size(settings: Settings) -> int:
    """Return the size (bytes) of the DuckDB database file, if it exists."""

    db_path = settings.duckdb_path
    if db_path.exists():
        return db_path.stat().st_size
    return 0


def summarize_members_file_storage(settings: Settings) -> Optional[int]:
    """Return the size (bytes) of the members file, local or remote."""

    local_path = settings.members_local_path
    if local_path.exists():
        return local_path.stat().st_size

    with contextlib.suppress(requests.RequestException):
        response = requests.head(settings.members_uri, timeout=30)
        if response.status_code == 200:
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
    return None


def format_session(session: str | int) -> str:
    """Normalise a session identifier to the expected zero-padded string."""

    text = str(session)
    return text.zfill(3)


def list_local_vote_files(settings: Settings) -> List[Path]:
    """Return sorted local vote CSV paths."""

    return sorted(settings.votes_dir.rglob("S*_votes.csv"))


def list_local_vote_sessions(settings: Settings) -> List[str]:
    """Return discovered session identifiers present in the local folder."""

    sessions = set()
    for file_path in list_local_vote_files(settings):
        match = _SESSION_PATTERN.search(file_path.name)
        if match:
            sessions.add(match.group(1))
    return sorted(sessions)


def build_vote_uris(settings: Settings, sessions: Sequence[str | int]) -> List[str]:
    """Construct HTTP URIs for the provided session identifiers."""

    uris: List[str] = []
    template = settings.vote_uri_template
    for session in sessions:
        padded = format_session(session)
        uris.append(template.format(session=padded))
    return uris


def summarize_remote_vote_size(settings: Settings, sessions: Iterable[str]) -> Optional[int]:
    """Estimate cumulative size of remote vote files via HTTP HEAD requests."""

    total = 0
    found_any = False
    for session in sessions:
        uri = settings.vote_uri_template.format(session=format_session(session))
        with contextlib.suppress(requests.RequestException):
            response = requests.head(uri, timeout=30)
            if response.status_code == 200:
                found_any = True
                content_length = response.headers.get("Content-Length")
                if content_length:
                    total += int(content_length)
    if found_any:
        return total
    return None


def get_missing_sessions(settings: Settings, sessions: Sequence[str | int]) -> List[str]:
    """Return zero-padded session identifiers missing from the local store."""

    desired = {format_session(session) for session in sessions}
    local = set(list_local_vote_sessions(settings))
    return sorted(desired - local)


def download_vote_files(
    settings: Settings,
    sessions: Sequence[str | int],
    *,
    overwrite: bool = True,
) -> List[Path]:
    """Download specified sessions via DuckDB partitioned export."""

    sessions_formatted = [format_session(session) for session in sessions]
    if not sessions_formatted:
        return []

    uris = build_vote_uris(settings, sessions_formatted)
    uri_list_sql = ", ".join(_sql_quote(uri) for uri in uris)
    connection = initialize_database(settings)

    before_files = set(list_local_vote_files(settings))

    temp_table = "__remote_vote_download"
    read_sql = f"""
        CREATE OR REPLACE TEMP TABLE {temp_table} AS
        SELECT *,
               regexp_extract(filename, 'S(\\d{{3}})_votes\\.csv', 1) AS session_num
        FROM read_csv([
            {uri_list_sql}
        ],
        auto_detect=TRUE,
        header=TRUE,
        union_by_name=TRUE,
        ignore_errors=TRUE,
        sample_size=-1,
        filename=TRUE,
        parallel=TRUE);
    """

    try:
        connection.execute(read_sql)
        for session in sessions_formatted:
            target_path = settings.votes_dir / f"S{session}_votes.csv"
            if target_path.exists() and not overwrite:
                logger.debug("Skipping existing vote file: %s", target_path)
                continue
            copy_sql = f"""
                COPY (
                    SELECT * EXCLUDE (session_num, filename)
                    FROM {temp_table}
                    WHERE session_num = '{_escape_literal(session)}'
                )
                TO '{_escape_literal(target_path.as_posix())}'
                (FORMAT 'csv', HEADER TRUE, OVERWRITE_OR_IGNORE TRUE);
            """
            connection.execute(copy_sql)
    finally:
        connection.close()

    after_files = set(list_local_vote_files(settings))
    return sorted(after_files - before_files)


def ensure_vote_files(
    settings: Settings,
    sessions: Sequence[str | int],
    *,
    download: bool = True,
) -> List[str]:
    """Ensure all sessions are available locally, downloading if needed."""

    missing = get_missing_sessions(settings, sessions)
    if download and missing:
        download_vote_files(settings, missing)
    return missing


def fetch_congress_dates(settings: Settings) -> pd.DataFrame:
    """
    Fetch congressional session dates from senate.gov and save to CSV.
    
    Args:
        settings: Project settings with votes_dir path
        
    Returns:
        DataFrame with congress session date information
    """
    url = "https://www.senate.gov/legislative/DatesofSessionsofCongress.htm"
    temp_html = settings.votes_dir / "_temp_congress_dates.html"
    output_csv = settings.votes_dir / "congress_dates.csv"
    
    try:
        # Use wget to fetch the HTML
        logger.info("Fetching congressional session dates from %s", url)
        subprocess.run(
            ["wget", "-q", "-O", str(temp_html), url],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Use pandas to read the HTML table
        tables = pd.read_html(str(temp_html))
        
        # Find the table with Congress data (should be the main one)
        congress_table = None
        for table in tables:
            if 'Congress' in table.columns and 'Begin Date' in table.columns:
                congress_table = table
                break
        
        if congress_table is None or congress_table.empty:
            raise ValueError("Could not find congressional dates table in HTML")
        
        # Process the table into structured data
        congress_data = []
        for _, row in congress_table.iterrows():
            try:
                congress_num = str(row['Congress']).strip()
                if not congress_num or not congress_num.replace('.', '').isdigit():
                    continue
                    
                congress_num = int(float(congress_num))
                
                # Get the begin date - could have multiple sessions
                begin_date = str(row['Begin Date']).strip()
                if not begin_date or begin_date == 'nan':
                    continue
                
                # Extract year from first date (format like "Jan 3, 2025")
                # Handle cases with multiple sessions separated by newlines
                first_date = begin_date.split('\n')[0].strip() if '\n' in begin_date else begin_date
                date_parts = first_date.split()
                
                if len(date_parts) >= 3:
                    year_str = date_parts[-1].rstrip('.,;')
                    start_year = int(year_str)
                    
                    congress_data.append({
                        'congress': congress_num,
                        'session_num': str(congress_num).zfill(3),
                        'start_year': start_year,
                        'end_year': start_year + 2,
                        'label': f"{congress_num} ({start_year}-{start_year + 2})"
                    })
            except (ValueError, IndexError, KeyError) as e:
                logger.debug("Skipping row due to parse error: %s", e)
                continue
        
        # Convert to DataFrame and save
        df = pd.DataFrame(congress_data)
        df = df.sort_values('congress')
        df.to_csv(output_csv, index=False)
        
        logger.info("Saved %s congressional session dates to %s", len(df), output_csv)
        
        # Clean up temp HTML
        if temp_html.exists():
            temp_html.unlink()
            
        return df
        
    except Exception as exc:
        logger.error("Error fetching congress dates: %s", exc)
        # Clean up temp file if it exists
        if temp_html.exists():
            temp_html.unlink()
        raise


def load_congress_dates(settings: Settings) -> pd.DataFrame:
    """Load congressional session date mappings from CSV file.
    
    Args:
        settings: Project settings with votes_dir path
        
    Returns:
        DataFrame with congress session date information
    """
    dates_file = settings.votes_dir / "congress_dates.csv"
    
    if not dates_file.exists():
        logger.info("Congress dates file not found. Fetching...")
        return fetch_congress_dates(settings)
    
    return pd.read_csv(dates_file)
