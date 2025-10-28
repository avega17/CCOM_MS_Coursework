"""Configuration helpers for the Senate polarization project."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(value: Optional[str], default: str) -> Path:
    target = value or default
    path = Path(target)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def _resolve_glob(value: Optional[str], default: str) -> str:
    target = value or default
    if "://" in target:
        return target
    glob_path = (BASE_DIR / target).resolve()
    return glob_path.as_posix()


@dataclass(frozen=True)
class Settings:
    """Project-wide configuration derived from the `.env` file."""

    duckdb_path: Path
    temp_directory: Path
    votes_dir: Path
    votes_glob: str
    members_uri: str
    vote_uri_template: str
    members_local_path: Path
    raw_votes_table: str
    processed_votes_table: str
    members_table: str

    def ensure_runtime_directories(self) -> None:
        """Ensure directories DuckDB relies on exist."""
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        self.votes_dir.mkdir(parents=True, exist_ok=True)
        self.members_local_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Load project settings from environment variables and defaults."""

    duckdb_path = _resolve_path(os.getenv("DUCKDB_PATH"), "senate_analysis.duckdb")
    temp_directory = _resolve_path(os.getenv("TEMP_DIRECTORY"), ".duckdb-temp")
    votes_dir = _resolve_path(os.getenv("VOTES_DIR"), "senate_dataset")
    votes_glob = _resolve_glob(os.getenv("VOTES_GLOB"), "senate_dataset/S*_votes.csv")
    members_uri = os.getenv(
        "MEMBERS_URI",
        "https://voteview.com/static/data/out/members/HSall_members.csv",
    )
    vote_uri_template = os.getenv(
        "VOTE_URI_TEMPLATE",
        "https://voteview.com/static/data/out/votes/S{session}_votes.csv",
    )
    members_local_path = _resolve_path(
        os.getenv("MEMBERS_LOCAL_PATH"), "senate_dataset/HSall_members.csv"
    )
    raw_votes_table = os.getenv("RAW_VOTES_TABLE", "raw_votes")
    processed_votes_table = os.getenv(
        "PROCESSED_VOTES_TABLE", "senate_votes_processed"
    )
    members_table = os.getenv("MEMBERS_TABLE", "members")

    settings = Settings(
        duckdb_path=duckdb_path,
        temp_directory=temp_directory,
        votes_dir=votes_dir,
        votes_glob=votes_glob,
        members_uri=members_uri,
        vote_uri_template=vote_uri_template,
        members_local_path=members_local_path,
        raw_votes_table=raw_votes_table,
        processed_votes_table=processed_votes_table,
        members_table=members_table,
    )
    settings.ensure_runtime_directories()
    return settings
