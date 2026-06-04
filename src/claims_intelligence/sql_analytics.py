from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import PipelineConfig


DEFAULT_SQL_PATH = Path("sql") / "05_analytics" / "healthcare_claims_risk_queries.sql"


def run_sql_analytics(config: PipelineConfig, sql_path: Path | None = None) -> Path:
    query_path = _resolve_sql_path(config.project_root, sql_path or DEFAULT_SQL_PATH)
    output_dir = config.reports_dir / "sql_analytics"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    queries = _load_named_queries(query_path)
    with sqlite3.connect(config.sqlite_path) as conn:
        for name, query in queries.items():
            frame = pd.read_sql_query(query, conn)
            csv_path = output_dir / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            summary.append(
                {
                    "name": name,
                    "sql_file": _relative(query_path, config.project_root),
                    "csv_path": _relative(csv_path, config.project_root),
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                }
            )
    summary_path = config.reports_dir / "sql_analytics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _load_named_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("-- name:"):
            _store_query(queries, current_name, current_lines)
            current_name = stripped.split(":", 1)[1].strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    _store_query(queries, current_name, current_lines)
    if not queries:
        raise ValueError(f"No named SQL queries found in {path}")
    return queries


def _resolve_sql_path(project_root: Path, sql_path: Path) -> Path:
    candidate = project_root / sql_path
    if candidate.exists():
        return candidate
    package_root = Path(__file__).resolve().parents[2]
    fallback = package_root / sql_path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"SQL analytics file not found at {candidate} or {fallback}")


def _store_query(queries: dict[str, str], name: str | None, lines: list[str]) -> None:
    if not name:
        return
    query = "\n".join(lines).strip().rstrip(";")
    if query:
        queries[name] = query


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
