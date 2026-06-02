from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def ensure_output_dirs(project_root: Path) -> None:
    for path in [
        project_root / "data" / "processed",
        project_root / "reports",
        project_root / "reports" / "dashboard",
        project_root / "reports" / "figures",
        project_root / "docs",
        project_root / "dist",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_sqlite_database(tables: dict[str, pd.DataFrame], sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    with sqlite3.connect(sqlite_path) as conn:
        for name, frame in tables.items():
            frame.to_sql(name, conn, index=False, if_exists="replace")


def export_csv_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

