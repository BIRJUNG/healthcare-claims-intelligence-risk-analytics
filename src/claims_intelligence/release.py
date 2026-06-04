from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .models import ModelOutputs


def write_release_manifest(
    config: PipelineConfig,
    all_tables: dict[str, pd.DataFrame],
    quality_report: pd.DataFrame,
    model_outputs: ModelOutputs,
) -> Path:
    manifest_path = config.reports_dir / "release_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    quality_failures = int((quality_report["status"] == "FAIL").sum())
    manifest = {
        "project": "Healthcare Claims Intelligence & Risk Analytics",
        "portfolio_owner": "Birjung Thapa",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_policy": "Synthetic or approved de-identified claims data only; do not publish PHI.",
        "deployment": {
            "github_pages_url": "https://birjung.github.io/healthcare-claims-intelligence-risk-analytics/",
            "dashboard_html": _relative(config.dashboard_path, config.project_root),
            "static_site_artifact": "dist/index.html",
            "workflow": ".github/workflows/deploy-pages.yml",
        },
        "outputs": {
            "sqlite_warehouse": _relative(config.sqlite_path, config.project_root),
            "executive_summary": _relative(config.summary_path, config.project_root),
            "model_card": _relative(config.model_card_path, config.project_root),
            "data_quality_report": _relative(config.reports_dir / "data_quality_report.csv", config.project_root),
            "csv_export_dir": _relative(config.data_processed_dir, config.project_root),
            "sql_analytics_summary": _relative(config.reports_dir / "sql_analytics_summary.json", config.project_root),
            "sql_analytics_dir": _relative(config.reports_dir / "sql_analytics", config.project_root),
        },
        "quality": {
            "status": "PASS" if quality_failures == 0 else "FAIL",
            "checks": int(len(quality_report)),
            "failures": quality_failures,
        },
        "record_counts": {name: int(len(frame)) for name, frame in sorted(all_tables.items())},
        "model_metrics": _json_ready(model_outputs.metrics),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
