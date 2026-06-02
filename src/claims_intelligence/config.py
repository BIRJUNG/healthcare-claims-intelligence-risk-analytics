from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path
    seed: int = 42
    claim_count: int = 18_000
    member_count: int = 5_200
    provider_count: int = 180
    plan_year: int = 2025
    custom_claims_csv: Path | None = None

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def data_processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def dashboard_dir(self) -> Path:
        return self.reports_dir / "dashboard"

    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    @property
    def sqlite_path(self) -> Path:
        return self.data_processed_dir / "healthcare_claims_intelligence.db"

    @property
    def dashboard_path(self) -> Path:
        return self.dashboard_dir / "healthcare_claims_intelligence_dashboard.html"

    @property
    def summary_path(self) -> Path:
        return self.reports_dir / "executive_summary.md"

    @property
    def model_card_path(self) -> Path:
        return self.docs_dir / "model_cards.md"

