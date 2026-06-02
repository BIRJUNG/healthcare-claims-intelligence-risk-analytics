from __future__ import annotations

import shutil
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "reports" / "dashboard" / "healthcare_claims_intelligence_dashboard.html"
    dist = root / "dist"
    target = dist / "index.html"
    if not source.exists():
        raise SystemExit("Dashboard not found. Run python scripts/run_healthcare_claims_pipeline.py first.")
    dist.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Built static site: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

