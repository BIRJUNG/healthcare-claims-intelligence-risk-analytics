from __future__ import annotations

from pathlib import Path


def write_project_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs = {
        "data_sources.md": DATA_SOURCES,
        "architecture.md": ARCHITECTURE,
        "kpi_dictionary.md": KPI_DICTIONARY,
        "data_dictionary.md": DATA_DICTIONARY,
        "dashboard_spec.md": DASHBOARD_SPEC,
        "custom_data_guide.md": CUSTOM_DATA_GUIDE,
    }
    for name, content in docs.items():
        (docs_dir / name).write_text(content.strip() + "\n", encoding="utf-8")


DATA_SOURCES = """
# Data Sources

The default build uses synthetic Medicare-style claims generated locally. The simulation is designed for portfolio demonstration and does not contain PHI.

Reference sources used for domain design:

- CMS Medicare Claims Synthetic Public Use Files
- CMS Synthetic Medicare Enrollment, Fee-for-Service Claims, and Prescription Drug Event PUF
- CMS Risk Adjustment and HCC documentation
- CMS Hospital Readmissions Reduction Program context
- CMS Physician and Other Practitioners utilization files

Custom CSV data can be loaded with `--custom-claims`.
"""

ARCHITECTURE = """
# Architecture

The platform follows a practical payer analytics architecture:

1. Synthetic or custom source claims
2. Conformed dimensions and facts
3. SQLite warehouse
4. Analytics marts
5. ML score tables
6. Quality report
7. Standalone dashboard
8. Static deployment artifact
"""

KPI_DICTIONARY = """
# KPI Dictionary

| KPI | Definition |
|---|---|
| PMPM | Paid amount divided by eligible member months |
| Paid amount | Final payer reimbursement amount |
| Allowed amount | Contracted or approved allowed amount |
| High-cost member | Member in top cost tier by annual paid amount or model probability |
| Admissions per 1,000 | Inpatient admissions divided by member months multiplied by 1,000 |
| ED visits per 1,000 | ED visits divided by member months multiplied by 1,000 |
| Readmission rate | Inpatient episodes with next admission within 30 days divided by inpatient episodes |
| RAF-like score | Synthetic risk adjustment score inspired by HCC methodology |
| Provider outlier score | Composite score using peer cost, utilization, quality, and readmission patterns |
| FWA score | Composite payment-integrity anomaly score |
"""

DATA_DICTIONARY = """
# Data Dictionary

| Table | Grain | Purpose |
|---|---|---|
| `dim_member` | One row per member | Demographics, plan, chronic burden, risk segment |
| `dim_provider` | One row per provider | Provider identity, specialty, market, quality context |
| `dim_service` | One row per service line | Claim type and place of service |
| `dim_diagnosis` | One row per diagnosis group | Diagnosis and HCC-like category |
| `fact_claim` | One row per claim | Paid, allowed, service, diagnosis, utilization flags |
| `fact_member_month` | One row per eligible member month | PMPM denominator |
| `fact_inpatient_episode` | One row per inpatient claim | Readmission logic |
| `fact_member_risk` | One row per member-year | HCC, RAF-like score, care management risk |
"""

DASHBOARD_SPEC = """
# Dashboard Specification

The dashboard includes:

- executive overview KPIs
- PMPM trend
- utilization trend
- risk segment cost concentration
- provider performance matrix
- high-cost member queue
- readmission queue
- FWA queue
- HCC gap worklist
- governance and model metrics
"""

CUSTOM_DATA_GUIDE = """
# Custom Data Guide

Run:

```powershell
python scripts\\run_healthcare_claims_pipeline.py --custom-claims data\\raw\\custom_claims_template.csv
```

Recommended columns:

```text
claim_id,member_id,provider_id,provider_name,specialty_group,claim_type,service_line,diagnosis_group,allowed_amount,paid_amount,claim_from_date,claim_thru_date,state_code,plan_type
```

Optional columns:

```text
member_responsibility_amount,inpatient_flag,ed_flag,readmission_flag,chronic_condition_count,raf_like_score
```

Do not upload PHI to a public repository.
"""


def write_model_card(model_outputs, output_path: Path) -> None:
    lines = ["# Model Cards", ""]
    for name, metrics in model_outputs.metrics.items():
        lines.append(f"## {name.replace('_', ' ').title()}")
        for key, value in metrics.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.append("Models are demonstration models trained on synthetic or user-provided de-identified data. They should not be used for clinical or payment decisions without validation, governance, and compliance review.")
    output_path.write_text("\n".join(lines), encoding="utf-8")

