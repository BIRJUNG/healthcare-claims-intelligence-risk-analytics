from __future__ import annotations

import sqlite3

from claims_intelligence.config import PipelineConfig
from claims_intelligence.data_generation import generate_dataset
from claims_intelligence.marts import build_marts
from claims_intelligence.models import train_and_score_models
from claims_intelligence.pipeline import run_pipeline
from claims_intelligence.quality import run_quality_checks


def test_generated_dataset_builds_claims_marts_and_models(tmp_path):
    generated = generate_dataset(seed=7, member_count=700, provider_count=60, claim_count=2_400)
    tables = generated.as_tables()
    marts = build_marts(tables)
    model_outputs = train_and_score_models(tables, tmp_path)
    quality = run_quality_checks(tables, marts, model_outputs.metrics)

    assert len(tables["fact_claim"]) == 2_400
    assert not tables["fact_member_month"].empty
    assert not marts["mart_high_cost_member"].empty
    assert not marts["mart_provider_performance"].empty
    assert not model_outputs.score_tables["model_high_cost_member_scores"].empty
    assert (quality["status"] == "FAIL").sum() == 0


def test_end_to_end_pipeline_writes_outputs(tmp_path):
    config = PipelineConfig(project_root=tmp_path, seed=11, claim_count=2_800, member_count=850, provider_count=70)
    result = run_pipeline(config)

    assert result["quality_failures"] == 0
    assert config.sqlite_path.exists()
    assert config.dashboard_path.exists()
    assert config.summary_path.exists()
    assert (config.data_processed_dir / "mart_member_month_pmpm.csv").exists()
    assert (config.data_processed_dir / "model_high_cost_member_scores.csv").exists()

    conn = sqlite3.connect(config.sqlite_path)
    try:
        claim_count = conn.execute("SELECT COUNT(*) FROM fact_claim").fetchone()[0]
        mart_count = conn.execute("SELECT COUNT(*) FROM mart_provider_performance").fetchone()[0]
    finally:
        conn.close()

    assert claim_count == 2_800
    assert mart_count > 0


def test_pipeline_accepts_custom_claims_csv(tmp_path):
    custom = tmp_path / "custom_claims.csv"
    custom.write_text(
        "\n".join(
            [
                "claim_id,member_id,provider_id,provider_name,specialty_group,claim_type,service_line,diagnosis_group,allowed_amount,paid_amount,member_responsibility_amount,claim_from_date,claim_thru_date,inpatient_flag,ed_flag,readmission_flag,state_code,plan_type,chronic_condition_count,raf_like_score",
                "C1,M1,P1,Metro Hospital,Facility,Inpatient,Inpatient stay,CHF,18000,14100,800,2025-01-02,2025-01-06,1,0,1,TX,MA-PD,3,1.6",
                "C2,M1,P2,Metro PCP,Primary care,Professional,Office visit,Diabetes,250,190,20,2025-01-21,2025-01-21,0,0,0,TX,MA-PD,3,1.6",
                "C3,M2,P3,Valley ED,Emergency medicine,Outpatient,Emergency department,COPD,1600,1100,110,2025-02-10,2025-02-10,0,1,0,FL,Medicare Advantage,2,1.1",
                "C4,M3,P4,Imaging Group,Radiology,Outpatient,Imaging,Musculoskeletal,900,620,80,2025-03-01,2025-03-01,0,0,0,CA,Medicare Advantage,1,0.7",
            ]
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(project_root=tmp_path, custom_claims_csv=custom)
    result = run_pipeline(config)

    assert result["quality_failures"] == 0
    assert result["claim_count"] == 4
    assert result["member_count"] == 3
    assert config.dashboard_path.exists()

