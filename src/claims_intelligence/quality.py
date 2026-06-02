from __future__ import annotations

import pandas as pd


def run_quality_checks(
    base_tables: dict[str, pd.DataFrame],
    marts: dict[str, pd.DataFrame],
    model_metrics: dict[str, dict[str, object]],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, value: object, threshold: str) -> None:
        checks.append({"check_name": name, "status": "PASS" if passed else "FAIL", "value": value, "threshold": threshold})

    claim = base_tables["fact_claim"]
    member = base_tables["dim_member"]
    provider = base_tables["dim_provider"]
    member_month = base_tables["fact_member_month"]
    episodes = base_tables["fact_inpatient_episode"]

    add("claim_rows_positive", len(claim) > 0, len(claim), "> 0")
    add("claim_id_unique", claim["claim_id"].is_unique, int(claim["claim_id"].nunique()), "unique")
    add("member_key_integrity", claim["member_key"].isin(member["member_key"]).all(), int(claim["member_key"].isna().sum()), "all claims map to members")
    add("provider_key_integrity", claim["provider_key"].isin(provider["provider_key"]).all(), int(claim["provider_key"].isna().sum()), "all claims map to providers")
    add("paid_amount_non_negative", (claim["paid_amount"] >= 0).all(), float(claim["paid_amount"].min()), ">= 0")
    add("allowed_amount_non_negative", (claim["allowed_amount"] >= 0).all(), float(claim["allowed_amount"].min()), ">= 0")
    add(
        "claim_date_order",
        (claim["claim_from_date_key"] <= claim["claim_thru_date_key"]).all(),
        int((claim["claim_from_date_key"] > claim["claim_thru_date_key"]).sum()),
        "from <= thru",
    )
    add(
        "member_month_unique",
        not member_month.duplicated(["member_key", "month_key"]).any(),
        int(member_month.duplicated(["member_key", "month_key"]).sum()),
        "0 duplicates",
    )
    add("member_months_positive", int(member_month["eligible_flag"].sum()) > 0, int(member_month["eligible_flag"].sum()), "> 0")
    if not episodes.empty:
        add("readmission_days_non_negative", (episodes["days_to_readmission"] >= 0).all(), int(episodes["days_to_readmission"].min()), ">= 0")
    else:
        add("readmission_days_non_negative", True, 0, "skipped no episodes")
    for name, frame in marts.items():
        add(f"{name}_not_empty", not frame.empty, len(frame), "> 0 rows")
    add("high_cost_model_ready", model_metrics.get("high_cost_member_model", {}).get("rows", 0) > 0, model_metrics.get("high_cost_member_model", {}).get("rows", 0), "> 0 rows")
    add("readmission_model_ready", "readmission_model" in model_metrics, model_metrics.get("readmission_model", {}).get("status", "missing"), "present")
    add("provider_anomaly_model_ready", model_metrics.get("provider_anomaly_model", {}).get("rows", 0) > 0, model_metrics.get("provider_anomaly_model", {}).get("rows", 0), "> 0 rows")
    return pd.DataFrame(checks)

