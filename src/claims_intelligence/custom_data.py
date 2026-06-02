from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_generation import GeneratedDataset, _diagnoses, _episodes, _member_months, _member_risk, _services


ALIASES = {
    "claim_id": ["claim_id", "claim_number", "claim", "claim_no"],
    "member_id": ["member_id", "bene_id", "beneficiary_id", "patient_id", "subscriber_id"],
    "provider_id": ["provider_id", "npi", "billing_provider_id", "rendering_provider_id"],
    "provider_name": ["provider_name", "provider", "facility", "billing_provider"],
    "specialty_group": ["specialty_group", "specialty", "provider_specialty", "taxonomy_group"],
    "claim_type": ["claim_type", "claim_category", "setting", "place_category"],
    "service_line": ["service_line", "service", "department", "procedure_group"],
    "diagnosis_group": ["diagnosis_group", "diagnosis", "condition", "primary_diagnosis"],
    "allowed_amount": ["allowed_amount", "allowed", "contracted_amount", "approved_amount"],
    "paid_amount": ["paid_amount", "paid", "payment_amount", "medicare_payment_amount", "reimbursed_amount"],
    "member_responsibility_amount": ["member_responsibility_amount", "member_cost_share", "coinsurance", "copay"],
    "claim_from_date": ["claim_from_date", "from_date", "service_date", "claim_date"],
    "claim_thru_date": ["claim_thru_date", "through_date", "thru_date", "paid_date"],
    "inpatient_flag": ["inpatient_flag", "is_inpatient", "admission_flag"],
    "ed_flag": ["ed_flag", "emergency_flag", "is_ed"],
    "readmission_flag": ["readmission_flag", "readmission_within_30_days"],
    "state_code": ["state_code", "state", "member_state"],
    "plan_type": ["plan_type", "plan", "product", "line_of_business"],
    "chronic_condition_count": ["chronic_condition_count", "chronic_count", "condition_count"],
    "raf_like_score": ["raf_like_score", "risk_score", "raf", "risk_adjustment_score"],
}


def load_custom_claims_dataset(path: Path, plan_year: int = 2025) -> GeneratedDataset:
    raw = pd.read_csv(path)
    data = _canonicalize(raw)
    if data.empty:
        raise ValueError("Custom claims CSV has no rows.")
    data = _fill_defaults(data, plan_year)
    dim_member = _custom_members(data)
    dim_provider = _custom_providers(data)
    dim_service = _custom_services(data)
    dim_diagnosis = _custom_diagnoses(data)
    fact_claim = _custom_claims(data, dim_member, dim_provider, dim_service, dim_diagnosis, plan_year)
    fact_member_month = _member_months(dim_member, fact_claim, plan_year)
    fact_inpatient_episode = _episodes(fact_claim)
    if "readmission_flag" in data.columns and data["readmission_flag"].notna().any() and not fact_inpatient_episode.empty:
        readmission_map = dict(zip(fact_claim["claim_key"], data["readmission_flag"].fillna(0).astype(int)))
        fact_inpatient_episode["readmission_within_30_days"] = fact_inpatient_episode["claim_key"].map(readmission_map).fillna(0).astype(int)
        fact_inpatient_episode["days_to_readmission"] = np.where(
            fact_inpatient_episode["readmission_within_30_days"].eq(1),
            np.minimum(fact_inpatient_episode["days_to_readmission"], 30),
            fact_inpatient_episode["days_to_readmission"],
        )
    rng = np.random.default_rng(2025)
    fact_member_risk = _member_risk(dim_member, fact_claim, fact_inpatient_episode, plan_year, rng)
    return GeneratedDataset(
        dim_member=dim_member,
        dim_provider=dim_provider,
        dim_service=dim_service,
        dim_diagnosis=dim_diagnosis,
        fact_claim=fact_claim,
        fact_member_month=fact_member_month,
        fact_inpatient_episode=fact_inpatient_episode,
        fact_member_risk=fact_member_risk,
    )


def _canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in raw.columns}
    output = pd.DataFrame(index=raw.index)
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                output[canonical] = raw[normalized[alias]]
                break
    return output


def _fill_defaults(data: pd.DataFrame, plan_year: int) -> pd.DataFrame:
    result = data.copy()
    row_count = len(result)
    defaults = {
        "claim_id": [f"CUSTOM{i + 1:06d}" for i in range(row_count)],
        "member_id": [f"MEM{i % max(1, row_count // 3) + 1:05d}" for i in range(row_count)],
        "provider_id": [f"PRV{i % max(1, row_count // 5) + 1:05d}" for i in range(row_count)],
        "provider_name": "Custom Provider",
        "specialty_group": "Primary care",
        "claim_type": "Professional",
        "service_line": "Office visit",
        "diagnosis_group": "Preventive/other",
        "allowed_amount": 250.0,
        "paid_amount": 180.0,
        "member_responsibility_amount": 25.0,
        "claim_from_date": f"{plan_year}-01-01",
        "claim_thru_date": f"{plan_year}-01-01",
        "inpatient_flag": 0,
        "ed_flag": 0,
        "readmission_flag": 0,
        "state_code": "US",
        "plan_type": "Medicare Advantage",
        "chronic_condition_count": 1,
        "raf_like_score": 0.85,
    }
    for col, default in defaults.items():
        if col not in result.columns:
            result[col] = default
    for col in ["allowed_amount", "paid_amount", "member_responsibility_amount", "chronic_condition_count", "raf_like_score"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(defaults[col]).clip(lower=0)
    for col in ["inpatient_flag", "ed_flag", "readmission_flag"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    result["claim_from_date"] = pd.to_datetime(result["claim_from_date"], errors="coerce").fillna(pd.Timestamp(plan_year, 1, 1))
    result["claim_thru_date"] = pd.to_datetime(result["claim_thru_date"], errors="coerce").fillna(result["claim_from_date"])
    result["claim_thru_date"] = result[["claim_from_date", "claim_thru_date"]].max(axis=1)
    return result


def _custom_members(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("member_id", as_index=False).agg(
        state_code=("state_code", "first"),
        plan_type=("plan_type", "first"),
        chronic_condition_count=("chronic_condition_count", "max"),
        raf_like_score=("raf_like_score", "max"),
    )
    grouped["member_key"] = np.arange(1, len(grouped) + 1)
    grouped["age"] = np.clip(64 + grouped["chronic_condition_count"] * 5, 18, 92).astype(int)
    grouped["age_band"] = pd.cut(grouped["age"], bins=[0, 34, 49, 64, 74, 84, 120], labels=["18-34", "35-49", "50-64", "65-74", "75-84", "85+"]).astype(str)
    grouped["gender"] = "Unknown"
    grouped["county_name"] = "Custom"
    grouped["dual_eligible_flag"] = (grouped["plan_type"].astype(str).str.contains("dual", case=False)).astype(int)
    for col in ["diabetes_flag", "copd_flag", "chf_flag", "ckd_flag", "behavioral_health_flag"]:
        grouped[col] = (grouped["chronic_condition_count"] > 1).astype(int)
    grouped["hcc_count"] = np.ceil(grouped["chronic_condition_count"] * 0.9).astype(int)
    grouped["risk_segment"] = pd.cut(grouped["raf_like_score"], bins=[0, 0.7, 1.15, 1.75, 9], labels=["Low", "Moderate", "High", "Complex"], include_lowest=True).astype(str)
    grouped["eligibility_months"] = 12
    return grouped[
        [
            "member_key",
            "member_id",
            "age",
            "age_band",
            "gender",
            "state_code",
            "county_name",
            "plan_type",
            "dual_eligible_flag",
            "chronic_condition_count",
            "diabetes_flag",
            "copd_flag",
            "chf_flag",
            "ckd_flag",
            "behavioral_health_flag",
            "hcc_count",
            "raf_like_score",
            "risk_segment",
            "eligibility_months",
        ]
    ]


def _custom_providers(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("provider_id", as_index=False).agg(
        provider_name=("provider_name", "first"),
        specialty_group=("specialty_group", "first"),
        state_code=("state_code", "first"),
    )
    grouped["provider_key"] = np.arange(1, len(grouped) + 1)
    grouped["provider_type"] = np.where(grouped["specialty_group"].astype(str).str.contains("facility|hospital", case=False), "Facility", "Professional")
    grouped["market"] = "Custom"
    grouped["network_tier"] = "Standard"
    grouped["quality_score"] = 78.0
    grouped["readmission_index"] = 1.0
    return grouped[
        ["provider_key", "provider_id", "provider_name", "specialty_group", "provider_type", "state_code", "market", "network_tier", "quality_score", "readmission_index"]
    ]


def _custom_services(data: pd.DataFrame) -> pd.DataFrame:
    base = _services()[["claim_type", "service_line", "place_of_service", "inpatient_default_flag", "ed_default_flag", "base_allowed_amount"]]
    custom = data[["claim_type", "service_line"]].drop_duplicates().copy()
    custom = custom.merge(base, on=["claim_type", "service_line"], how="left")
    custom["place_of_service"] = custom["place_of_service"].fillna(custom["claim_type"])
    custom["inpatient_default_flag"] = custom["inpatient_default_flag"].fillna(custom["claim_type"].astype(str).str.contains("inpatient", case=False).astype(int))
    custom["ed_default_flag"] = custom["ed_default_flag"].fillna(custom["service_line"].astype(str).str.contains("emergency|ed", case=False).astype(int))
    custom["base_allowed_amount"] = custom["base_allowed_amount"].fillna(250.0)
    custom["service_key"] = np.arange(1, len(custom) + 1)
    return custom[["service_key", "claim_type", "service_line", "place_of_service", "inpatient_default_flag", "ed_default_flag", "base_allowed_amount"]]


def _custom_diagnoses(data: pd.DataFrame) -> pd.DataFrame:
    ref = _diagnoses()[["diagnosis_group", "diagnosis_code", "hcc_category"]]
    custom = data[["diagnosis_group"]].drop_duplicates().copy().merge(ref, on="diagnosis_group", how="left")
    custom["diagnosis_code"] = custom["diagnosis_code"].fillna("CUSTOM")
    custom["hcc_category"] = custom["hcc_category"].fillna("HCC0")
    custom["diagnosis_key"] = np.arange(1, len(custom) + 1)
    return custom[["diagnosis_key", "diagnosis_code", "diagnosis_group", "hcc_category"]]


def _custom_claims(
    data: pd.DataFrame,
    members: pd.DataFrame,
    providers: pd.DataFrame,
    services: pd.DataFrame,
    diagnoses: pd.DataFrame,
    plan_year: int,
) -> pd.DataFrame:
    member_key = dict(zip(members["member_id"], members["member_key"]))
    provider_key = dict(zip(providers["provider_id"], providers["provider_key"]))
    service_key = {(r.claim_type, r.service_line): r.service_key for r in services.itertuples(index=False)}
    diagnosis_key = dict(zip(diagnoses["diagnosis_group"], diagnoses["diagnosis_key"]))
    result = data.copy()
    result["claim_key"] = np.arange(1, len(result) + 1)
    result["member_key"] = result["member_id"].map(member_key)
    result["provider_key"] = result["provider_id"].map(provider_key)
    result["service_key"] = [service_key[(row.claim_type, row.service_line)] for row in result.itertuples()]
    result["primary_diagnosis_key"] = result["diagnosis_group"].map(diagnosis_key)
    result["claim_from_date_key"] = pd.to_datetime(result["claim_from_date"]).dt.strftime("%Y%m%d").astype(int)
    result["claim_thru_date_key"] = pd.to_datetime(result["claim_thru_date"]).dt.strftime("%Y%m%d").astype(int)
    result["paid_month_key"] = pd.to_datetime(result["claim_from_date"]).dt.strftime("%Y%m").astype(int)
    result["length_of_stay"] = (pd.to_datetime(result["claim_thru_date"]) - pd.to_datetime(result["claim_from_date"])).dt.days.clip(lower=0).astype(int)
    result["avoidable_ed_flag"] = ((result["ed_flag"] == 1) & (result["paid_amount"] < 1800)).astype(int)
    result["pharmacy_flag"] = result["claim_type"].astype(str).str.contains("pharmacy", case=False).astype(int)
    result["prior_auth_flag"] = ((result["inpatient_flag"] == 1) | (result["allowed_amount"] > 3500)).astype(int)
    result["denied_flag"] = (result["paid_amount"] <= 0).astype(int)
    return result[
        [
            "claim_key",
            "claim_id",
            "member_key",
            "provider_key",
            "service_key",
            "primary_diagnosis_key",
            "claim_from_date_key",
            "claim_thru_date_key",
            "paid_month_key",
            "allowed_amount",
            "paid_amount",
            "member_responsibility_amount",
            "length_of_stay",
            "inpatient_flag",
            "ed_flag",
            "avoidable_ed_flag",
            "pharmacy_flag",
            "prior_auth_flag",
            "denied_flag",
        ]
    ]

