from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GeneratedDataset:
    dim_member: pd.DataFrame
    dim_provider: pd.DataFrame
    dim_service: pd.DataFrame
    dim_diagnosis: pd.DataFrame
    fact_claim: pd.DataFrame
    fact_member_month: pd.DataFrame
    fact_inpatient_episode: pd.DataFrame
    fact_member_risk: pd.DataFrame

    def as_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "dim_member": self.dim_member,
            "dim_provider": self.dim_provider,
            "dim_service": self.dim_service,
            "dim_diagnosis": self.dim_diagnosis,
            "fact_claim": self.fact_claim,
            "fact_member_month": self.fact_member_month,
            "fact_inpatient_episode": self.fact_inpatient_episode,
            "fact_member_risk": self.fact_member_risk,
        }


STATES = ["CA", "TX", "FL", "NY", "PA", "OH", "GA", "NC", "MI", "AZ", "IL", "WA"]
PLAN_TYPES = ["Medicare Advantage", "MA-PD", "Dual Eligible SNP", "Employer Group Waiver"]
SPECIALTIES = [
    ("Primary care", "Professional"),
    ("Cardiology", "Professional"),
    ("Orthopedics", "Professional"),
    ("Emergency medicine", "Professional"),
    ("Radiology", "Professional"),
    ("Oncology", "Professional"),
    ("Behavioral health", "Professional"),
    ("Facility", "Facility"),
    ("Pharmacy", "Pharmacy"),
]
SERVICES = [
    ("Professional", "Office visit", "Office", 0, 0, 210),
    ("Professional", "Specialist visit", "Office", 0, 0, 480),
    ("Outpatient", "Emergency department", "Emergency", 0, 1, 1450),
    ("Outpatient", "Imaging", "Outpatient", 0, 0, 820),
    ("Outpatient", "Outpatient surgery", "Outpatient", 0, 0, 5200),
    ("Inpatient", "Inpatient stay", "Facility", 1, 0, 18500),
    ("Inpatient", "SNF/post-acute", "Post acute", 1, 0, 9300),
    ("Pharmacy", "Part D drug", "Pharmacy", 0, 0, 165),
    ("Professional", "Behavioral health", "Office", 0, 0, 260),
    ("Outpatient", "Lab/pathology", "Outpatient", 0, 0, 130),
]
DIAGNOSES = [
    ("E11", "Diabetes", "HCC18"),
    ("I50", "CHF", "HCC85"),
    ("J44", "COPD", "HCC111"),
    ("N18", "CKD", "HCC138"),
    ("F32", "Behavioral health", "HCC59"),
    ("C34", "Oncology", "HCC8"),
    ("M17", "Musculoskeletal", "HCC0"),
    ("I10", "Hypertension", "HCC0"),
    ("S72", "Fracture/trauma", "HCC170"),
    ("Z00", "Preventive/other", "HCC0"),
]


def generate_dataset(
    seed: int = 42,
    member_count: int = 5_200,
    provider_count: int = 180,
    claim_count: int = 18_000,
    plan_year: int = 2025,
) -> GeneratedDataset:
    rng = np.random.default_rng(seed)
    dim_member = _members(rng, member_count)
    dim_provider = _providers(rng, provider_count)
    dim_service = _services()
    dim_diagnosis = _diagnoses()
    fact_claim = _claims(rng, dim_member, dim_provider, dim_service, dim_diagnosis, claim_count, plan_year)
    fact_member_month = _member_months(dim_member, fact_claim, plan_year)
    fact_inpatient_episode = _episodes(fact_claim)
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


def _members(rng: np.random.Generator, member_count: int) -> pd.DataFrame:
    age = rng.integers(18, 93, member_count)
    chronic_base = rng.poisson(np.clip((age - 35) / 28, 0.05, 2.4))
    diabetes = rng.random(member_count) < np.clip(0.05 + age / 450, 0.05, 0.30)
    copd = rng.random(member_count) < np.clip(0.03 + age / 650, 0.03, 0.18)
    chf = rng.random(member_count) < np.clip(0.02 + age / 800, 0.02, 0.16)
    ckd = rng.random(member_count) < np.clip(0.02 + age / 900, 0.02, 0.14)
    behavioral = rng.random(member_count) < 0.15
    chronic = np.maximum(chronic_base, diabetes.astype(int) + copd.astype(int) + chf.astype(int) + ckd.astype(int) + behavioral.astype(int))
    dual = rng.random(member_count) < np.clip(0.12 + chronic * 0.04, 0.12, 0.36)
    hcc_count = np.maximum(chronic + rng.binomial(1, 0.28, member_count), 0)
    raf = 0.32 + (age >= 65) * 0.18 + chronic * 0.23 + dual * 0.22 + rng.normal(0, 0.12, member_count)
    raf = np.clip(raf, 0.18, 3.8).round(3)
    risk_segment = pd.cut(raf, bins=[0, 0.7, 1.15, 1.75, 9], labels=["Low", "Moderate", "High", "Complex"], include_lowest=True)
    return pd.DataFrame(
        {
            "member_key": np.arange(1, member_count + 1),
            "member_id": [f"M{100000 + i}" for i in range(member_count)],
            "age": age,
            "age_band": pd.cut(age, bins=[0, 34, 49, 64, 74, 84, 120], labels=["18-34", "35-49", "50-64", "65-74", "75-84", "85+"]),
            "gender": rng.choice(["Female", "Male"], member_count),
            "state_code": rng.choice(STATES, member_count, p=np.array([0.13, 0.12, 0.11, 0.10, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.07, 0.06])),
            "county_name": rng.choice(["Metro", "North", "South", "Central", "Coastal", "Valley"], member_count),
            "plan_type": rng.choice(PLAN_TYPES, member_count, p=[0.42, 0.34, 0.16, 0.08]),
            "dual_eligible_flag": dual.astype(int),
            "chronic_condition_count": chronic.astype(int),
            "diabetes_flag": diabetes.astype(int),
            "copd_flag": copd.astype(int),
            "chf_flag": chf.astype(int),
            "ckd_flag": ckd.astype(int),
            "behavioral_health_flag": behavioral.astype(int),
            "hcc_count": hcc_count.astype(int),
            "raf_like_score": raf,
            "risk_segment": risk_segment.astype(str),
            "eligibility_months": rng.integers(8, 13, member_count),
        }
    )


def _providers(rng: np.random.Generator, provider_count: int) -> pd.DataFrame:
    specialty = rng.choice(len(SPECIALTIES), provider_count, p=[0.26, 0.10, 0.09, 0.08, 0.09, 0.05, 0.07, 0.18, 0.08])
    names = []
    for i, idx in enumerate(specialty):
        group = SPECIALTIES[idx][0]
        suffix = "Hospital" if group == "Facility" else "Medical Group" if group == "Primary care" else "Associates"
        names.append(f"{group} {suffix} {i + 1:03d}")
    return pd.DataFrame(
        {
            "provider_key": np.arange(1, provider_count + 1),
            "provider_id": [f"P{200000 + i}" for i in range(provider_count)],
            "provider_name": names,
            "specialty_group": [SPECIALTIES[idx][0] for idx in specialty],
            "provider_type": [SPECIALTIES[idx][1] for idx in specialty],
            "state_code": rng.choice(STATES, provider_count),
            "market": rng.choice(["Urban core", "Suburban", "Rural", "Regional hub"], provider_count, p=[0.36, 0.34, 0.16, 0.14]),
            "network_tier": rng.choice(["Preferred", "Standard", "Watchlist"], provider_count, p=[0.42, 0.48, 0.10]),
            "quality_score": np.clip(rng.normal(78, 11, provider_count), 35, 99).round(2),
            "readmission_index": np.clip(rng.normal(1.0, 0.18, provider_count), 0.55, 1.75).round(3),
        }
    )


def _services() -> pd.DataFrame:
    rows = []
    for i, (claim_type, service_line, place, inpatient, ed, base_allowed) in enumerate(SERVICES, start=1):
        rows.append(
            {
                "service_key": i,
                "claim_type": claim_type,
                "service_line": service_line,
                "place_of_service": place,
                "inpatient_default_flag": inpatient,
                "ed_default_flag": ed,
                "base_allowed_amount": base_allowed,
            }
        )
    return pd.DataFrame(rows)


def _diagnoses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diagnosis_key": np.arange(1, len(DIAGNOSES) + 1),
            "diagnosis_code": [row[0] for row in DIAGNOSES],
            "diagnosis_group": [row[1] for row in DIAGNOSES],
            "hcc_category": [row[2] for row in DIAGNOSES],
        }
    )


def _claims(
    rng: np.random.Generator,
    members: pd.DataFrame,
    providers: pd.DataFrame,
    services: pd.DataFrame,
    diagnoses: pd.DataFrame,
    claim_count: int,
    plan_year: int,
) -> pd.DataFrame:
    member_weights = members["raf_like_score"].to_numpy() ** 1.25
    member_weights = member_weights / member_weights.sum()
    member_key = rng.choice(members["member_key"].to_numpy(), claim_count, p=member_weights)
    member_lookup = members.set_index("member_key")
    member_risk = member_lookup.loc[member_key, "raf_like_score"].to_numpy()
    member_chronic = member_lookup.loc[member_key, "chronic_condition_count"].to_numpy()

    service_weights = np.array([0.23, 0.14, 0.10, 0.10, 0.07, 0.08, 0.04, 0.17, 0.04, 0.03])
    service_key = rng.choice(services["service_key"].to_numpy(), claim_count, p=service_weights)
    service_lookup = services.set_index("service_key")
    base_allowed = service_lookup.loc[service_key, "base_allowed_amount"].to_numpy()
    inpatient_flag = service_lookup.loc[service_key, "inpatient_default_flag"].to_numpy()
    ed_flag = service_lookup.loc[service_key, "ed_default_flag"].to_numpy()

    provider_weights = np.ones(len(providers))
    provider_weights[providers["provider_type"].eq("Facility").to_numpy()] = 1.35
    provider_weights = provider_weights / provider_weights.sum()
    provider_key = rng.choice(providers["provider_key"].to_numpy(), claim_count, p=provider_weights)

    diagnosis_probs = np.array([0.16, 0.10, 0.09, 0.08, 0.09, 0.04, 0.14, 0.16, 0.04, 0.10])
    diagnosis_key = rng.choice(diagnoses["diagnosis_key"].to_numpy(), claim_count, p=diagnosis_probs)
    start = datetime(plan_year, 1, 1)
    offsets = rng.integers(0, 365, claim_count)
    from_dates = np.array([start + timedelta(days=int(o)) for o in offsets], dtype="datetime64[ns]")
    los = np.where(inpatient_flag == 1, rng.integers(2, 9, claim_count), 0)
    thru_dates = pd.to_datetime(from_dates) + pd.to_timedelta(los, unit="D")
    severity = np.clip(0.82 + member_risk * 0.22 + member_chronic * 0.04 + rng.lognormal(0, 0.25, claim_count), 0.6, 4.2)
    allowed = np.round(base_allowed * severity * rng.uniform(0.72, 1.28, claim_count), 2)
    paid = np.round(allowed * rng.uniform(0.68, 0.93, claim_count), 2)
    responsibility = np.round(np.maximum(allowed - paid, 0) * rng.uniform(0.18, 0.42, claim_count), 2)
    avoidable_ed = ((ed_flag == 1) & (rng.random(claim_count) < np.clip(0.12 + member_chronic * 0.03, 0.12, 0.32))).astype(int)
    prior_auth = ((inpatient_flag == 1) | (allowed > 3500)).astype(int)
    denied = (rng.random(claim_count) < np.clip(0.025 + (prior_auth * 0.018) + (allowed > 10000) * 0.015, 0.02, 0.10)).astype(int)
    paid = np.where(denied == 1, np.round(paid * rng.uniform(0, 0.35, claim_count), 2), paid)

    return pd.DataFrame(
        {
            "claim_key": np.arange(1, claim_count + 1),
            "claim_id": [f"CLM{plan_year}{i:07d}" for i in range(1, claim_count + 1)],
            "member_key": member_key,
            "provider_key": provider_key,
            "service_key": service_key,
            "primary_diagnosis_key": diagnosis_key,
            "claim_from_date_key": pd.to_datetime(from_dates).strftime("%Y%m%d").astype(int),
            "claim_thru_date_key": pd.to_datetime(thru_dates).strftime("%Y%m%d").astype(int),
            "paid_month_key": pd.to_datetime(from_dates).strftime("%Y%m").astype(int),
            "allowed_amount": allowed,
            "paid_amount": paid,
            "member_responsibility_amount": responsibility,
            "length_of_stay": los,
            "inpatient_flag": inpatient_flag.astype(int),
            "ed_flag": ed_flag.astype(int),
            "avoidable_ed_flag": avoidable_ed,
            "pharmacy_flag": (service_key == 8).astype(int),
            "prior_auth_flag": prior_auth,
            "denied_flag": denied,
        }
    )


def _member_months(members: pd.DataFrame, claims: pd.DataFrame, plan_year: int) -> pd.DataFrame:
    rows = []
    paid_by_member_month = claims.groupby(["member_key", "paid_month_key"], as_index=False)["paid_amount"].sum()
    paid_lookup = {(int(r.member_key), int(r.paid_month_key)): float(r.paid_amount) for r in paid_by_member_month.itertuples()}
    for row in members.itertuples(index=False):
        for month in range(1, int(row.eligibility_months) + 1):
            month_key = plan_year * 100 + month
            rows.append(
                {
                    "member_key": int(row.member_key),
                    "month_key": month_key,
                    "eligible_flag": 1,
                    "plan_type": row.plan_type,
                    "state_code": row.state_code,
                    "risk_segment": row.risk_segment,
                    "paid_amount": paid_lookup.get((int(row.member_key), month_key), 0.0),
                }
            )
    return pd.DataFrame(rows)


def _episodes(claims: pd.DataFrame) -> pd.DataFrame:
    inpatient = claims[claims["inpatient_flag"] == 1].sort_values(["member_key", "claim_from_date_key"]).copy()
    if inpatient.empty:
        return pd.DataFrame(
            columns=[
                "episode_key",
                "claim_key",
                "member_key",
                "provider_key",
                "admission_date_key",
                "discharge_date_key",
                "length_of_stay",
                "principal_diagnosis_key",
                "readmission_within_30_days",
                "days_to_readmission",
            ]
        )
    inpatient["next_admit"] = inpatient.groupby("member_key")["claim_from_date_key"].shift(-1)
    discharge = pd.to_datetime(inpatient["claim_thru_date_key"].astype(str), format="%Y%m%d", errors="coerce")
    next_admit_key = inpatient["next_admit"].astype("Int64").astype("string")
    next_admit = pd.to_datetime(next_admit_key, format="%Y%m%d", errors="coerce")
    days = (next_admit - discharge).dt.days
    inpatient["days_to_readmission"] = days.fillna(9999).clip(lower=0).astype(int)
    inpatient["readmission_within_30_days"] = ((days >= 0) & (days <= 30)).fillna(False).astype(int)
    return inpatient.assign(episode_key=np.arange(1, len(inpatient) + 1))[
        [
            "episode_key",
            "claim_key",
            "member_key",
            "provider_key",
            "claim_from_date_key",
            "claim_thru_date_key",
            "length_of_stay",
            "primary_diagnosis_key",
            "readmission_within_30_days",
            "days_to_readmission",
        ]
    ].rename(
        columns={
            "claim_from_date_key": "admission_date_key",
            "claim_thru_date_key": "discharge_date_key",
            "primary_diagnosis_key": "principal_diagnosis_key",
        }
    )


def _member_risk(
    members: pd.DataFrame,
    claims: pd.DataFrame,
    episodes: pd.DataFrame,
    plan_year: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    paid = claims.groupby("member_key", as_index=False).agg(
        paid_12m=("paid_amount", "sum"),
        claim_count=("claim_key", "nunique"),
        ed_visits=("ed_flag", "sum"),
        admissions=("inpatient_flag", "sum"),
    )
    if episodes.empty:
        readmits = pd.DataFrame({"member_key": members["member_key"], "readmission_count": 0})
    else:
        readmits = episodes.groupby("member_key", as_index=False).agg(readmission_count=("readmission_within_30_days", "sum"))
    risk = members[
        ["member_key", "hcc_count", "raf_like_score", "risk_segment", "chronic_condition_count", "dual_eligible_flag"]
    ].merge(paid, on="member_key", how="left").merge(readmits, on="member_key", how="left").fillna(0)
    risk["plan_year"] = plan_year
    risk["suspected_hcc_gap_count"] = np.maximum(
        0,
        (risk["chronic_condition_count"] - risk["hcc_count"] + rng.binomial(1, 0.18, len(risk))).astype(int),
    )
    risk["risk_adjustment_action"] = np.where(
        risk["suspected_hcc_gap_count"] > 0,
        "Chart review and provider documentation outreach",
        "Monitor annual wellness and chronic recapture",
    )
    return risk[
        [
            "member_key",
            "plan_year",
            "hcc_count",
            "raf_like_score",
            "risk_segment",
            "chronic_condition_count",
            "dual_eligible_flag",
            "paid_12m",
            "claim_count",
            "ed_visits",
            "admissions",
            "readmission_count",
            "suspected_hcc_gap_count",
            "risk_adjustment_action",
        ]
    ]
