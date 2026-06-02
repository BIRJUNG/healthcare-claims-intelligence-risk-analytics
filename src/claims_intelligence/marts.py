from __future__ import annotations

import numpy as np
import pandas as pd


def build_marts(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    claim = _enrich_claims(tables)
    members = tables["dim_member"]
    providers = tables["dim_provider"]
    member_month = tables["fact_member_month"]
    episodes = _enrich_episodes(tables)
    risk = tables["fact_member_risk"].merge(members[["member_key", "member_id", "state_code", "plan_type"]], on="member_key", how="left")
    marts = {
        "mart_claims_financials": _claims_financials(claim),
        "mart_member_month_pmpm": _member_month_pmpm(member_month),
        "mart_high_cost_member": _high_cost_member(claim, members, risk),
        "mart_provider_performance": _provider_performance(claim, providers, episodes),
        "mart_readmission_queue": _readmission_queue(episodes, members, providers),
        "mart_fwa_queue": _fwa_queue(claim, providers),
        "mart_hcc_risk": _hcc_risk(risk),
        "mart_utilization": _utilization(member_month, claim),
    }
    return marts


def _enrich_claims(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return (
        tables["fact_claim"]
        .merge(tables["dim_member"], on="member_key", how="left")
        .merge(tables["dim_provider"], on="provider_key", how="left", suffixes=("_member", "_provider"))
        .merge(tables["dim_service"], on="service_key", how="left")
        .merge(tables["dim_diagnosis"], left_on="primary_diagnosis_key", right_on="diagnosis_key", how="left")
    )


def _enrich_episodes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    episode = tables["fact_inpatient_episode"]
    if episode.empty:
        return episode.copy()
    return (
        episode.merge(tables["dim_member"], on="member_key", how="left")
        .merge(tables["dim_provider"], on="provider_key", how="left", suffixes=("_member", "_provider"))
        .merge(tables["dim_diagnosis"], left_on="principal_diagnosis_key", right_on="diagnosis_key", how="left")
    )


def _claims_financials(claim: pd.DataFrame) -> pd.DataFrame:
    mart = (
        claim.groupby(["paid_month_key", "state_code_member", "plan_type", "claim_type", "service_line", "risk_segment"], dropna=False)
        .agg(
            claims=("claim_key", "nunique"),
            members=("member_key", "nunique"),
            allowed_amount=("allowed_amount", "sum"),
            paid_amount=("paid_amount", "sum"),
            member_responsibility_amount=("member_responsibility_amount", "sum"),
            denied_claims=("denied_flag", "sum"),
            inpatient_claims=("inpatient_flag", "sum"),
            ed_claims=("ed_flag", "sum"),
            avoidable_ed_claims=("avoidable_ed_flag", "sum"),
        )
        .reset_index()
    )
    mart["paid_per_claim"] = mart["paid_amount"] / mart["claims"].clip(lower=1)
    mart["denial_rate"] = mart["denied_claims"] / mart["claims"].clip(lower=1)
    return mart.sort_values(["paid_month_key", "paid_amount"], ascending=[True, False])


def _member_month_pmpm(member_month: pd.DataFrame) -> pd.DataFrame:
    mart = (
        member_month.groupby(["month_key", "state_code", "plan_type", "risk_segment"], dropna=False)
        .agg(
            member_months=("eligible_flag", "sum"),
            eligible_members=("member_key", "nunique"),
            paid_amount=("paid_amount", "sum"),
        )
        .reset_index()
    )
    mart["pmpm"] = mart["paid_amount"] / mart["member_months"].clip(lower=1)
    return mart.sort_values("month_key")


def _high_cost_member(claim: pd.DataFrame, members: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    top_diag = claim.sort_values("paid_amount", ascending=False).groupby("member_key").head(1)[["member_key", "diagnosis_group", "service_line"]]
    agg = (
        claim.groupby("member_key", as_index=False)
        .agg(
            paid_12m=("paid_amount", "sum"),
            allowed_12m=("allowed_amount", "sum"),
            claim_count=("claim_key", "nunique"),
            admissions=("inpatient_flag", "sum"),
            ed_visits=("ed_flag", "sum"),
            avoidable_ed_visits=("avoidable_ed_flag", "sum"),
            denied_claims=("denied_flag", "sum"),
        )
        .merge(members, on="member_key", how="left")
        .merge(top_diag, on="member_key", how="left")
        .merge(risk[["member_key", "readmission_count", "suspected_hcc_gap_count"]], on="member_key", how="left")
    )
    agg["paid_percentile"] = agg["paid_12m"].rank(pct=True)
    agg["risk_priority_score"] = (
        0.40 * agg["paid_percentile"]
        + 0.20 * (agg["raf_like_score"].rank(pct=True))
        + 0.15 * (agg["admissions"].rank(pct=True))
        + 0.15 * (agg["ed_visits"].rank(pct=True))
        + 0.10 * (agg["suspected_hcc_gap_count"].fillna(0).rank(pct=True))
    )
    agg["priority_tier"] = pd.cut(agg["risk_priority_score"], bins=[0, 0.55, 0.75, 0.90, 1.01], labels=["Monitor", "Rising risk", "High risk", "Complex care"], include_lowest=True)
    agg["recommended_intervention"] = np.select(
        [
            agg["admissions"] > 1,
            agg["avoidable_ed_visits"] > 0,
            agg["suspected_hcc_gap_count"].fillna(0) > 0,
            agg["paid_percentile"] >= 0.90,
        ],
        [
            "Post-discharge care management",
            "ED diversion and PCP access outreach",
            "Risk adjustment chart review",
            "Complex care management review",
        ],
        default="Monitor utilization trend",
    )
    return agg.sort_values("risk_priority_score", ascending=False)


def _provider_performance(claim: pd.DataFrame, providers: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    readmit = pd.DataFrame({"provider_key": providers["provider_key"], "readmissions": 0, "episodes": 0})
    if not episodes.empty:
        readmit = episodes.groupby("provider_key", as_index=False).agg(readmissions=("readmission_within_30_days", "sum"), episodes=("episode_key", "nunique"))
    mart = (
        claim.groupby("provider_key", as_index=False)
        .agg(
            claims=("claim_key", "nunique"),
            members=("member_key", "nunique"),
            paid_amount=("paid_amount", "sum"),
            allowed_amount=("allowed_amount", "sum"),
            inpatient_claims=("inpatient_flag", "sum"),
            ed_claims=("ed_flag", "sum"),
            denied_claims=("denied_flag", "sum"),
        )
        .merge(providers, on="provider_key", how="left")
        .merge(readmit, on="provider_key", how="left")
        .fillna({"readmissions": 0, "episodes": 0})
    )
    mart["payment_per_member"] = mart["paid_amount"] / mart["members"].clip(lower=1)
    mart["services_per_member"] = mart["claims"] / mart["members"].clip(lower=1)
    mart["readmission_rate"] = mart["readmissions"] / mart["episodes"].clip(lower=1)
    mart["peer_cost_percentile"] = mart.groupby("specialty_group")["payment_per_member"].rank(pct=True)
    mart["peer_utilization_percentile"] = mart.groupby("specialty_group")["services_per_member"].rank(pct=True)
    mart["provider_outlier_score"] = (
        0.38 * mart["peer_cost_percentile"]
        + 0.25 * mart["peer_utilization_percentile"]
        + 0.20 * mart["readmission_rate"].rank(pct=True)
        + 0.17 * (1 - mart["quality_score"].rank(pct=True))
    )
    mart["performance_tier"] = pd.cut(mart["provider_outlier_score"], bins=[0, 0.45, 0.70, 0.88, 1.01], labels=["Efficient", "Standard", "Watchlist", "Intervention"], include_lowest=True)
    mart["recommended_action"] = np.select(
        [
            mart["provider_outlier_score"] >= 0.88,
            mart["peer_cost_percentile"] >= 0.90,
            mart["readmission_rate"] >= 0.12,
        ],
        [
            "Provider relations review and payment integrity audit",
            "Peer cost benchmark discussion",
            "Readmission reduction workflow",
        ],
        default="Monitor peer trend",
    )
    return mart.sort_values("provider_outlier_score", ascending=False)


def _readmission_queue(episodes: pd.DataFrame, members: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=["episode_key", "member_id", "provider_name", "diagnosis_group", "readmission_within_30_days", "readmission_priority_score"])
    queue = episodes.copy()
    queue["readmission_priority_score"] = (
        0.30 * queue["raf_like_score"].rank(pct=True)
        + 0.25 * queue["length_of_stay"].rank(pct=True)
        + 0.20 * queue["readmission_index"].rank(pct=True)
        + 0.15 * queue["chronic_condition_count"].rank(pct=True)
        + 0.10 * queue["readmission_within_30_days"]
    )
    queue["priority_tier"] = pd.cut(queue["readmission_priority_score"], bins=[0, 0.55, 0.75, 0.90, 1.01], labels=["Monitor", "Follow-up", "High risk", "Immediate outreach"], include_lowest=True)
    queue["recommended_action"] = np.where(
        queue["readmission_within_30_days"].eq(1),
        "Review discharge transition and avoidable readmission drivers",
        "7-day follow-up, medication reconciliation, PCP access check",
    )
    return queue.sort_values("readmission_priority_score", ascending=False)


def _fwa_queue(claim: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    duplicate_counts = claim.groupby(["member_key", "provider_key", "service_key", "claim_from_date_key"], as_index=False).agg(similar_claim_count=("claim_key", "nunique"))
    queue = claim.merge(duplicate_counts, on=["member_key", "provider_key", "service_key", "claim_from_date_key"], how="left")
    provider_paid = claim.groupby("provider_key")["paid_amount"].sum().rank(pct=True)
    queue["provider_paid_percentile"] = queue["provider_key"].map(provider_paid)
    queue["claim_paid_percentile"] = queue["paid_amount"].rank(pct=True)
    queue["fwa_score"] = (
        0.36 * queue["claim_paid_percentile"]
        + 0.26 * queue["provider_paid_percentile"].fillna(0)
        + 0.18 * (queue["similar_claim_count"].clip(1, 4) / 4)
        + 0.12 * queue["denied_flag"]
        + 0.08 * queue["prior_auth_flag"]
    )
    queue["review_reason"] = np.select(
        [
            queue["similar_claim_count"] > 1,
            queue["claim_paid_percentile"] >= 0.98,
            queue["provider_paid_percentile"] >= 0.95,
            queue["denied_flag"] == 1,
        ],
        ["Duplicate-like same-day pattern", "High paid claim outlier", "High-volume provider payment pattern", "Denied or adjusted claim"],
        default="Composite anomaly score",
    )
    queue["recommended_action"] = np.where(queue["fwa_score"] >= 0.82, "SIU/payment integrity review", "Analyst validation")
    fields = [
        "claim_id",
        "member_id",
        "provider_name",
        "specialty_group",
        "service_line",
        "diagnosis_group",
        "paid_amount",
        "similar_claim_count",
        "fwa_score",
        "review_reason",
        "recommended_action",
    ]
    return queue[queue["fwa_score"] >= queue["fwa_score"].quantile(0.82)][fields].sort_values("fwa_score", ascending=False)


def _hcc_risk(risk: pd.DataFrame) -> pd.DataFrame:
    mart = risk.copy()
    mart["raf_percentile"] = mart["raf_like_score"].rank(pct=True)
    mart["hcc_gap_priority"] = (
        0.45 * mart["suspected_hcc_gap_count"].rank(pct=True)
        + 0.35 * mart["raf_like_score"].rank(pct=True)
        + 0.20 * mart["paid_12m"].rank(pct=True)
    )
    mart["priority_tier"] = pd.cut(mart["hcc_gap_priority"], bins=[0, 0.55, 0.75, 0.90, 1.01], labels=["Monitor", "Review", "High priority", "Immediate chart review"], include_lowest=True)
    return mart.sort_values("hcc_gap_priority", ascending=False)


def _utilization(member_month: pd.DataFrame, claim: pd.DataFrame) -> pd.DataFrame:
    util = claim.groupby("paid_month_key", as_index=False).agg(
        admissions=("inpatient_flag", "sum"),
        ed_visits=("ed_flag", "sum"),
        avoidable_ed_visits=("avoidable_ed_flag", "sum"),
        pharmacy_claims=("pharmacy_flag", "sum"),
    )
    months = member_month.groupby("month_key", as_index=False).agg(member_months=("eligible_flag", "sum"), paid_amount=("paid_amount", "sum"))
    mart = months.merge(util, left_on="month_key", right_on="paid_month_key", how="left").fillna(0)
    mart["admissions_per_1000"] = mart["admissions"] / mart["member_months"].clip(lower=1) * 1000
    mart["ed_visits_per_1000"] = mart["ed_visits"] / mart["member_months"].clip(lower=1) * 1000
    mart["avoidable_ed_per_1000"] = mart["avoidable_ed_visits"] / mart["member_months"].clip(lower=1) * 1000
    mart["pmpm"] = mart["paid_amount"] / mart["member_months"].clip(lower=1)
    return mart.sort_values("month_key")

