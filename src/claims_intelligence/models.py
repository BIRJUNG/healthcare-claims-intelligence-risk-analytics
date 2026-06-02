from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ModelOutputs:
    metrics: dict[str, dict[str, float | int | str]]
    score_tables: dict[str, pd.DataFrame]


def train_and_score_models(tables: dict[str, pd.DataFrame], output_dir: Path) -> ModelOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    member_scores, high_cost_metrics = _high_cost_model(tables)
    readmission_scores, readmission_metrics = _readmission_model(tables)
    provider_scores, provider_metrics = _provider_anomaly_model(tables)
    metrics = {
        "high_cost_member_model": high_cost_metrics,
        "readmission_model": readmission_metrics,
        "provider_anomaly_model": provider_metrics,
    }
    score_tables = {
        "model_high_cost_member_scores": member_scores,
        "model_readmission_scores": readmission_scores,
        "model_provider_anomaly_scores": provider_scores,
    }
    for name, frame in score_tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return ModelOutputs(metrics=metrics, score_tables=score_tables)


def _high_cost_model(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    risk = tables["fact_member_risk"].merge(tables["dim_member"][["member_key", "member_id", "state_code", "plan_type"]], on="member_key", how="left")
    threshold = risk["paid_12m"].quantile(0.90)
    risk["target_high_cost"] = (risk["paid_12m"] >= threshold).astype(int)
    features = [
        "hcc_count",
        "raf_like_score",
        "chronic_condition_count",
        "dual_eligible_flag",
        "claim_count",
        "ed_visits",
        "admissions",
        "readmission_count",
        "suspected_hcc_gap_count",
    ]
    scores, metrics = _fit_classifier(risk, features, "target_high_cost", "predicted_high_cost_probability")
    output = risk[["member_key", "member_id", "state_code", "plan_type", "paid_12m", "target_high_cost"]].copy()
    output["predicted_high_cost_probability"] = scores
    output["model_priority_tier"] = pd.cut(scores, bins=[0, 0.30, 0.55, 0.75, 1.01], labels=["Low", "Rising", "High", "Complex"], include_lowest=True).astype(str)
    return output.sort_values("predicted_high_cost_probability", ascending=False), metrics


def _readmission_model(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    episode = tables["fact_inpatient_episode"]
    if episode.empty:
        return pd.DataFrame(columns=["episode_key", "predicted_readmission_probability"]), {"status": "skipped_no_inpatient_episodes", "rows": 0}
    data = (
        episode.merge(tables["dim_member"], on="member_key", how="left")
        .merge(tables["dim_provider"], on="provider_key", how="left")
        .merge(tables["dim_diagnosis"], left_on="principal_diagnosis_key", right_on="diagnosis_key", how="left")
    )
    features = ["length_of_stay", "age", "chronic_condition_count", "hcc_count", "raf_like_score", "readmission_index", "quality_score"]
    scores, metrics = _fit_classifier(data, features, "readmission_within_30_days", "predicted_readmission_probability")
    output = data[["episode_key", "claim_key", "member_key", "provider_key", "readmission_within_30_days"]].copy()
    output["predicted_readmission_probability"] = scores
    output["readmission_risk_tier"] = pd.cut(scores, bins=[0, 0.20, 0.40, 0.65, 1.01], labels=["Low", "Moderate", "High", "Immediate outreach"], include_lowest=True).astype(str)
    return output.sort_values("predicted_readmission_probability", ascending=False), metrics


def _provider_anomaly_model(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    claim = tables["fact_claim"]
    providers = tables["dim_provider"]
    data = (
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
    )
    data["payment_per_member"] = data["paid_amount"] / data["members"].clip(lower=1)
    data["services_per_member"] = data["claims"] / data["members"].clip(lower=1)
    data["denial_rate"] = data["denied_claims"] / data["claims"].clip(lower=1)
    features = data[["payment_per_member", "services_per_member", "inpatient_claims", "ed_claims", "denial_rate", "quality_score", "readmission_index"]].fillna(0)
    if len(data) < 8:
        score = data["payment_per_member"].rank(pct=True).to_numpy()
        status = "fallback_rank_score"
    else:
        model = IsolationForest(n_estimators=120, contamination=0.12, random_state=42)
        raw = -model.fit(features).score_samples(features)
        score = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
        status = "trained_isolation_forest"
    data["provider_anomaly_score"] = score
    data["anomaly_tier"] = pd.cut(score, bins=[0, 0.45, 0.70, 0.88, 1.01], labels=["Normal", "Monitor", "Watchlist", "Audit candidate"], include_lowest=True).astype(str)
    return data.sort_values("provider_anomaly_score", ascending=False), {
        "status": status,
        "rows": int(len(data)),
        "audit_candidate_count": int((data["provider_anomaly_score"] >= 0.88).sum()),
    }


def _fit_classifier(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    score_name: str,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    x = data[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = data[target].astype(int)
    if y.nunique() < 2 or len(data) < 40:
        baseline = np.full(len(data), float(y.mean()))
        return baseline, {"status": "fallback_constant_score", "rows": int(len(data)), "positive_rate": float(y.mean())}
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=stratify)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=140, max_depth=7, min_samples_leaf=8, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    try:
        roc_auc = float(roc_auc_score(y_test, test_scores))
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_test, test_scores))
    except ValueError:
        pr_auc = float(y.mean())
    top_decile_capture = float(y.loc[pd.Series(scores, index=data.index).nlargest(max(1, len(data) // 10)).index].mean())
    return scores, {
        "status": "trained_random_forest",
        "rows": int(len(data)),
        "positive_rate": float(y.mean()),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "top_decile_capture_rate": round(top_decile_capture, 4),
        "score_name": score_name,
    }

