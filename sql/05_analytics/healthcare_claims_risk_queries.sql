-- Named SQL analytics used by the Python pipeline after the SQLite warehouse is built.
-- Each query must begin with "-- name:" so src/claims_intelligence/sql_analytics.py can execute it.

-- name: high_cost_member_interventions
SELECT
    member_id,
    state_code,
    plan_type,
    risk_segment,
    ROUND(paid_12m, 2) AS paid_12m,
    admissions,
    ed_visits,
    ROUND(risk_priority_score, 4) AS risk_priority_score,
    priority_tier,
    recommended_intervention
FROM mart_high_cost_member
ORDER BY risk_priority_score DESC
LIMIT 30;

-- name: readmission_outreach_queue
SELECT
    episode_key,
    member_id,
    provider_name,
    diagnosis_group,
    length_of_stay,
    readmission_within_30_days,
    ROUND(readmission_priority_score, 4) AS readmission_priority_score,
    recommended_action
FROM mart_readmission_queue
ORDER BY readmission_priority_score DESC
LIMIT 30;

-- name: fwa_payment_integrity_queue
SELECT
    claim_id,
    member_id,
    provider_name,
    specialty_group,
    service_line,
    ROUND(paid_amount, 2) AS paid_amount,
    ROUND(fwa_score, 4) AS fwa_score,
    review_reason,
    recommended_action
FROM mart_fwa_queue
ORDER BY fwa_score DESC, paid_amount DESC
LIMIT 30;

-- name: pmpm_market_trends
SELECT
    month_key,
    state_code,
    plan_type,
    risk_segment,
    member_months,
    ROUND(paid_amount, 2) AS paid_amount,
    ROUND(pmpm, 2) AS pmpm
FROM mart_member_month_pmpm
ORDER BY month_key, state_code, risk_segment
LIMIT 60;
