-- Representative mart SQL for the Healthcare Claims Intelligence platform.
-- The Python pipeline builds equivalent pandas marts and exports them to SQLite/CSV.

WITH provider_claims AS (
    SELECT
        c.provider_key,
        COUNT(DISTINCT c.claim_key) AS claims,
        COUNT(DISTINCT c.member_key) AS members,
        SUM(c.paid_amount) AS paid_amount,
        SUM(c.inpatient_flag) AS inpatient_claims,
        SUM(c.ed_flag) AS ed_claims
    FROM fact_claim c
    GROUP BY c.provider_key
),
provider_scored AS (
    SELECT
        p.provider_name,
        p.specialty_group,
        pc.claims,
        pc.members,
        pc.paid_amount,
        pc.paid_amount * 1.0 / NULLIF(pc.members, 0) AS payment_per_member,
        pc.claims * 1.0 / NULLIF(pc.members, 0) AS services_per_member,
        p.quality_score
    FROM provider_claims pc
    JOIN dim_provider p
        ON pc.provider_key = p.provider_key
)
SELECT *
FROM provider_scored
ORDER BY payment_per_member DESC;

