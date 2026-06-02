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
