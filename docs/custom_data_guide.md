# Custom Data Guide

Run:

```powershell
python scripts\run_healthcare_claims_pipeline.py --custom-claims data\raw\custom_claims_template.csv
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
