# Model Cards

## High Cost Member Model
- `status`: trained_random_forest
- `rows`: 5200
- `positive_rate`: 0.1
- `roc_auc`: 0.962
- `pr_auc`: 0.7589
- `top_decile_capture_rate`: 0.7269
- `score_name`: predicted_high_cost_probability

## Readmission Model
- `status`: trained_random_forest
- `rows`: 2071
- `positive_rate`: 0.04635441815548044
- `roc_auc`: 0.7233
- `pr_auc`: 0.0918
- `top_decile_capture_rate`: 0.314
- `score_name`: predicted_readmission_probability

## Provider Anomaly Model
- `status`: trained_isolation_forest
- `rows`: 180
- `audit_candidate_count`: 2

Models are demonstration models trained on synthetic or user-provided de-identified data. They should not be used for clinical or payment decisions without validation, governance, and compliance review.