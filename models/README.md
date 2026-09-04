# cloud-full-cic model bundles

Full CICFlowMeter feature set (~76 columns). These folders **are tracked** in the nested git repo.

## CURRENT -> model_20260713_162252_full_cic_honest

ISS-06 honest protocol: split raw first, fit preprocess on train only, Src/Dst Port dropped.

| Model | Accuracy | F1 weighted | F1 macro |
|-------|----------|-------------|----------|
| random_forest | 0.9990 | 0.9990 | 0.9079 |
| xgboost | 0.9969 | 0.9969 | 0.9035 |
| lightgbm | 0.9968 | 0.9968 | 0.8988 |

Valid only when /predict receives the full input_feature_names list (not five imputed fields).

## model_20250728_222231_full_cic_leaky

Historical leaky protocol (ports kept, fit-then-split). Kept for comparison. Do not treat as the release.
