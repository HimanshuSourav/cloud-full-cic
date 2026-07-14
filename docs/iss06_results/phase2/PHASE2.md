# ISS-06 Phase 2 — Ablation comparison

Same subsample/seed as Phase 1. **No canonical trainer edits.**

## Run settings

- CSV: `/home/hsourav/ml/ACI-IoT-2023.csv`
- max_rows: `150000` (used `149999`)
- min_class_count: `20`
- random_state: `42`
- n_estimators: `100`
- dropped rare labels: `['ARP Spoofing']`

## Metrics comparison

| Config | Protocol | Ports | accuracy | f1_weighted | f1_macro | n_features |
|--------|----------|-------|----------|-------------|----------|------------|
| `baseline_leaky` | leaky_fit_all_then_split | kept | 0.999067 | 0.999066 | 0.996380 | 79 |
| `honest_fit` | honest_split_then_fit | kept | 0.999067 | 0.999066 | 0.996354 | 79 |
| `honest_no_ports` | honest_no_ports | dropped | 0.996033 | 0.996051 | 0.993448 | 77 |
| `shuffle_labels` | honest_shuffle_labels | dropped | 0.273100 | 0.255352 | 0.088913 | 77 |

## Interpretation

- **Suspect A: not strongly supported** on this subsample — accuracy drop leaky→honest = 0.0000 (threshold 0.01).
- **Suspect B: not strongly supported** on this subsample — drop honest→no_ports = 0.0030.
  - Ports still carry RF importance under honest fit: Src Port=0.077, Dst Port=0.051
- **Metric pipeline sanity: OK** — shuffle_labels accuracy = 0.2731 (uniform chance ≈ 0.091; majority prior ≈ 0.358). Accuracy collapsed vs ~0.999 labeled runs.
- **Suspect C (easy separation): plausible** — honest_no_ports accuracy still 0.9960.

## Per-config artifacts

Under `docs/iss06_results/phase2/` each config has `summary.json`, `classification_report.json`, `confusion_matrix.json`.

## Next

Phase 3: harden canonical trainer from confirmed suspects; Phase 4: full-data honest retrain.
