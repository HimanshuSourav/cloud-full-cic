# ISS-06 Phase 1 — Current leaky baseline

Standalone reproduction of the **current** training protocol (`fit` preprocessor on all rows → then train/test split) on the ISS-01–05 codebase. **No trainer file changes.**

## Run settings

- CSV: `/home/hsourav/ml/ACI-IoT-2023.csv`
- max_rows: `150000`
- min_class_count: `20`
- random_state: `42`
- n_estimators: `100`
- rows after filters: `149999`
- dropped rare labels: `['ARP Spoofing']`
- dropped feature cols: `['Flow Bytes/s', 'Flow Packets/s', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']`

## Metrics (test set)

| Metric | Value |
|--------|-------|
| accuracy | 0.999067 |
| f1_weighted | 0.999066 |
| f1_macro | 0.996380 |
| precision_weighted | 0.999067 |
| recall_weighted | 0.999067 |

Train/test sizes: 119999 / 30000; features: 79.

## Port importances (if present)

| Feature | Importance |
|---------|------------|
| Src Port | 0.075659 |
| Dst Port | 0.046379 |

## Top 10 feature importances

| Feature | Importance |
|---------|------------|
| Idle Max | 0.085176 |
| Src Port | 0.075659 |
| Idle Min | 0.075648 |
| Idle Mean | 0.071203 |
| Fwd Seg Size Min | 0.061217 |
| Fwd Header Length | 0.051863 |
| Dst Port | 0.046379 |
| RST Flag Count | 0.038033 |
| Average Packet Size | 0.037545 |
| FWD Init Win Bytes | 0.031139 |

## Artifacts

Written under `/home/hsourav/ml/docker-deploy/docs/iss06_results/baseline_current/`:

- `summary.json`
- `classification_report.json`
- `confusion_matrix.json`
- `PHASE1.md` (this file)

## Next

Phase 2: same subsample/seed, compare `honest_fit`, `honest_no_ports`, `shuffle_labels` without editing the canonical trainer.
