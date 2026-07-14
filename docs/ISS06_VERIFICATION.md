# ISS-06 Verification Log

Harness: [`scripts/verify_iss06.py`](../scripts/verify_iss06.py)

---

## Phase 1 — Current leaky baseline (done)

High accuracy **reproduced** on ISS-01–05 tree (~150k subsample, leaky fit-all): **accuracy 0.9991**.

Artifacts: [`iss06_results/baseline_current/`](iss06_results/baseline_current/)

---

## Phase 2 — Ablations (done)

| Config | accuracy |
|--------|----------|
| baseline_leaky | 0.9991 |
| honest_fit | 0.9991 |
| honest_no_ports | 0.9960 |
| shuffle_labels | 0.273 |

**Verdict:** A/B not the main story; **C (easy ACI separation)** likely; metric pipeline OK.

Artifacts: [`iss06_results/phase2/`](iss06_results/phase2/)

---

## Phase 3 — Trainer hardened (done)

Canonical [`train_model_quantized_no_customclass.py`](../train_model_quantized_no_customclass.py):

- Split **raw** rows first; fit preprocessor + label encoder on **train only**
- Drop `Src Port` / `Dst Port` (plus prior ID/time drops)
- Save `classification_reports.json`, `confusion_matrices.json`
- Metadata `evaluation_protocol: train_only_preprocess_v1`

[`model_bundle.save_bundle`](../model_bundle.py) extended accordingly.

---

## Phase 4 — Full-data honest retrain (done)

**Bundle:** `models/model_20260713_162252/` (does not overwrite `model_20250728_222231`)

| Model | accuracy | f1_weighted | f1_macro |
|-------|----------|-------------|----------|
| random_forest | 0.9990 | 0.9990 | 0.9079 |
| xgboost | 0.9969 | 0.9969 | 0.9035 |
| lightgbm | 0.9968 | 0.9968 | 0.8988 |

**Reading:** Weighted accuracy stays ~99.9% under honest protocol + no ports (**supports C**). Macro F1 ~0.90 is dragged down by ultra-rare **ARP Spoofing** (test support=1, F1=0). Weighted metrics alone were masking that (**supports documenting D**).

Serve smoke: `/ready` loads new bundle; `/predict` returns 200.

---

## Final ISS-06 conclusion

| Suspect | Outcome |
|---------|---------|
| A fit-all leakage | Closed in code; little metric impact on ACI subsample/full |
| B ports | Dropped for release; small Δ only |
| C easy separation | **Primary explanation** for high weighted accuracy |
| D global-only metrics | Fixed — bundles now ship per-class reports; watch macro / rare classes |
| E temporal/group | Not needed given C |

Old bundle `model_20250728_222231` remains historical (leaky protocol, ports kept). Prefer `model_20260713_162252` for serving (lexicographically latest).
