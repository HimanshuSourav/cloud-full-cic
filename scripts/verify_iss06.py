#!/usr/bin/env python3
"""ISS-06 verification harness.

Phase 1: reproduce the *current* leaky training protocol without editing
train_model_quantized_no_customclass.py.

Later phases add honest-fit / no-ports / shuffle ablations via --configs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_iss06")

DEFAULT_DROP = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
]
PORT_COLS = ["Src Port", "Dst Port"]


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Match train_model_quantized_no_customclass.enhance_features."""
    df = df.copy()
    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        df["packet_ratio"] = np.divide(
            df["Total Fwd Packets"],
            df["Total Backward Packets"].replace(0, 1),
            out=np.zeros(len(df), dtype=float),
            where=df["Total Backward Packets"] != 0,
        )
    if (
        "Total Length of Fwd Packets" in df.columns
        and "Total Length of Bwd Packets" in df.columns
    ):
        df["byte_ratio"] = np.divide(
            df["Total Length of Fwd Packets"],
            df["Total Length of Bwd Packets"].replace(0, 1),
            out=np.zeros(len(df), dtype=float),
            where=df["Total Length of Bwd Packets"] != 0,
        )
    return df


def load_dataframe(
    csv_path: Path,
    max_rows: Optional[int],
    min_class_count: int,
    random_state: int,
) -> pd.DataFrame:
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s rows, %s columns", f"{len(df):,}", len(df.columns))

    if "Label" not in df.columns:
        raise ValueError("Label column required")

    counts = df["Label"].value_counts()
    keep_labels = counts[counts >= min_class_count].index
    dropped = sorted(set(counts.index) - set(keep_labels))
    if dropped:
        logger.info(
            "Excluding rare classes (< %s rows): %s",
            min_class_count,
            dropped,
        )
        df = df[df["Label"].isin(keep_labels)].reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        # Stratified subsample for tractable Phase 1/2 runs.
        frac = max_rows / len(df)
        parts = []
        for label, group in df.groupby("Label", sort=False):
            n = max(1, int(round(len(group) * frac)))
            n = min(n, len(group))
            parts.append(group.sample(n=n, random_state=random_state))
        df = pd.concat(parts, ignore_index=True)
        # Trim slight overshoot from rounding.
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        logger.info(
            "Stratified subsample -> %s rows; label counts:\n%s",
            f"{len(df):,}",
            df["Label"].value_counts().to_string(),
        )

    return df


def prepare_xy(
    df: pd.DataFrame, drop_ports: bool
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = enhance_features(df)
    drop_cols = list(DEFAULT_DROP)
    if drop_ports:
        drop_cols.extend(PORT_COLS)
    present = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=present)
    X = df.drop(columns=["Label"])
    y = df["Label"].copy()
    return X, y, present


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Include remaining ints that are boolean-like already covered; force any leftover numeric.
    leftover = [c for c in X.columns if c not in numeric_cols and c not in categorical_cols]
    numeric_cols.extend(leftover)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def feature_names_after_fit(
    preprocessor: ColumnTransformer,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> List[str]:
    # Match current trainer naming (unprefixed numeric + onehot names).
    if categorical_cols:
        cat_names = list(
            preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
                categorical_cols
            )
        )
    else:
        cat_names = []
    return list(numeric_cols) + cat_names


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: Sequence[str],
    *,
    protocol: str,
    drop_ports: bool,
    random_state: int,
    n_estimators: int,
) -> Dict:
    model = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=random_state
    )
    logger.info("Training RandomForest (n_estimators=%s)…", n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    class_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names)))).tolist()

    importances = sorted(
        zip(feature_names, model.feature_importances_.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )
    top_importances = [
        {"feature": name, "importance": float(score)} for name, score in importances[:25]
    ]
    port_importances = [
        {"feature": name, "importance": float(score)}
        for name, score in importances
        if name in PORT_COLS
    ]

    return {
        "protocol": protocol,
        "drop_ports": drop_ports,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_names),
        "metrics": metrics_dict(y_test, y_pred),
        "label_classes": class_names,
        "classification_report": report,
        "confusion_matrix": cm,
        "top_feature_importances": top_importances,
        "port_feature_importances": port_importances,
        "artifacts": {
            "model": model,
            "label_encoder": label_encoder,
        },
    }


def run_leaky_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
) -> Dict:
    """Current trainer order: fit preprocessor on ALL X, then split, then train."""
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    logger.info("LEAKY protocol: fit_transform preprocessor on all %s rows", f"{len(X):,}")
    X_processed = preprocessor.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    names = feature_names_after_fit(preprocessor, numeric_cols, categorical_cols)
    X_df = pd.DataFrame(X_processed, columns=names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=y_encoded,
    )

    result = evaluate_rf(
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        names,
        protocol="leaky_fit_all_then_split",
        drop_ports=False,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    result["artifacts"]["preprocessor"] = preprocessor
    return result


def run_honest(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
    *,
    drop_ports: bool,
    shuffle_labels: bool,
) -> Dict:
    """Honest protocol: split raw rows first, fit preprocessor on train only."""
    y_work = y.copy()
    if shuffle_labels:
        rng = np.random.RandomState(random_state)
        y_work = pd.Series(rng.permutation(y_work.to_numpy()), index=y_work.index)
        logger.info("SHUFFLE: randomly permuted labels before split/fit")

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y_work,
        test_size=0.2,
        random_state=random_state,
        stratify=y,  # stratify on true labels so class balance matches even when shuffled
    )

    label_encoder = LabelEncoder()
    # Fit on full label set so shuffle folds never see unknown classes at transform time.
    label_encoder.fit(y_work)
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train_raw)
    logger.info(
        "HONEST protocol: fit preprocessor on train only (%s rows); drop_ports=%s",
        f"{len(X_train_raw):,}",
        drop_ports,
    )
    X_train_p = preprocessor.fit_transform(X_train_raw)
    X_test_p = preprocessor.transform(X_test_raw)
    names = feature_names_after_fit(preprocessor, numeric_cols, categorical_cols)
    X_train = pd.DataFrame(X_train_p, columns=names)
    X_test = pd.DataFrame(X_test_p, columns=names)

    protocol = "honest_split_then_fit"
    if shuffle_labels:
        protocol = "honest_shuffle_labels"
    elif drop_ports:
        protocol = "honest_no_ports"

    result = evaluate_rf(
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        names,
        protocol=protocol,
        drop_ports=drop_ports,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    result["artifacts"]["preprocessor"] = preprocessor
    result["shuffle_labels"] = shuffle_labels
    return result


CONFIG_SPECS = {
    "baseline_leaky": {"drop_ports": False, "honest": False, "shuffle": False},
    "honest_fit": {"drop_ports": False, "honest": True, "shuffle": False},
    "honest_no_ports": {"drop_ports": True, "honest": True, "shuffle": False},
    "shuffle_labels": {"drop_ports": True, "honest": True, "shuffle": True},
}


def run_config(
    config: str,
    df: pd.DataFrame,
    random_state: int,
    n_estimators: int,
) -> Tuple[Dict, List[str]]:
    if config not in CONFIG_SPECS:
        raise ValueError(f"Unknown config '{config}'. Choose from {sorted(CONFIG_SPECS)}")
    spec = CONFIG_SPECS[config]
    X, y, dropped_cols = prepare_xy(df, drop_ports=spec["drop_ports"])
    if not spec["honest"]:
        result = run_leaky_baseline(X, y, random_state, n_estimators)
    else:
        result = run_honest(
            X,
            y,
            random_state,
            n_estimators,
            drop_ports=spec["drop_ports"],
            shuffle_labels=spec["shuffle"],
        )
    result["config"] = config
    return result, dropped_cols


def save_run(out_dir: Path, result: Dict, save_models: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = result.pop("artifacts", None)

    payload = {
        k: result[k]
        for k in (
            "config",
            "protocol",
            "drop_ports",
            "shuffle_labels",
            "n_train",
            "n_test",
            "n_features",
            "metrics",
            "label_classes",
            "port_feature_importances",
            "top_feature_importances",
        )
        if k in result
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    (out_dir / "classification_report.json").write_text(
        json.dumps(result["classification_report"], indent=2) + "\n"
    )
    (out_dir / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "labels": result["label_classes"],
                "matrix": result["confusion_matrix"],
            },
            indent=2,
        )
        + "\n"
    )

    if save_models and artifacts:
        joblib.dump(artifacts["model"], out_dir / "random_forest.joblib")
        joblib.dump(artifacts["preprocessor"], out_dir / "preprocessor.joblib")
        joblib.dump(artifacts["label_encoder"], out_dir / "label_encoder.joblib")

    if artifacts is not None:
        result["artifacts"] = artifacts


def write_comparison_markdown(
    out_root: Path,
    meta: Dict,
    summaries: List[Dict],
) -> None:
    lines = [
        "# ISS-06 Phase 2 — Ablation comparison",
        "",
        "Same subsample/seed as Phase 1. **No canonical trainer edits.**",
        "",
        "## Run settings",
        "",
        f"- CSV: `{meta['csv_path']}`",
        f"- max_rows: `{meta['max_rows']}` (used `{meta['n_rows_used']}`)",
        f"- min_class_count: `{meta['min_class_count']}`",
        f"- random_state: `{meta['random_state']}`",
        f"- n_estimators: `{meta['n_estimators']}`",
        f"- dropped rare labels: `{meta['dropped_rare_labels']}`",
        "",
        "## Metrics comparison",
        "",
        "| Config | Protocol | Ports | accuracy | f1_weighted | f1_macro | n_features |",
        "|--------|----------|-------|----------|-------------|----------|------------|",
    ]
    baseline_acc = None
    for s in summaries:
        m = s["metrics"]
        ports = "dropped" if s.get("drop_ports") else "kept"
        lines.append(
            f"| `{s['config']}` | {s['protocol']} | {ports} | "
            f"{m['accuracy']:.6f} | {m['f1_weighted']:.6f} | {m['f1_macro']:.6f} | "
            f"{s['n_features']} |"
        )
        if s["config"] == "baseline_leaky":
            baseline_acc = m["accuracy"]

    lines.extend(["", "## Interpretation", ""])

    by = {s["config"]: s for s in summaries}
    if "baseline_leaky" in by and "honest_fit" in by:
        d = by["baseline_leaky"]["metrics"]["accuracy"] - by["honest_fit"]["metrics"]["accuracy"]
        if d > 0.01:
            lines.append(
                f"- **Suspect A (fit-all leakage): CONFIRMED** — accuracy drop "
                f"leaky→honest = {d:.4f}."
            )
        else:
            lines.append(
                f"- **Suspect A: not strongly supported** on this subsample — accuracy drop "
                f"leaky→honest = {d:.4f} (threshold 0.01)."
            )
    if "honest_fit" in by and "honest_no_ports" in by:
        d = by["honest_fit"]["metrics"]["accuracy"] - by["honest_no_ports"]["metrics"]["accuracy"]
        if d > 0.01:
            lines.append(
                f"- **Suspect B (ports): CONFIRMED** — accuracy drop "
                f"honest→no_ports = {d:.4f}."
            )
        else:
            lines.append(
                f"- **Suspect B: not strongly supported** on this subsample — drop "
                f"honest→no_ports = {d:.4f}."
            )
            # Still note importances from honest_fit if ports present
            ports = by["honest_fit"].get("port_feature_importances") or []
            if ports:
                lines.append(
                    "  - Ports still carry RF importance under honest fit: "
                    + ", ".join(f"{p['feature']}={p['importance']:.3f}" for p in ports)
                )
    if "shuffle_labels" in by:
        sh = by["shuffle_labels"]["metrics"]["accuracy"]
        n_classes = len(by["shuffle_labels"]["label_classes"])
        chance = 1.0 / n_classes
        # Majority baseline ≈ largest class share in the subsample test-ish prior.
        majority = max(meta.get("label_counts_used", {}).values()) / max(
            meta.get("n_rows_used", 1), 1
        ) if meta.get("label_counts_used") else 0.5
        if sh <= max(majority + 0.02, chance * 4):
            lines.append(
                f"- **Metric pipeline sanity: OK** — shuffle_labels accuracy = {sh:.4f} "
                f"(uniform chance ≈ {chance:.3f}; majority prior ≈ {majority:.3f}). "
                "Accuracy collapsed vs ~0.999 labeled runs."
            )
        else:
            lines.append(
                f"- **Metric pipeline suspect** — shuffle_labels accuracy stayed high "
                f"({sh:.4f} vs majority prior ≈ {majority:.3f})."
            )

    if "honest_no_ports" in by:
        acc = by["honest_no_ports"]["metrics"]["accuracy"]
        if acc >= 0.99:
            lines.append(
                f"- **Suspect C (easy separation): plausible** — honest_no_ports accuracy "
                f"still {acc:.4f}."
            )

    lines.extend(
        [
            "",
            "## Per-config artifacts",
            "",
            f"Under `{out_root}/` each config has `summary.json`, "
            "`classification_report.json`, `confusion_matrix.json`.",
            "",
            "## Next",
            "",
            "Phase 3: harden canonical trainer from confirmed suspects; "
            "Phase 4: full-data honest retrain.",
            "",
        ]
    )
    (out_root / "PHASE2.md").write_text("\n".join(lines))
    (out_root / "comparison.json").write_text(
        json.dumps({"meta": meta, "summaries": [
            {k: s[k] for k in s if k != "artifacts" and k not in (
                "classification_report", "confusion_matrix"
            )}
            for s in summaries
        ]}, indent=2)
        + "\n"
    )


def write_phase1_markdown(out_dir: Path, meta: Dict, result: Dict) -> None:
    m = result["metrics"]
    lines = [
        "# ISS-06 Phase 1 — Current leaky baseline",
        "",
        "Standalone reproduction of the **current** training protocol "
        "(`fit` preprocessor on all rows → then train/test split) on the "
        "ISS-01–05 codebase. **No trainer file changes.**",
        "",
        "## Run settings",
        "",
        f"- CSV: `{meta['csv_path']}`",
        f"- max_rows: `{meta['max_rows']}`",
        f"- min_class_count: `{meta['min_class_count']}`",
        f"- random_state: `{meta['random_state']}`",
        f"- n_estimators: `{meta['n_estimators']}`",
        f"- rows after filters: `{meta['n_rows_used']}`",
        f"- dropped rare labels: `{meta['dropped_rare_labels']}`",
        f"- dropped feature cols: `{meta['dropped_feature_cols']}`",
        "",
        "## Metrics (test set)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| accuracy | {m['accuracy']:.6f} |",
        f"| f1_weighted | {m['f1_weighted']:.6f} |",
        f"| f1_macro | {m['f1_macro']:.6f} |",
        f"| precision_weighted | {m['precision_weighted']:.6f} |",
        f"| recall_weighted | {m['recall_weighted']:.6f} |",
        "",
        f"Train/test sizes: {result['n_train']} / {result['n_test']}; "
        f"features: {result['n_features']}.",
        "",
        "## Port importances (if present)",
        "",
    ]
    if result["port_feature_importances"]:
        lines.append("| Feature | Importance |")
        lines.append("|---------|------------|")
        for row in result["port_feature_importances"]:
            lines.append(f"| {row['feature']} | {row['importance']:.6f} |")
    else:
        lines.append("_No Src/Dst Port columns in the fitted feature set._")

    lines.extend(
        [
            "",
            "## Top 10 feature importances",
            "",
            "| Feature | Importance |",
            "|---------|------------|",
        ]
    )
    for row in result["top_feature_importances"][:10]:
        lines.append(f"| {row['feature']} | {row['importance']:.6f} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"Written under `{out_dir}/`:",
            "",
            "- `summary.json`",
            "- `classification_report.json`",
            "- `confusion_matrix.json`",
            "- `PHASE1.md` (this file)",
            "",
        ]
    )
    (out_dir / "PHASE1.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ISS-06 verification harness")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "ACI-IoT-2023.csv",
        help="Path to ACI-IoT-2023.csv",
    )
    p.add_argument("--max-rows", type=int, default=150_000)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: docs/iss06_results/phase2 or baseline_current)",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["baseline_leaky"],
        choices=sorted(CONFIG_SPECS),
        help="Configs to run",
    )
    p.add_argument("--save-models", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        alt = Path(__file__).resolve().parents[1] / ".." / "ACI-IoT-2023.csv"
        alt = alt.resolve()
        if alt.is_file():
            args.csv = alt
        else:
            raise FileNotFoundError(f"CSV not found: {args.csv}")

    if args.out_dir is None:
        root = Path(__file__).resolve().parents[1] / "docs" / "iss06_results"
        phase2_set = {"honest_fit", "honest_no_ports", "shuffle_labels"}
        if phase2_set.intersection(args.configs) or len(args.configs) > 1:
            args.out_dir = root / "phase2"
        else:
            args.out_dir = root / "baseline_current"

    label_counts_full = pd.read_csv(args.csv, usecols=["Label"])["Label"].value_counts()
    dropped_rare = sorted(
        label_counts_full[label_counts_full < args.min_class_count].index.tolist()
    )

    df = load_dataframe(
        args.csv,
        max_rows=args.max_rows,
        min_class_count=args.min_class_count,
        random_state=args.random_state,
    )

    meta = {
        "csv_path": str(args.csv),
        "max_rows": args.max_rows,
        "min_class_count": args.min_class_count,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "n_rows_used": len(df),
        "dropped_rare_labels": dropped_rare,
        "label_counts_used": df["Label"].value_counts().astype(int).to_dict(),
        "configs": list(args.configs),
    }

    summaries: List[Dict] = []
    for config in args.configs:
        logger.info("=== Running config: %s ===", config)
        result, dropped_cols = run_config(
            config, df, args.random_state, args.n_estimators
        )
        config_dir = args.out_dir / config
        run_meta = {**meta, "config": config, "dropped_feature_cols": dropped_cols}
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")
        save_run(config_dir, result, save_models=args.save_models)
        if config == "baseline_leaky" and len(args.configs) == 1:
            write_phase1_markdown(config_dir, run_meta, result)
        logger.info("%s metrics: %s", config, json.dumps(result["metrics"]))
        summaries.append({k: v for k, v in result.items() if k != "artifacts"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if len(summaries) > 1 or any(c != "baseline_leaky" for c in args.configs):
        write_comparison_markdown(args.out_dir, meta, summaries)
        logger.info("Wrote Phase 2 comparison to %s", args.out_dir / "PHASE2.md")


if __name__ == "__main__":
    main()
