import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, Dict, List, Union, Any, Optional
import warnings
import gc
import time
from tqdm import tqdm
import mlflow
import joblib
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Filter out the specific deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message="'Series.swapaxes' is deprecated")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Timer:
    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        logging.info(f"{self.name} completed in {timedelta(seconds=int(elapsed_time))}")

class ProgressLogger:
    def __init__(self, total_steps):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        
    def update(self, step_name):
        self.current_step += 1
        elapsed_time = time.time() - self.start_time
        eta = (elapsed_time / self.current_step) * (self.total_steps - self.current_step)
        
        logging.info(
            f"Step {self.current_step}/{self.total_steps}: {step_name} | "
            f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
            f"ETA: {timedelta(seconds=int(eta))}"
        )

from model_bundle import save_bundle


class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42,
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42,
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                n_jobs=-1,
                min_child_samples=20,
                min_split_gain=0.0,
                max_depth=15,
                num_leaves=31,
                learning_rate=0.1,
                colsample_bytree=0.8,
                subsample=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
                random_state=42,
            )
        }
        self.results = {}
        self.classification_reports = {}
        self.confusion_matrices = {}

    def train_and_evaluate(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        label_names: Optional[List[str]] = None,
    ):
        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                with Timer(f"Training {name}"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    self.results[name] = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(
                            precision_score(
                                y_test, y_pred, average='weighted', zero_division=0
                            )
                        ),
                        'recall': float(
                            recall_score(
                                y_test, y_pred, average='weighted', zero_division=0
                            )
                        ),
                        'f1': float(
                            f1_score(
                                y_test, y_pred, average='weighted', zero_division=0
                            )
                        ),
                        'f1_macro': float(
                            f1_score(y_test, y_pred, average='macro', zero_division=0)
                        ),
                    }

                    labels = list(range(len(label_names))) if label_names else None
                    self.classification_reports[name] = classification_report(
                        y_test,
                        y_pred,
                        labels=labels,
                        target_names=label_names,
                        output_dict=True,
                        zero_division=0,
                    )
                    self.confusion_matrices[name] = {
                        "labels": label_names,
                        "matrix": confusion_matrix(
                            y_test, y_pred, labels=labels
                        ).tolist(),
                    }

                    mlflow.log_metrics(self.results[name])

                    input_example = X_train.iloc[:1]
                    mlflow.sklearn.log_model(
                        model,
                        name,
                        input_example=input_example,
                        signature=mlflow.models.signature.infer_signature(
                            X_train, y_train
                        ),
                    )

        return self.results


class ModelDeployment:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        self.model_info = {}
        os.makedirs(base_path, exist_ok=True)

    def save_models(
        self,
        models: Dict,
        preprocessor: Any,
        results: Dict,
        label_encoder: Any = None,
        feature_names: Any = None,
        input_feature_names: Any = None,
        dropped_columns: Any = None,
        classification_reports: Any = None,
        confusion_matrices: Any = None,
        evaluation_protocol: str = "train_only_preprocess_v1",
    ) -> str:
        if label_encoder is None:
            raise ValueError("label_encoder is required for the ISS-03 serve contract")
        save_dir = save_bundle(
            models=models,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            results=results,
            base_path=self.base_path,
            feature_names=feature_names,
            input_feature_names=input_feature_names,
            evaluation_protocol=evaluation_protocol,
            dropped_columns=dropped_columns,
            classification_reports=classification_reports,
            confusion_matrices=confusion_matrices,
        )
        self.model_info[os.path.basename(save_dir)] = {"path": save_dir}
        return save_dir


def display_results(results):
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=list(results.keys()),
            y=[metrics['accuracy'] for metrics in results.values()],
        )
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xlabel("Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out = os.path.join("models", "latest_accuracy_comparison.png")
        os.makedirs("models", exist_ok=True)
        plt.savefig(out)
        plt.close()
        logging.info("Saved accuracy comparison plot to %s", out)
    except Exception as e:
        logging.warning("Could not render results plot: %s", e)


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
        df['packet_ratio'] = np.divide(
            df['Total Fwd Packets'],
            df['Total Backward Packets'].replace(0, 1),
            out=np.zeros(len(df), dtype=float),
            where=df['Total Backward Packets'] != 0
        )

    if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
        df['byte_ratio'] = np.divide(
            df['Total Length of Fwd Packets'],
            df['Total Length of Bwd Packets'].replace(0, 1),
            out=np.zeros(len(df), dtype=float),
            where=df['Total Length of Bwd Packets'] != 0
        )

    return df


def main():
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        with Timer("Total execution"):
            logging.info(
                "Starting data pipeline (ISS-06 honest protocol: "
                "split raw → fit preprocess on train only; drop Src/Dst Port)"
            )

            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ACI-IoT-2023.csv"))
            if not os.path.isfile(csv_path):
                csv_path = "../ACI-IoT-2023.csv"

            chunk_size = 10000
            chunks = []
            total_rows = sum(1 for _ in open(csv_path)) - 1
            total_chunks = (total_rows // chunk_size) + 1

            with tqdm(total=total_chunks, desc="Loading data chunks") as pbar:
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    pbar.update(1)

            df = pd.concat(chunks, ignore_index=True)

            # Drop ultra-rare classes that break stratified splits (< 2 samples).
            counts = df["Label"].value_counts()
            rare = counts[counts < 2].index.tolist()
            if rare:
                logging.warning("Dropping classes with <2 rows before stratify: %s", rare)
                df = df[~df["Label"].isin(rare)].reset_index(drop=True)

            df = enhance_features(df)

            drop_cols = [
                "Flow Bytes/s",
                "Flow Packets/s",
                "Flow ID",
                "Src IP",
                "Dst IP",
                "Timestamp",
                # ISS-06 / Phase 2: ports are predictive but optional leaky IDs in lab data
                "Src Port",
                "Dst Port",
            ]
            present_drops = [col for col in drop_cols if col in df.columns]
            df.drop(columns=present_drops, inplace=True)

            if 'Label' not in df.columns:
                raise ValueError("Label column is required")
            X = df.drop(columns=['Label'])
            y = df['Label']

            # ISS-06: split RAW rows first, then fit preprocessor on train only.
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train_raw)
            y_test = label_encoder.transform(y_test_raw)
            label_names = [str(c) for c in label_encoder.classes_]

            numeric_cols = X_train_raw.select_dtypes(
                include=["int64", "float64", "int32", "float32"]
            ).columns.tolist()
            categorical_cols = X_train_raw.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
            leftover = [
                c
                for c in X_train_raw.columns
                if c not in numeric_cols and c not in categorical_cols
            ]
            numeric_cols.extend(leftover)

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8))
            ])
            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ])

            logging.info(
                "Fitting preprocessing pipeline on train only (%s rows)",
                f"{len(X_train_raw):,}",
            )
            X_train_p = preprocessor.fit_transform(X_train_raw)
            X_test_p = preprocessor.transform(X_test_raw)

            if categorical_cols:
                cat_out = list(
                    preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
                        categorical_cols
                    )
                )
            else:
                cat_out = []
            feature_names = list(numeric_cols) + cat_out
            X_train = pd.DataFrame(X_train_p, columns=feature_names)
            X_test = pd.DataFrame(X_test_p, columns=feature_names)

            trainer = ModelTrainer()
            logging.info("Training models")
            results = trainer.train_and_evaluate(
                X_train, X_test, y_train, y_test, label_names=label_names
            )

            logging.info("\nTraining Results:")
            for model_name, metrics in results.items():
                logging.info(f"\n{model_name.upper()} Results:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value:.4f}")

            display_results(results)

            deployment = ModelDeployment()
            save_dir = deployment.save_models(
                models=trainer.models,
                preprocessor=preprocessor,
                results=results,
                label_encoder=label_encoder,
                feature_names=feature_names,
                input_feature_names=list(X.columns),
                dropped_columns=present_drops,
                classification_reports=trainer.classification_reports,
                confusion_matrices=trainer.confusion_matrices,
                evaluation_protocol="train_only_preprocess_v1",
            )

            print(f"Models and preprocessor saved to: {save_dir}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Pipeline failed.")
        raise


if __name__ == "__main__":
    main()
