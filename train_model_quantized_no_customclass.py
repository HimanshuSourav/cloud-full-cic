import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, Dict, List, Union, Any
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

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                n_jobs=-1
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
                verbose=-1
            )
        }
        self.results = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                with Timer(f"Training {name}"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    self.results[name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }

                    mlflow.log_metrics(self.results[name])

                    input_example = X_train.iloc[:1]
                    mlflow.sklearn.log_model(
                        model, 
                        name, 
                        input_example=input_example,
                        signature=mlflow.models.signature.infer_signature(X_train, y_train)
                    )
        
        return self.results

class ModelDeployment:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        self.model_info = {}
        os.makedirs(base_path, exist_ok=True)

    def save_models(self, models: Dict, preprocessor: Any, results: Dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_path, f"model_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models with compression
        for model_name, model in models.items():
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path, compress=('zlib', 3))  # Added compression
        
        # Save preprocessor with compression
        preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path, compress=('zlib', 3))  # Added compression

        metadata = {
            "timestamp": timestamp,
            "results": results,
            "feature_names": preprocessor.feature_names.tolist() if hasattr(preprocessor, 'feature_names') else None,
            "label_encoder_classes": preprocessor.label_encoder.classes_.tolist() if hasattr(preprocessor, 'label_encoder') else None,
            "compression": "zlib level 3"  # Added compression info to metadata
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        self.model_info[timestamp] = {
            "path": save_dir,
            "metadata": metadata
        }
        
        return save_dir


def display_results(results):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=[metrics['accuracy'] for metrics in results.values()])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
            logging.info("Starting data pipeline without custom classes")

            # Load CSV in chunks
            chunk_size = 10000
            chunks = []
            total_rows = sum(1 for _ in open("../ACI-IoT-2023.csv")) - 1
            total_chunks = (total_rows // chunk_size) + 1

            with tqdm(total=total_chunks, desc="Loading data chunks") as pbar:
                for chunk in pd.read_csv("../ACI-IoT-2023.csv", chunksize=chunk_size):
                    chunks.append(chunk)
                    pbar.update(1)

            df = pd.concat(chunks, ignore_index=True)

            # Step 1: Enhance features
            df = enhance_features(df)

            # Step 2: Drop irrelevant columns
            drop_cols = ["Flow Bytes/s", "Flow Packets/s", "Flow ID", "Src IP", "Dst IP", "Timestamp"]
            df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

            # Step 3: Prepare X and y
            if 'Label' not in df.columns:
                raise ValueError("Label column is required")
            X = df.drop(columns=['Label'])
            y = df['Label']

            # Step 4: Identify types
            numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

            # Step 5: Define preprocessing pipeline
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

            # Step 6: Fit + transform
            logging.info("Fitting preprocessing pipeline")
            X_processed = preprocessor.fit_transform(X)

            # Step 7: Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Step 8: Build feature names
            feature_names = (
                numeric_cols +
                list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols))
            )
            X_df = pd.DataFrame(X_processed, columns=feature_names)

            # Step 9: Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Step 10: Train models
            trainer = ModelTrainer()
            logging.info("Training models")
            results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

            # Print results
            logging.info("\nTraining Results:")
            for model_name, metrics in results.items():
                logging.info(f"\n{model_name.upper()} Results:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value:.4f}")

            display_results(results)

            # Step 11: Save models + pipeline + encoder
            deployment = ModelDeployment()
            save_dir = deployment.save_models(
                models=trainer.models,
                preprocessor=preprocessor,
                results=results
            )

            # Save label encoder separately
            joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.joblib"))

            print(f"Models and preprocessor saved to: {save_dir}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Pipeline failed.")


if __name__ == "__main__":
    main()
