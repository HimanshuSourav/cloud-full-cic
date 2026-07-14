"""Legacy custom DataPreprocessor trainer (non-canonical).

Does not write the sklearn_column_transformer_v1 serve bundle.
Canonical trainer: ../train_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

class DataPreprocessor:
    def __init__(self, batch_size=10000):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.batch_size = batch_size
        self.progress = None
        self.feature_groups = {
            'packet_features': {
                'required': {'Total Fwd Packets', 'Total Backward Packets'},
                'derived': ['packet_ratio', 'packet_rate']
            },
            'byte_features': {
                'required': {'Total Length of Fwd Packets', 'Total Length of Bwd Packets'},
                'derived': ['byte_ratio', 'byte_rate']
            },
            'flow_features': {
                'required': {'Flow Duration'},
                'derived': ['packet_rate', 'byte_rate']
            }
        }
        self.absolutely_required = {'Label'}
        self.feature_names = None
        self.cat_columns = None 

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.feature_names is None:
                raise ValueError("Preprocessor has not been fitted. Call `preprocess_data` first.")

            dtypes = self.get_column_dtypes(df)
            for col in df.columns:
                if 'float' in dtypes[col] or 'int' in dtypes[col]:
                    df[col] = self.process_numeric_column(df[col])
                else:
                    df[col] = df[col].fillna('NA')

            cols_to_drop = ["Flow Bytes/s", "Flow Packets/s", "Flow ID",
                            "Src IP", "Dst IP", "Timestamp"]
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=existing_cols)

            if self.cat_columns:
                df = pd.get_dummies(df, columns=self.cat_columns, dtype=np.int8)

            df = df.reindex(columns=self.feature_names, fill_value=0)

            scaled_data = []
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                scaled_batch = self.scaler.transform(batch)
                scaled_data.append(scaled_batch)

            return pd.DataFrame(np.vstack(scaled_data), columns=self.feature_names)

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise
            
    def get_feature_names_out(self):
        if self.feature_names is None:
            raise ValueError("Feature names are not available. Call `preprocess_data` first.")
        return self.feature_names

    def get_column_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def process_numeric_column(self, series: pd.Series) -> pd.Series:
        if series.isnull().any():
            chunk_means = []
            chunk_counts = []

            for chunk in np.array_split(series, max(1, len(series) // 10000)):
                chunk_means.append(chunk.mean())
                chunk_counts.append(len(chunk))

            overall_mean = np.average(chunk_means, weights=chunk_counts)
            series = series.fillna(overall_mean)

        return series.transpose()

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        try:
            self.progress = ProgressLogger(total_steps=5)
            
            self.progress.update("Validating input data")
            if 'Label' not in df.columns:
                raise ValueError("Label column missing from dataset")

            self.progress.update("Processing columns")
            dtypes = self.get_column_dtypes(df)
            for col in tqdm(df.columns, desc="Processing columns"):
                if 'float' in dtypes[col] or 'int' in dtypes[col]:
                    df[col] = self.process_numeric_column(df[col])
                else:
                    df[col] = df[col].fillna('NA')

            self.progress.update("Feature extraction")
            cols_to_drop = ["Flow Bytes/s", "Flow Packets/s", "Flow ID",
                          "Src IP", "Dst IP", "Timestamp"]
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=existing_cols)
            X = df.drop('Label', axis=1)
            y = df['Label'].copy()

            self.progress.update("Processing categorical variables")
            cat_columns = [col for col, dtype in dtypes.items()
                         if 'float' not in dtype and 'int' not in dtype
                         and col in X.columns]
            if cat_columns:
                X = pd.get_dummies(X, columns=cat_columns, dtype=np.int8)

            self.progress.update("Scaling features")
            # ISS-04: fit scaler on all rows (not only the first batch).
            from preproc_scale import fit_transform_scaled

            X, self.scaler = fit_transform_scaled(
                X, self.scaler, batch_size=self.batch_size
            )

            y = self.label_encoder.fit_transform(y)
            self.feature_names = X.columns
            return X, y

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            enhanced_dfs = []
            for start_idx in range(0, len(df), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(df))
                batch = df.iloc[start_idx:end_idx].copy()

                for group, features in self.feature_groups.items():
                    required_cols = features['required']
                    if all(col in batch.columns for col in required_cols):
                        if group == 'packet_features':
                            batch['packet_ratio'] = np.divide(
                                batch['Total Fwd Packets'],
                                batch['Total Backward Packets'].replace(0, 1),
                                out=np.zeros(len(batch), dtype=float),
                                where=batch['Total Backward Packets'] != 0
                            )
                        elif group == 'byte_features':
                            batch['byte_ratio'] = np.divide(
                                batch['Total Length of Fwd Packets'],
                                batch['Total Length of Bwd Packets'].replace(0, 1),
                                out=np.zeros(len(batch), dtype=float),
                                where=batch['Total Length of Bwd Packets'] != 0
                            )
                    else:
                        for derived in features['derived']:
                            batch[derived] = 0

                        missing_cols = required_cols - set(batch.columns)
                        for col in missing_cols:
                            batch[col] = 0

                enhanced_dfs.append(batch)

                del batch
                gc.collect()

            return pd.concat(enhanced_dfs, axis=0, ignore_index=True)

        except Exception as e:
            logging.error(f"Error in feature enhancement: {str(e)}")
            raise

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

#class ModelDeployment:
 #   def __init__(self, base_path: str = "models"):
  #      self.base_path = base_path
   #     self.model_info = {}
    #    os.makedirs(base_path, exist_ok=True)

   # def save_models(self, models: Dict, preprocessor: Any, results: Dict) -> str:
    #    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     #   save_dir = os.path.join(self.base_path, f"model_{timestamp}")
      #  os.makedirs(save_dir, exist_ok=True)
        
       # for model_name, model in models.items():
        #    model_path = os.path.join(save_dir, f"{model_name}.joblib")
         #   joblib.dump(model, model_path)
        
        #preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
        #joblib.dump(preprocessor, preprocessor_path)

        #metadata = {
         #   "timestamp": timestamp,
          #  "results": results,
           # "feature_names": preprocessor.feature_names.tolist() if hasattr(preprocessor, 'feature_names') else None,
            #"label_encoder_classes": preprocessor.label_encoder.classes_.tolist() if hasattr(preprocessor, 'label_encoder') else None
       # }
        
        #metadata_path = os.path.join(save_dir, "metadata.json")
        #with open(metadata_path, 'w') as f:
         #   json.dump(metadata, f, indent=4)
            
        #self.model_info[timestamp] = {
         #   "path": save_dir,
          #  "metadata": metadata
        #}
        
        #return save_dir

def display_results(results):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=[metrics['accuracy'] for metrics in results.values()])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        with Timer("Total execution"):
            logging.info("Starting data processing pipeline")
            
            chunk_size = 10000
            chunks = []
            total_rows = sum(1 for _ in open("../ACI-IoT-2023.csv")) - 1
            total_chunks = (total_rows // chunk_size) + 1
            
            with tqdm(total=total_chunks, desc="Loading data chunks") as pbar:
                for chunk in pd.read_csv("../ACI-IoT-2023.csv", chunksize=chunk_size):
                    chunks.append(chunk)
                    pbar.update(1)
            
            df = pd.concat(chunks, ignore_index=True)
            
            preprocessor = DataPreprocessor(batch_size=10000)
            trainer = ModelTrainer()
            
            logging.info("Starting feature enhancement")
            df = preprocessor.enhance_features(df)
            logging.info("Starting data preprocessing")
            X, y = preprocessor.preprocess_data(df)
            
            logging.info("Splitting data into train/test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logging.info("Starting model training")
            results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            logging.info("\nTraining Results:")
            for model_name, metrics in results.items():
                logging.info(f"\n{model_name.upper()} Results:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value:.4f}")
            
            logging.info("\nTraining completed successfully!")
            display_results(results)

            deployment = ModelDeployment()
            save_dir = deployment.save_models(
                models=trainer.models,
                preprocessor=preprocessor,
                results=trainer.results
            )
            print(f"Models saved to: {save_dir}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Training failed!")
        return

if __name__ == "__main__":
    main()
