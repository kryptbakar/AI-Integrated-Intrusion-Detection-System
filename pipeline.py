from __future__ import annotations
import time
import warnings
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils import resample
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
except ImportError as e:
    import sys
    print(f"Warning: imbalanced-learn import failed: {e}")
    print("Attempting alternative import...")
    
    # Try downgrading sklearn temporarily
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn==1.3.2"])
    
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")


class OptimizedVotingHybridPipeline:
    '''
    Enhanced Hybrid IDS with explainability artifacts for SHAP/LIME.
    Maintains original API while storing intermediate results.
    '''

    def __init__(
        self,
        data_path: str,
        xgb_threshold: float = 0.05,
        ocsvm_nu: float = 0.05,
        ocsvm_sample_ratio: float = 0.3,
        use_linear_kernel: bool = False
    ) -> None:
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.xgb_threshold = float(xgb_threshold)
        self.ocsvm_nu = float(ocsvm_nu)
        self.ocsvm_sample_ratio = float(ocsvm_sample_ratio)
        self.use_linear_kernel = bool(use_linear_kernel)
        self.xgb_model: Optional[XGBClassifier] = None
        self.ocsvm_model: Optional[OneClassSVM] = None

        # Configuration
        self.test_size: float = 0.3

        # Artifacts for explainability and visualization
        self.df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[np.ndarray] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.X_test_scaled: Optional[np.ndarray] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred_xgb: Optional[np.ndarray] = None
        self.y_pred_ocsvm: Optional[np.ndarray] = None
        self.y_pred_voting: Optional[np.ndarray] = None
        self.proba_xgb: Optional[np.ndarray] = None
        self.proba_ocsvm: Optional[np.ndarray] = None
        self.proba_voting: Optional[np.ndarray] = None

    def load_and_clean(self) -> pd.DataFrame:
        '''Load dataset and handle missing values'''
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")

        # Handle infinite values
        for col in ['Flow Bytes/s', 'Flow Packets/s']:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)

        if 'Label' not in self.df.columns:
            raise ValueError("Expected 'Label' column in dataset.")

        # Convert label to binary
        self.df['Label'] = self.df['Label'].apply(
            lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
        )

        print(f"\nClass distribution:\n{self.df['Label'].value_counts()}")
        return self.df

    def temporal_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Strict temporal split - NO SHUFFLING'''
        print("\n" + "="*70)
        print("TEMPORAL SPLIT - CHRONOLOGICAL (NO SHUFFLING)")
        print("="*70)

        n = len(df)
        split_idx = int(n * (1 - test_size))

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\nTrain: {len(train_df):,} samples")
        print(f"Test:  {len(test_df):,} samples")

        return train_df, test_df

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''Prepare features with proper imputation'''
        y_train = train_df['Label'].copy()
        y_test = test_df['Label'].copy()

        # Get numeric columns
        num_cols = train_df.drop(columns=['Label']).select_dtypes(
            include=[np.number]
        ).columns

        X_train = train_df[num_cols].copy()
        X_test = test_df[num_cols].copy()

        # Impute using training statistics only
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med).fillna(0)
        X_test = X_test.fillna(med).fillna(0)

        self.feature_names = num_cols.to_numpy()
        self.y_train, self.y_test = y_train, y_test

        return X_train, X_test, y_train, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Scale features - FIT on training only'''
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.X_train_scaled = X_train_s
        self.X_test_scaled = X_test_s

        print("✓ Features scaled using training statistics only")
        return X_train_s, X_test_s

    def aggressive_balance(
        self,
        X_train: np.ndarray,
        y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Aggressive balancing for XGBoost'''
        print("\n" + "="*70)
        print("AGGRESSIVE BALANCING (For XGBoost)")
        print("="*70)

        y = pd.Series(y_train)
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        ratio = (n_pos / max(n_neg, 1)) if n_neg else 0.0

        # Light undersampling if needed
        if ratio < 0.05:
            target_ratio = min(0.08, ratio * 2) if ratio > 0 else 0.08
            rus = RandomUnderSampler(sampling_strategy=target_ratio, random_state=42)
            X_rus, y_rus = rus.fit_resample(X_train, y_train)
            print(f"After undersampling: {pd.Series(y_rus).value_counts().to_dict()}")
        else:
            X_rus, y_rus = X_train, y_train

        # SMOTE
        try:
            smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
            X_sm, y_sm = smote.fit_resample(X_rus, y_rus)
            print(f"After SMOTE: {pd.Series(y_sm).value_counts().to_dict()}")
        except Exception:
            smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=3)
            X_sm, y_sm = smote.fit_resample(X_rus, y_rus)

        # ADASYN
        try:
            adasyn = ADASYN(sampling_strategy=1.0, random_state=42, n_neighbors=5)
            X_bal, y_bal = adasyn.fit_resample(X_sm, y_sm)
            print(f"After ADASYN: {pd.Series(y_bal).value_counts().to_dict()}")
        except Exception:
            X_bal, y_bal = X_sm, y_sm

        return X_bal, y_bal

    def build_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[XGBClassifier, float]:
        '''Build XGBoost model'''
        print("\n" + "="*70)
        print("TRAINING XGBOOST")
        print("="*70)

        y = pd.Series(y_train)
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        spw = (n_neg / max(n_pos, 1)) * 3 if n_pos else 1.0

        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='aucpr',
            tree_method='hist'
        )

        t0 = time.time()
        self.xgb_model.fit(X_train, y_train)
        training_time = time.time() - t0

        print(f"✓ Training completed in {training_time:.2f}s")
        return self.xgb_model, training_time

    def build_optimized_ocsvm(
        self,
        X_train_benign: np.ndarray
    ) -> Tuple[OneClassSVM, float]:
        '''Build optimized One-Class SVM'''
        print("\n" + "="*70)
        print("TRAINING OPTIMIZED ONE-CLASS SVM")
        print("="*70)

        original_size = len(X_train_benign)

        # Sample if dataset is large
        if original_size > 100_000:
            n_sample = max(100_000, int(original_size * self.ocsvm_sample_ratio))
            X_benign = resample(
                X_train_benign,
                n_samples=n_sample,
                random_state=42,
                replace=False
            )
            print(f"✓ Sampled {n_sample:,} from {original_size:,} samples")
        else:
            X_benign = X_train_benign
            print(f"✓ Using all {original_size:,} samples")

        kernel = 'linear' if self.use_linear_kernel else 'rbf'
        gamma = 'auto' if self.use_linear_kernel else 'scale'

        self.ocsvm_model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=self.ocsvm_nu,
            cache_size=2000,
            max_iter=1000,
            verbose=False
        )

        t0 = time.time()
        self.ocsvm_model.fit(X_benign)
        training_time = time.time() - t0

        print(f"✓ Training completed in {training_time:.2f}s")
        return self.ocsvm_model, training_time

    def predict_xgboost(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''XGBoost predictions with threshold'''
        proba = self.xgb_model.predict_proba(X)[:, 1]
        yhat = (proba >= self.xgb_threshold).astype(int)
        return yhat, proba

    def predict_ocsvm(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''OCSVM predictions'''
        signed = self.ocsvm_model.predict(X)  # 1=inlier, -1=outlier
        yhat = np.where(signed == -1, 1, 0)
        scores = self.ocsvm_model.decision_function(X)
        proba = 1 / (1 + np.exp(scores))
        return yhat, proba

    def evaluate_model(
        self,
        model_name: str,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        training_time: float = 0
    ) -> Dict[str, Any]:
        '''Comprehensive evaluation'''
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0

        results = {
            'model_name': model_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc,
            'fpr': fpr,
            'fnr': fnr,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'training_time': float(training_time)
        }

        print(f"\n" + "="*70)
        print(f"{model_name} - PERFORMANCE")
        print(f"="*70)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc:
            print(f"ROC-AUC:   {roc:.4f}")
        print(f"\nTP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")

        return results

    def run_pipeline(self) -> Dict[str, Any]:
        '''Execute complete pipeline with artifacts'''
        print("="*80)
        print("OPTIMIZED HYBRID VOTING PIPELINE")
        print("="*80)

        # Load and split
        df = self.load_and_clean()
        train_df, test_df = self.temporal_split(df, test_size=self.test_size)

        # Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(train_df, test_df)
        X_train_s, X_test_s = self.scale_features(X_train, X_test)

        # Balance for XGBoost
        X_xgb, y_xgb = self.aggressive_balance(X_train_s, y_train)

        # Extract benign for OCSVM
        X_benign = X_train_s[(y_train.values == 0)]

        # Train models
        _, t_xgb = self.build_xgboost(X_xgb, y_xgb)
        y_xgb, p_xgb = self.predict_xgboost(X_test_s)

        _, t_oc = self.build_optimized_ocsvm(X_benign)
        y_oc, p_oc = self.predict_ocsvm(X_test_s)

        # Voting ensemble
        print("\n" + "="*70)
        print("CREATING VOTING HYBRID ENSEMBLE")
        print("="*70)
        print("Strategy: Predict ATTACK if EITHER model detects it (OR logic)")

        y_vote = np.maximum(y_xgb, y_oc).astype(int)
        p_vote = np.maximum(p_xgb, p_oc)
        total_t = t_xgb + t_oc

        # Store all artifacts
        self.y_pred_xgb = y_xgb
        self.y_pred_ocsvm = y_oc
        self.y_pred_voting = y_vote
        self.proba_xgb = p_xgb
        self.proba_ocsvm = p_oc
        self.proba_voting = p_vote

        # Evaluate
        results = self.evaluate_model(
            'Optimized Voting Hybrid (XGBoost OR OCSVM)',
            y_test, y_vote, p_vote, total_t
        )

        print("\n✅ PIPELINE COMPLETED")

        # Return comprehensive artifacts
        return {
            "results": results,
            "y_test": y_test.values,
            "y_pred_voting": y_vote,
            "proba_voting": p_vote,
            "proba_xgb": p_xgb,
            "proba_ocsvm": p_oc,
            "y_pred_xgb": y_xgb,
            "y_pred_ocsvm": y_oc,
            "feature_names": self.feature_names,
            "X_train_scaled": self.X_train_scaled,
            "X_test_scaled": self.X_test_scaled,
            "xgb_model": self.xgb_model,
            "ocsvm_model": self.ocsvm_model,
        }