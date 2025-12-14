from __future__ import annotations
from typing import Optional, Callable, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")


def _safe_background(
    X_train: np.ndarray,
    max_rows: int = 1000
) -> np.ndarray:
    """
    Create a safe background dataset for SHAP by sampling from training data.
    
    Args:
        X_train: Training data array
        max_rows: Maximum number of rows to include in background
        
    Returns:
        Background dataset for SHAP explainer
    """
    if len(X_train) <= max_rows:
        return X_train
    
    # Use kmeans or random sampling
    try:
        return shap.kmeans(X_train, min(max_rows, len(X_train)))
    except Exception:
        # Fallback to random sampling
        indices = np.random.choice(len(X_train), size=max_rows, replace=False)
        return X_train[indices]


def build_tree_explainer(
    model,
    background_data: Optional[np.ndarray] = None
) -> shap.TreeExplainer:
    """
    Build SHAP TreeExplainer for tree-based models like XGBoost.
    
    Args:
        model: Trained tree-based model (XGBoost, RandomForest, etc.)
        background_data: Optional background dataset for explainer
        
    Returns:
        SHAP TreeExplainer instance
    """
    if background_data is not None:
        explainer = shap.TreeExplainer(model, background_data)
    else:
        explainer = shap.TreeExplainer(model)
    
    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X_test: np.ndarray
) -> shap.Explanation:
    """
    Compute SHAP values for test data.
    
    Args:
        explainer: SHAP TreeExplainer instance
        X_test: Test data to explain
        
    Returns:
        SHAP Explanation object containing values and base values
    """
    shap_values = explainer(X_test)
    return shap_values


def plot_shap_bar(
    shap_values: shap.Explanation,
    X_test: np.ndarray,
    feature_names: np.ndarray,
    max_display: int = 20
) -> plt.Figure:
    """
    Create SHAP bar plot showing global feature importance.
    
    Args:
        shap_values: SHAP Explanation object
        X_test: Test data
        feature_names: Array of feature names
        max_display: Maximum number of features to display
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False
    )
    
    plt.tight_layout()
    return plt.gcf()


def plot_shap_beeswarm(
    shap_values: shap.Explanation,
    X_test: np.ndarray,
    feature_names: np.ndarray,
    max_display: int = 20
) -> plt.Figure:
    """
    Create SHAP beeswarm plot showing feature impact distribution.
    
    Args:
        shap_values: SHAP Explanation object
        X_test: Test data
        feature_names: Array of feature names
        max_display: Maximum number of features to display
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False
    )
    
    plt.tight_layout()
    return plt.gcf()


def plot_shap_waterfall_for_row(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_names: np.ndarray,
    max_display: int = 20
) -> plt.Figure:
    """
    Create SHAP waterfall plot for a single prediction.
    
    Args:
        explainer: SHAP TreeExplainer instance
        X_row: Single row of data to explain
        feature_names: Array of feature names
        max_display: Maximum number of features to display
        
    Returns:
        Matplotlib figure object
    """
    # Ensure X_row is 2D
    if X_row.ndim == 1:
        X_row = X_row.reshape(1, -1)
    
    # Compute SHAP values for this row
    shap_values = explainer(X_row)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.plots.waterfall(
        shap_values[0],
        max_display=max_display,
        show=False
    )
    
    plt.tight_layout()
    return plt.gcf()


class LimeHelper:
    """
    Helper class for LIME (Local Interpretable Model-agnostic Explanations).
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        feature_names: np.ndarray,
        mode: str = 'classification',
        discretize_continuous: bool = True
    ):
        """
        Initialize LIME explainer.
        
        Args:
            X_train: Training data for building explainer
            feature_names: Array of feature names
            mode: 'classification' or 'regression'
            discretize_continuous: Whether to discretize continuous features
        """
        self.feature_names = feature_names
        self.mode = mode
        
        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=list(feature_names),
            mode=mode,
            discretize_continuous=discretize_continuous,
            random_state=42
        )
    
    def explain_row(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_row: np.ndarray,
        top_k: int = 10,
        num_samples: int = 5000
    ) -> pd.DataFrame:
        """
        Explain a single prediction using LIME.
        
        Args:
            predict_fn: Prediction function that takes X and returns probabilities
            X_row: Single row to explain
            top_k: Number of top features to return
            num_samples: Number of samples for LIME to generate
            
        Returns:
            DataFrame with feature names and their weights
        """
        # Ensure X_row is 1D
        if X_row.ndim > 1:
            X_row = X_row.flatten()
        
        # Get explanation
        exp = self.explainer.explain_instance(
            data_row=X_row,
            predict_fn=predict_fn,
            num_features=top_k,
            num_samples=num_samples
        )
        
        # Extract feature weights
        weights = exp.as_list()
        
        # Parse into DataFrame
        features = []
        values = []
        
        for feature_desc, weight in weights:
            # Extract feature name (before inequality/range)
            feature_name = feature_desc.split('<=')[0].split('>')[0].split('<')[0].strip()
            features.append(feature_name)
            values.append(weight)
        
        df = pd.DataFrame({
            'feature': features,
            'weight': values,
            'description': [w[0] for w in weights]
        })
        
        return df
    
    def plot_explanation(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_row: np.ndarray,
        top_k: int = 10,
        num_samples: int = 5000
    ) -> plt.Figure:
        """
        Create visualization of LIME explanation.
        
        Args:
            predict_fn: Prediction function
            X_row: Single row to explain
            top_k: Number of top features
            num_samples: Number of samples for LIME
            
        Returns:
            Matplotlib figure
        """
        # Ensure X_row is 1D
        if X_row.ndim > 1:
            X_row = X_row.flatten()
        
        # Get explanation
        exp = self.explainer.explain_instance(
            data_row=X_row,
            predict_fn=predict_fn,
            num_features=top_k,
            num_samples=num_samples
        )
        
        # Create figure
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        
        return fig


def plot_shap_force(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_names: np.ndarray
) -> shap.plots.force:
    """
    Create SHAP force plot for a single prediction.
    
    Args:
        explainer: SHAP TreeExplainer instance
        X_row: Single row of data to explain
        feature_names: Array of feature names
        
    Returns:
        SHAP force plot object
    """
    # Ensure X_row is 2D
    if X_row.ndim == 1:
        X_row = X_row.reshape(1, -1)
    
    # Compute SHAP values
    shap_values = explainer(X_row)
    
    # Create force plot
    force_plot = shap.plots.force(
        shap_values[0],
        show=False
    )
    
    return force_plot


def plot_shap_decision(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_names: np.ndarray,
    feature_order: str = 'importance'
) -> plt.Figure:
    """
    Create SHAP decision plot showing how features contribute to prediction.
    
    Args:
        explainer: SHAP TreeExplainer instance
        X_row: Single row of data to explain
        feature_names: Array of feature names
        feature_order: Order of features ('importance' or 'original')
        
    Returns:
        Matplotlib figure object
    """
    # Ensure X_row is 2D
    if X_row.ndim == 1:
        X_row = X_row.reshape(1, -1)
    
    # Compute SHAP values
    shap_values = explainer(X_row)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.decision_plot(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values[0],
        features=X_row[0],
        feature_names=list(feature_names),
        feature_order=feature_order,
        show=False
    )
    
    plt.tight_layout()
    return plt.gcf()


def compute_feature_importance_shap(
    shap_values: shap.Explanation,
    feature_names: np.ndarray
) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.
    
    Args:
        shap_values: SHAP Explanation object
        feature_names: Array of feature names
        
    Returns:
        DataFrame with features ranked by importance
    """
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_values.values).mean(axis=0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return df
