"""
Lightweight Streamlit App - Inference Only
Loads pre-trained models and shows metrics/predictions
NO TRAINING - runs entirely on pre-computed results
"""

from __future__ import annotations
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Page config
st.set_page_config(
    page_title="IDS - Inference Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown('''
    <style>
    .main {background-color: #000000; color: #ffffff;}
    [data-testid="stSidebar"] {background-color: #1a1a1a;}
    .stMetric {background-color: #1a1a1a; padding: 10px; border-radius: 5px; border: 1px solid #333333;}
    h1, h2, h3, h4, h5, h6 {color: #ffffff !important;}
    </style>
''', unsafe_allow_html=True)

# Load pre-trained models and data
@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        with open('trained_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("❌ trained_models.pkl not found!")
        st.error("Run train_local.py on your local machine first")
        return None

@st.cache_data
def load_metrics():
    """Load pre-computed metrics"""
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_sample_predictions():
    """Load sample predictions"""
    try:
        with open('sample_predictions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_feature_stats():
    """Load feature statistics"""
    try:
        with open('feature_stats.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def predict_single_sample(models, features):
    """Make prediction on single sample"""
    # Scale features
    features_scaled = models['scaler'].transform([features])
    
    # XGBoost prediction
    xgb_proba = models['xgb_model'].predict_proba(features_scaled)[0, 1]
    xgb_pred = 1 if xgb_proba >= models['xgb_threshold'] else 0
    
    # OCSVM prediction
    ocsvm_decision = models['ocsvm_model'].predict(features_scaled)[0]
    ocsvm_pred = 1 if ocsvm_decision == -1 else 0
    ocsvm_proba = 1 / (1 + np.exp(models['ocsvm_model'].decision_function(features_scaled)[0]))
    
    # Voting
    voting_pred = max(xgb_pred, ocsvm_pred)
    voting_proba = max(xgb_proba, ocsvm_proba)
    
    return {
        'xgb_pred': xgb_pred,
        'xgb_proba': xgb_proba,
        'ocsvm_pred': ocsvm_pred,
        'ocsvm_proba': ocsvm_proba,
        'voting_pred': voting_pred,
        'voting_proba': voting_proba
    }

def plot_confusion_matrix_custom(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='black')
    ax.set_facecolor('black')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('True', color='white')
    ax.set_title('Confusion Matrix', color='white')
    ax.tick_params(colors='white')
    
    return fig

# Main app
st.title("🛡️ IDS Inference Dashboard")
st.markdown("**Pre-trained Hybrid Model: XGBoost + One-Class SVM**")
st.caption("Trained locally • Deployed for inference • No training overhead")
st.markdown("---")

# Load everything
models = load_models()
metrics = load_metrics()
sample_preds = load_sample_predictions()
feature_stats = load_feature_stats()

if models is None:
    st.error("### ⚠️ Models not found!")
    st.info("**Setup Instructions:**")
    st.code("""
# 1. On your local machine, run:
python train_local.py

# 2. This creates 4 files:
#    - trained_models.pkl
#    - metrics.json
#    - sample_predictions.json
#    - feature_stats.json

# 3. Add these files to your GitHub repo
# 4. Deploy this app to Streamlit Cloud
    """)
    st.stop()

# Sidebar - Mode selection
st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["📊 View Metrics", "🎯 Live Prediction", "📋 Sample Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Info")
if metrics:
    st.sidebar.info(f"**Trained:** {metrics.get('training_date', 'Unknown')[:10]}")
    st.sidebar.info(f"**Samples:** {metrics['dataset_info']['total_samples']:,}")
    st.sidebar.info(f"**Features:** {metrics['dataset_info']['features']}")

# MODE 1: View Metrics
if mode == "📊 View Metrics":
    st.header("📊 Model Performance Metrics")
    
    if metrics is None:
        st.error("Metrics not found!")
        st.stop()
    
    results = metrics['results']
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{results['accuracy']:.3f}")
    col2.metric("Precision", f"{results['precision']:.3f}")
    col3.metric("Recall", f"{results['recall']:.3f}")
    col4.metric("F1-Score", f"{results['f1_score']:.3f}")
    col5.metric("ROC-AUC", f"{results.get('roc_auc', 0):.3f}")
    
    st.markdown("---")
    
    # Detailed metrics
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix Breakdown")
        st.metric("True Positives", f"{results['true_positives']:,}")
        st.metric("True Negatives", f"{results['true_negatives']:,}")
        st.metric("False Positives", f"{results['false_positives']:,}")
        st.metric("False Negatives", f"{results['false_negatives']:,}")
    
    with col_b:
        st.subheader("Additional Metrics")
        st.metric("False Positive Rate", f"{results['fpr']:.4f}")
        st.metric("False Negative Rate", f"{results['fnr']:.4f}")
        st.metric("Training Time", f"{results['training_time']:.2f}s")
    
    # Dataset info
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", f"{metrics['dataset_info']['total_samples']:,}")
    col2.metric("Training Samples", f"{metrics['dataset_info']['train_samples']:,}")
    col3.metric("Test Samples", f"{metrics['dataset_info']['test_samples']:,}")
    
    # Class distribution
    st.markdown("---")
    st.subheader("Class Distribution")
    class_dist = metrics['dataset_info']['class_distribution']
    
    dist_df = pd.DataFrame({
        'Class': ['Benign', 'Attack'],
        'Count': [class_dist.get('0', 0), class_dist.get('1', 0)]
    })
    
    st.bar_chart(dist_df.set_index('Class'))

# MODE 2: Live Prediction
elif mode == "🎯 Live Prediction":
    st.header("🎯 Live Attack Prediction")
    st.markdown("Enter network flow features to predict if traffic is benign or attack")
    
    st.markdown("---")
    
    # Create input form
    st.subheader("📝 Enter Feature Values")
    
    feature_names = models['feature_names']
    n_features = len(feature_names)
    
    # Option 1: Use random sample
    if st.button("🎲 Generate Random Sample"):
        st.session_state.random_sample = True
    
    # Option 2: Manual input (show first 10 features for simplicity)
    st.markdown("**Quick Input (Top 10 Features):**")
    
    input_values = []
    
    # Create 2 columns for inputs
    col1, col2 = st.columns(2)
    
    # Show first 10 features for manual input
    for i, feature in enumerate(feature_names[:10]):
        if i % 2 == 0:
            with col1:
                if hasattr(st.session_state, 'random_sample') and st.session_state.random_sample:
                    default = np.random.randn()
                else:
                    default = 0.0
                val = st.number_input(feature, value=float(default), format="%.4f", key=f"feat_{i}")
                input_values.append(val)
        else:
            with col2:
                if hasattr(st.session_state, 'random_sample') and st.session_state.random_sample:
                    default = np.random.randn()
                else:
                    default = 0.0
                val = st.number_input(feature, value=float(default), format="%.4f", key=f"feat_{i}")
                input_values.append(val)
    
    # Fill remaining features with zeros or random
    if hasattr(st.session_state, 'random_sample') and st.session_state.random_sample:
        input_values.extend([np.random.randn() for _ in range(n_features - 10)])
    else:
        input_values.extend([0.0 for _ in range(n_features - 10)])
    
    st.markdown("---")
    
    # Predict button
    if st.button("🚀 Predict", type="primary"):
        with st.spinner("Making prediction..."):
            prediction = predict_single_sample(models, input_values)
        
        st.success("✅ Prediction Complete!")
        
        # Show results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### XGBoost")
            result = "🔴 **ATTACK**" if prediction['xgb_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(result)
            st.metric("Confidence", f"{prediction['xgb_proba']:.2%}")
        
        with col2:
            st.markdown("### OCSVM")
            result = "🔴 **ATTACK**" if prediction['ocsvm_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(result)
            st.metric("Confidence", f"{prediction['ocsvm_proba']:.2%}")
        
        with col3:
            st.markdown("### 🎖️ **Voting Ensemble**")
            result = "🔴 **ATTACK**" if prediction['voting_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(f"## {result}")
            st.metric("Confidence", f"{prediction['voting_proba']:.2%}")
        
        # Clear random sample flag
        if hasattr(st.session_state, 'random_sample'):
            st.session_state.random_sample = False

# MODE 3: Sample Predictions
elif mode == "📋 Sample Predictions":
    st.header("📋 Sample Test Predictions")
    
    if sample_preds is None:
        st.error("Sample predictions not found!")
        st.stop()
    
    # Create DataFrame
    df = pd.DataFrame({
        'True Label': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_true']],
        'XGBoost': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_pred_xgb']],
        'OCSVM': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_pred_ocsvm']],
        'Voting': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_pred_voting']],
        'XGB Prob': sample_preds['proba_xgb'],
        'OCSVM Prob': sample_preds['proba_ocsvm'],
        'Voting Prob': sample_preds['proba_voting']
    })
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_option = st.selectbox(
            "Filter",
            ["All", "Correct Predictions", "Incorrect Predictions", "Attacks Only", "Benign Only"]
        )
    
    with col2:
        n_rows = st.slider("Rows to display", 10, 100, 50)
    
    # Apply filter
    y_true = np.array(sample_preds['y_true'])
    y_pred = np.array(sample_preds['y_pred_voting'])
    
    if filter_option == "Correct Predictions":
        mask = y_true == y_pred
        df_filtered = df[mask]
    elif filter_option == "Incorrect Predictions":
        mask = y_true != y_pred
        df_filtered = df[mask]
    elif filter_option == "Attacks Only":
        mask = y_true == 1
        df_filtered = df[mask]
    elif filter_option == "Benign Only":
        mask = y_true == 0
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    st.dataframe(df_filtered.head(n_rows), use_container_width=True)
    
    # Download option
    csv = df_filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Sample Predictions",
        csv,
        "sample_predictions.csv",
        "text/csv"
    )