"""
Streamlit App - Manual Model Upload
Works without gdown - for 300MB files, use manual upload or external hosting
"""

from __future__ import annotations
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import os

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

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    
    # Check if file exists locally
    if os.path.exists('trained_models.pkl'):
        try:
            with open('trained_models.pkl', 'rb') as f:
                models = pickle.load(f)
            st.sidebar.success("✅ Models loaded")
            return models
        except Exception as e:
            st.error(f"❌ Error loading: {str(e)}")
            if st.button("🗑️ Delete and re-upload"):
                os.remove('trained_models.pkl')
                st.rerun()
    
    # Show upload interface
    st.error("### ❌ Model file not found!")
    st.markdown("---")
    
    st.markdown("""
    ### 📤 Upload Your Model File
    
    **Your model file is 300MB, which exceeds Streamlit's 200MB upload limit.**
    
    **Solutions:**
    
    #### Option 1: Use Git LFS (Recommended)
    ```bash
    # In your local repo:
    git lfs install
    git lfs track "trained_models.pkl"
    git add .gitattributes trained_models.pkl
    git commit -m "Add model with Git LFS"
    git push
    ```
    
    #### Option 2: Host Externally
    Upload to:
    - AWS S3 (with public link)
    - Dropbox (with direct link)
    - Google Cloud Storage
    
    Then update code to download from URL.
    
    #### Option 3: Compress the Model
    ```bash
    # Compress locally:
    python -c "
    import pickle, gzip
    with open('trained_models.pkl', 'rb') as f:
        with gzip.open('trained_models.pkl.gz', 'wb') as gz:
            gz.write(f.read())
    "
    # Result: ~100MB (fits in GitHub!)
    ```
    
    #### Option 4: Reduce Model Size
    Retrain with fewer trees:
    ```python
    # In train_catboost_lof.py:
    CatBoostClassifier(
        iterations=300,  # Down from 500
        depth=8          # Down from 10
    )
    # Result: ~150MB
    ```
    """)
    
    st.markdown("---")
    st.info("💡 Once you've set up one of the above options, redeploy the app!")
    
    st.stop()

@st.cache_data
def load_metrics():
    """Load pre-computed metrics"""
    if not os.path.exists('metrics.json'):
        return None
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
        return None

@st.cache_data
def load_sample_predictions():
    """Load sample predictions"""
    if not os.path.exists('sample_predictions.json'):
        return None
    try:
        with open('sample_predictions.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load sample predictions: {e}")
        return None

def predict_single_sample(models, features):
    """Make prediction - CatBoost + LOF"""
    features_scaled = models['scaler'].transform([features])
    
    # CatBoost
    catboost_proba = models['catboost_model'].predict_proba(features_scaled)[0, 1]
    catboost_pred = 1 if catboost_proba >= models['catboost_threshold'] else 0
    
    # LOF
    lof_decision = models['lof_model'].predict(features_scaled)[0]
    lof_pred = 1 if lof_decision == -1 else 0
    lof_score = models['lof_model'].score_samples(features_scaled)[0]
    scaled_score = lof_score * 2.0
    lof_proba = 1 / (1 + np.exp(scaled_score))
    lof_proba = np.clip(lof_proba, 0.01, 0.99)
    
    # Voting
    voting_pred = max(catboost_pred, lof_pred)
    voting_proba = max(catboost_proba, lof_proba)
    
    return {
        'catboost_pred': catboost_pred,
        'catboost_proba': catboost_proba,
        'lof_pred': lof_pred,
        'lof_proba': lof_proba,
        'voting_pred': voting_pred,
        'voting_proba': voting_proba
    }

# Main app
st.title("🛡️ IDS Inference Dashboard")
st.markdown("**CatBoost + LOF Hybrid Model**")
st.caption("Fast inference • No training overhead")
st.markdown("---")

# Load models
models = load_models()
metrics = load_metrics()
sample_preds = load_sample_predictions()

# Sidebar
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

if os.path.exists('trained_models.pkl'):
    file_size = os.path.getsize('trained_models.pkl') / 1024 / 1024
    st.sidebar.success(f"📦 Model: {file_size:.1f} MB")

# MODE 1: View Metrics
if mode == "📊 View Metrics":
    st.header("📊 Model Performance Metrics")
    
    if metrics is None:
        st.error("Metrics not found!")
        st.stop()
    
    results = metrics['results']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{results['accuracy']:.3f}")
    col2.metric("Precision", f"{results['precision']:.3f}")
    col3.metric("Recall", f"{results['recall']:.3f}")
    col4.metric("F1-Score", f"{results['f1_score']:.3f}")
    col5.metric("ROC-AUC", f"{results.get('roc_auc', 0):.3f}")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix")
        st.metric("True Positives", f"{results['true_positives']:,}")
        st.metric("True Negatives", f"{results['true_negatives']:,}")
        st.metric("False Positives", f"{results['false_positives']:,}")
        st.metric("False Negatives", f"{results['false_negatives']:,}")
    
    with col_b:
        st.subheader("Additional Metrics")
        st.metric("FPR", f"{results['fpr']:.4f}")
        st.metric("FNR", f"{results['fnr']:.4f}")
        st.metric("Training Time", f"{results['training_time']:.2f}s")

# MODE 2: Live Prediction
elif mode == "🎯 Live Prediction":
    st.header("🎯 Live Attack Prediction")
    
    feature_names = models['feature_names']
    n_features = len(feature_names)
    
    if st.button("🎲 Generate Random Sample"):
        st.session_state.random_sample = True
    
    st.markdown("**Quick Input (Top 10 Features):**")
    
    input_values = []
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(feature_names[:10]):
        with col1 if i % 2 == 0 else col2:
            default = np.random.randn() if getattr(st.session_state, 'random_sample', False) else 0.0
            val = st.number_input(feature, value=float(default), format="%.4f", key=f"feat_{i}")
            input_values.append(val)
    
    input_values.extend([np.random.randn() if getattr(st.session_state, 'random_sample', False) else 0.0 
                        for _ in range(n_features - 10)])
    
    st.markdown("---")
    
    if st.button("🚀 Predict", type="primary"):
        prediction = predict_single_sample(models, input_values)
        
        st.success("✅ Prediction Complete!")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### CatBoost")
            result = "🔴 **ATTACK**" if prediction['catboost_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(result)
            st.metric("Confidence", f"{prediction['catboost_proba']:.2%}")
        
        with col2:
            st.markdown("### LOF")
            result = "🔴 **ATTACK**" if prediction['lof_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(result)
            st.metric("Confidence", f"{prediction['lof_proba']:.2%}")
        
        with col3:
            st.markdown("### 🎖️ **Voting**")
            result = "🔴 **ATTACK**" if prediction['voting_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(f"## {result}")
            st.metric("Confidence", f"{prediction['voting_proba']:.2%}")
        
        if hasattr(st.session_state, 'random_sample'):
            st.session_state.random_sample = False

# MODE 3: Sample Predictions
elif mode == "📋 Sample Predictions":
    st.header("📋 Sample Test Predictions")
    
    if sample_preds is None:
        st.error("Sample predictions not found!")
        st.stop()
    
    df = pd.DataFrame({
        'True Label': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_true']],
        'CatBoost': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_pred_catboost']],
        'LOF': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_pred_lof']],
        'Voting': ['Attack' if y == 1 else 'Benign' for y in sample_preds['y_pred_voting']],
        'CatBoost Prob': sample_preds['proba_catboost'],
        'LOF Prob': sample_preds['proba_lof'],
        'Voting Prob': sample_preds['proba_voting']
    })
    
    col1, col2 = st.columns(2)
    with col1:
        filter_option = st.selectbox("Filter", ["All", "Correct Predictions", "Incorrect Predictions", "Attacks Only", "Benign Only"])
    with col2:
        n_rows = st.slider("Rows to display", 10, 100, 50)
    
    y_true = np.array(sample_preds['y_true'])
    y_pred = np.array(sample_preds['y_pred_voting'])
    
    if filter_option == "Correct Predictions":
        df_filtered = df[y_true == y_pred]
    elif filter_option == "Incorrect Predictions":
        df_filtered = df[y_true != y_pred]
    elif filter_option == "Attacks Only":
        df_filtered = df[y_true == 1]
    elif filter_option == "Benign Only":
        df_filtered = df[y_true == 0]
    else:
        df_filtered = df
    
    st.dataframe(df_filtered.head(n_rows), use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "predictions.csv", "text/csv")