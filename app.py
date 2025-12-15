"""
Streamlit App with Google Drive Auto-Download
Supports large model files (300MB+) via Google Drive
"""

from __future__ import annotations
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown

FEATURE_RANGES = {
    "Destination Port": (0, 65535),
    "Flow Duration": (0, 5_000_000),
    "Total Fwd Packets": (0, 10_000),
    "Total Backward Packets": (0, 10_000),
    "Flow Bytes/s": (0, 1e6),
    "Flow Packets/s": (0, 1e5),
    "Packet Length Mean": (40, 1500),
    "Packet Length Std": (0, 500),
    "Fwd Packet Length Mean": (40, 1500),
    "Bwd Packet Length Mean": (40, 1500),
}
def smart_default(feature):
    if feature in FEATURE_RANGES:
        low, high = FEATURE_RANGES[feature]
        return float(np.random.uniform(low, high))
    return 0.0

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

# Configuration - Set your Google Drive file ID here
GDRIVE_FILE_ID = None  # Set this after uploading to Google Drive
GDRIVE_URL = None      # Or set direct URL

@st.cache_resource
def download_from_gdrive(file_id=None, url=None):
    """Download model file from Google Drive"""
    if file_id:
        download_url = f"https://drive.google.com/uc?id={file_id}"
    elif url:
        download_url = url
    else:
        return None
    
    output = 'trained_models.pkl'
    
    try:
        with st.spinner(f"📥 Downloading model from Google Drive... (300MB)"):
            gdown.download(download_url, output, quiet=False, fuzzy=True)
        
        if os.path.exists(output) and os.path.getsize(output) > 1000000:  # >1MB
            st.success(f"✅ Downloaded {os.path.getsize(output) / 1024 / 1024:.1f} MB")
            return output
        else:
            st.error("❌ Download failed or file too small")
            return None
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load pre-trained models - with multiple loading options"""
    
    # Option 1: Check if file exists locally
    if os.path.exists('trained_models.pkl'):
        try:
            with open('trained_models.pkl', 'rb') as f:
                models = pickle.load(f)
            st.sidebar.success("✅ Models loaded from local file")
            return models
        except Exception as e:
            st.error(f"❌ Error loading local file: {str(e)}")
            if st.button("🗑️ Delete corrupted file"):
                os.remove('trained_models.pkl')
                st.rerun()
    
    # Option 2: Try Google Drive if configured
    if GDRIVE_FILE_ID or GDRIVE_URL:
        st.info("📥 Downloading from Google Drive...")
        downloaded = download_from_gdrive(GDRIVE_FILE_ID, GDRIVE_URL)
        if downloaded:
            st.rerun()  # Reload to load the downloaded file
    
    # Option 3: Show upload interface
    st.error("### ❌ Model file not found!")
    st.markdown("---")
    
    # Tab interface for different options
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "🔗 Google Drive Link", "📋 Instructions"])
    
    with tab1:
        st.markdown("### 📤 Upload Model File")
        st.warning("⚠️ Streamlit upload limit: 200MB. For 300MB files, use Google Drive (Tab 2)")
        
        uploaded_file = st.file_uploader(
            "Upload trained_models.pkl (max 200MB)",
            type=['pkl'],
            help="For files >200MB, use Google Drive option"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024
            st.info(f"File size: {file_size:.1f} MB")
            
            if file_size > 200:
                st.error("❌ File too large! Use Google Drive option instead.")
            else:
                with st.spinner("Uploading..."):
                    with open('trained_models.pkl', 'wb') as f:
                        f.write(uploaded_file.getvalue())
                st.success("✅ Uploaded! Reloading...")
                st.rerun()
    
    with tab2:
        st.markdown("### 🔗 Google Drive Link (For Large Files)")
        st.markdown("""
        **Steps:**
        1. Upload `trained_models.pkl` to Google Drive
        2. Right-click → Share → "Anyone with the link"
        3. Copy the link
        4. Paste below
        """)
        
        gdrive_link = st.text_input(
            "Google Drive Link:",
            placeholder="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"
        )
        
        if st.button("📥 Download from Google Drive", type="primary"):
            if gdrive_link:
                # Extract file ID
                if '/d/' in gdrive_link:
                    file_id = gdrive_link.split('/d/')[1].split('/')[0]
                elif 'id=' in gdrive_link:
                    file_id = gdrive_link.split('id=')[1].split('&')[0]
                else:
                    file_id = gdrive_link
                
                downloaded = download_from_gdrive(file_id=file_id)
                if downloaded:
                    st.success("✅ Download complete!")
                    st.rerun()
            else:
                st.error("Please enter a Google Drive link")
    
    with tab3:
        st.markdown("""
        ### 📋 How to Train Models
        
        **On your local machine:**
        
        ```bash
        # 1. Install dependencies
        pip install pandas numpy scikit-learn catboost imbalanced-learn
        
        # 2. Run training (4-6 minutes)
        python train_catboost_lof.py
        
        # 3. This creates trained_models.pkl (300MB)
        ```
        
        **Then choose:**
        - **Tab 1 (Upload):** For files <200MB
        - **Tab 2 (Google Drive):** For files >200MB ⭐ **Recommended**
        
        ---
        
        ### 🔗 Permanent Google Drive Setup
        
        **To avoid re-uploading every time:**
        
        1. Upload to Google Drive (one time)
        2. Get shareable link
        3. Set in code (line 29-30):
           ```python
           GDRIVE_FILE_ID = "your_file_id_here"
           ```
        4. Redeploy - auto-downloads on startup!
        """)
    
    st.stop()

@st.cache_data
def load_metrics():
    """Load pre-computed metrics safely"""
    possible_paths = [
        "metrics.json",
        "./metrics.json",
        os.path.join(os.getcwd(), "metrics.json")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Could not load metrics from {path}: {e}")

    st.error("❌ metrics.json not found in repository")
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
def smart_default(feature, feature_distributions, force_class=None):
    """
    Generate realistic random values based on training distributions.
    force_class: None | 'benign' | 'attack'
    """
    stats = feature_distributions.get(feature)
    if not stats:
        return 0.0

    benign = stats["benign"]
    attack = stats["attack"]

    if force_class == "benign":
        mu, sigma = benign["mean"], benign["std"]
    elif force_class == "attack":
        mu, sigma = attack["mean"], attack["std"]
    else:
        # Balanced: randomly choose
        if np.random.rand() < 0.5:
            mu, sigma = benign["mean"], benign["std"]
        else:
            mu, sigma = attack["mean"], attack["std"]

    return float(np.random.normal(mu, sigma))

def predict_single_sample(models, features):
    """Make prediction on single sample - CatBoost + LOF"""
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
st.markdown("**Pre-trained Hybrid Model: CatBoost + LOF**")
st.caption("Supports large files via Google Drive • 2-3x faster training")
st.markdown("---")

@st.cache_data
def load_feature_distributions():
    """
    Load per-feature distributions used for random sample generation.
    
    Priority:
    1) Use `feature_stats.json` (means/stds for each feature) – always present in this repo.
    2) Fallback to any legacy `feature_distributions` entry inside `sample_predictions.json`.
    """
    # 1) Preferred: feature_stats.json
    feature_stats_path = "feature_stats.json"
    if os.path.exists(feature_stats_path):
        try:
            with open(feature_stats_path, "r") as f:
                stats = json.load(f)
        except Exception:
            stats = None
        if isinstance(stats, dict):
            means = stats.get("means", {})
            stds = stats.get("stds", {})
            feature_distributions = {}
            for feat, mu in means.items():
                sigma = stds.get(feat, 1.0) or 1.0
                # We don't have separate benign/attack stats here, so we reuse
                # the same distribution for both – this keeps the existing
                # smart_default() interface and UI bias options working.
                feature_distributions[feat] = {
                    "benign": {"mean": mu, "std": sigma},
                    "attack": {"mean": mu, "std": sigma},
                }
            if feature_distributions:
                return feature_distributions

    # 2) Legacy fallback: look inside sample_predictions.json
    if os.path.exists("sample_predictions.json"):
        try:
            with open("sample_predictions.json", "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                legacy = data.get("feature_distributions")
                if legacy:
                    return legacy
        except Exception:
            pass

    # If nothing found, let callers handle the None case (falls back to zeros)
    return None

# Load models
models = load_models()
metrics = load_metrics()
sample_preds = load_sample_predictions()

feature_distributions = load_feature_distributions()

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

# File info
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
    # --- Random sample controls ---
    colA, colB = st.columns(2)

    with colA:
        random_sample = st.button("🎲 Generate Random Sample")

    with colB:
        force_class = st.selectbox(
            "Random Sample Bias",
            ["Balanced", "Benign", "Attack"]
        )

    if random_sample:
        st.session_state.random_sample = True
        st.session_state.force_class = (
            None if force_class == "Balanced" else force_class.lower()
        )

    
    st.markdown("**Quick Input (Top 10 Features):**")
    
    input_values = []
    col1, col2 = st.columns(2)

    # --- Top 10 visible features ---
    for i, feature in enumerate(feature_names[:10]):
        with col1 if i % 2 == 0 else col2:
            if getattr(st.session_state, "random_sample", False) and feature_distributions:
                default = smart_default(
                    feature,
                    feature_distributions,
                    getattr(st.session_state, "force_class", None)
                )
            else:
                default = 0.0

            val = st.number_input(
                feature,
                value=float(default),
                format="%.4f",
                key=f"feat_{i}"
            )
            input_values.append(val)

    # --- Remaining hidden features ---
    for feature in feature_names[10:]:
        if getattr(st.session_state, "random_sample", False) and feature_distributions:
            val = smart_default(
                feature,
                feature_distributions,
                getattr(st.session_state, "force_class", None)
            )
        else:
            val = 0.0

        input_values.append(val)

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