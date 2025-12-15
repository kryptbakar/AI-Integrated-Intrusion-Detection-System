"""
Streamlit App with Google Drive Download (No gdown dependency)
Uses requests library for Google Drive downloads
"""

from __future__ import annotations
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import os
import requests
from io import BytesIO

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

def download_from_google_drive(file_id, destination='trained_models.pkl'):
    """Download large file from Google Drive"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def extract_file_id(url):
    """Extract file ID from Google Drive URL"""
    if '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    elif 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    else:
        return url

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
            if st.button("🗑️ Delete and try again"):
                os.remove('trained_models.pkl')
                st.rerun()
    
    # Show upload interface
    st.error("### ❌ Model file not found!")
    st.markdown("---")
    
    # Tabs for different options
    tab1, tab2 = st.tabs(["🔗 Google Drive Link", "📋 Instructions"])
    
    with tab1:
        st.markdown("### 🔗 Download from Google Drive")
        st.markdown("""
        **Steps to get your link:**
        1. Upload `trained_models.pkl` to Google Drive
        2. Right-click → Share → "Anyone with the link"
        3. Copy the link
        4. Paste below and click Download
        """)
        
        st.markdown("---")
        
        gdrive_link = st.text_input(
            "Google Drive Link:",
            placeholder="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing",
            help="Paste your Google Drive shareable link here"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            download_btn = st.button("📥 Download", type="primary", use_container_width=True)
        
        if download_btn:
            if not gdrive_link:
                st.error("❌ Please enter a Google Drive link")
            else:
                try:
                    # Extract file ID
                    file_id = extract_file_id(gdrive_link)
                    st.info(f"File ID: {file_id}")
                    
                    # Download
                    with st.spinner("📥 Downloading from Google Drive (300MB)... This may take 1-2 minutes"):
                        download_from_google_drive(file_id, 'trained_models.pkl')
                    
                    # Check if downloaded
                    if os.path.exists('trained_models.pkl'):
                        file_size = os.path.getsize('trained_models.pkl') / 1024 / 1024
                        if file_size > 1:  # At least 1MB
                            st.success(f"✅ Downloaded {file_size:.1f} MB!")
                            st.info("🔄 Reloading page...")
                            st.rerun()
                        else:
                            st.error("❌ Download failed - file too small")
                            os.remove('trained_models.pkl')
                    else:
                        st.error("❌ Download failed")
                        
                except Exception as e:
                    st.error(f"❌ Download error: {str(e)}")
                    st.info("**Troubleshooting:**")
                    st.markdown("""
                    - Make sure the link is set to "Anyone with the link"
                    - Try copying the link again
                    - Check if file ID is correct
                    """)
        
        st.markdown("---")
        st.markdown("**Example link format:**")
        st.code("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing")
    
    with tab2:
        st.markdown("""
        ### 📋 How to Setup Google Drive Link
        
        **Step 1: Upload to Google Drive**
        ```bash
        # After training locally:
        python train_catboost_lof.py
        # Creates trained_models.pkl (300MB)
        ```
        
        1. Go to [drive.google.com](https://drive.google.com)
        2. Click "New" → "File upload"
        3. Select `trained_models.pkl`
        4. Wait for upload (2-3 minutes)
        
        **Step 2: Make Shareable**
        1. Right-click on the uploaded file
        2. Click "Share"
        3. Change to **"Anyone with the link"**
        4. Set permission to **"Viewer"**
        5. Click "Copy link"
        
        **Step 3: Use in App**
        1. Go to "🔗 Google Drive Link" tab
        2. Paste the link
        3. Click "📥 Download"
        4. Wait 1-2 minutes
        5. App reloads with models!
        
        ---
        
        ### ✅ Link Format Examples
        
        **Full URL (works):**
        ```
        https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
        ```
        
        **Just File ID (works):**
        ```
        1AbCdEfGhIjKlMnOpQrStUvWxYz
        ```
        
        **Open link (works):**
        ```
        https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUvWxYz
        ```
        
        ---
        
        ### ⚠️ Common Issues
        
        **"Download failed":**
        - Link must be set to "Anyone with the link"
        - Check if file is still in your Drive
        - Try generating a new link
        
        **"File too small":**
        - Google returned an error page instead of file
        - Usually means sharing settings are wrong
        - Re-share with "Anyone with the link"
        
        **Takes too long:**
        - 300MB file takes 1-2 minutes
        - Don't close the browser
        - Streamlit Cloud has good bandwidth
        """)
    
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
st.caption("Google Drive integration • No file size limits")
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