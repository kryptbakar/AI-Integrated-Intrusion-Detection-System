"""
Streamlit IDS App - CatBoost + LOF
Clean version with working random sample generation
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
    if not os.path.exists('trained_models.pkl'):
        st.error("### ❌ Model file not found!")
        st.info("Please upload trained_models.pkl via Google Drive or GitHub")
        st.stop()
    
    try:
        with open('trained_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

@st.cache_data
def load_metrics():
    """Load metrics"""
    if not os.path.exists('metrics.json'):
        return None
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_sample_predictions():
    """Load sample predictions"""
    if not os.path.exists('sample_predictions.json'):
        return None
    try:
        with open('sample_predictions.json', 'r') as f:
            return json.load(f)
    except:
        return None

def generate_realistic_sample(feature_names):
    """
    Generate realistic network traffic features
    Returns a mix of benign and attack-like patterns
    """
    sample = []
    
    # Randomly choose: benign or attack pattern
    is_benign = np.random.rand() > 0.5  # 50/50 chance
    
    for feature in feature_names:
        # Generate realistic values based on feature name
        feature_lower = feature.lower()
        
        if 'port' in feature_lower:
            if is_benign:
                # Common service ports: 80, 443, 22, 53, etc.
                common_ports = [22, 53, 80, 110, 143, 443, 587, 993, 3306, 8080]
                value = np.random.choice(common_ports)
            else:
                # Random high ports or unusual ports
                value = np.random.randint(1024, 65535)
        
        elif 'duration' in feature_lower:
            if is_benign:
                # Normal session: 1-300 seconds
                value = np.random.uniform(1000000, 300000000)
            else:
                # Very short or very long
                if np.random.rand() > 0.5:
                    value = np.random.uniform(1, 10000)  # Very short
                else:
                    value = np.random.uniform(500000000, 2000000000)  # Very long
        
        elif 'packet' in feature_lower and 'total' in feature_lower:
            if is_benign:
                # Normal traffic: 10-1000 packets
                value = np.random.uniform(10, 1000)
            else:
                # Flood: many packets or very few
                if np.random.rand() > 0.5:
                    value = np.random.uniform(5000, 50000)
                else:
                    value = np.random.uniform(1, 5)
        
        elif 'length' in feature_lower and 'mean' in feature_lower:
            if is_benign:
                # Normal packet size: 500-1400 bytes
                value = np.random.uniform(500, 1400)
            else:
                # Unusual sizes
                value = np.random.uniform(40, 200)
        
        elif 'bytes/s' in feature_lower or 'byte' in feature_lower and 'rate' in feature_lower:
            if is_benign:
                # Normal bandwidth: 1KB/s - 10MB/s
                value = np.random.uniform(1000, 10000000)
            else:
                # Flood or very low
                if np.random.rand() > 0.5:
                    value = np.random.uniform(50000000, 500000000)
                else:
                    value = np.random.uniform(1, 100)
        
        elif 'packets/s' in feature_lower or 'packet' in feature_lower and 'rate' in feature_lower:
            if is_benign:
                # Normal rate: 1-100 packets/sec
                value = np.random.uniform(1, 100)
            else:
                # Flood or very low
                if np.random.rand() > 0.5:
                    value = np.random.uniform(500, 10000)
                else:
                    value = np.random.uniform(0.01, 0.5)
        
        elif 'flag' in feature_lower or 'count' in feature_lower:
            if is_benign:
                # Normal flag counts
                value = np.random.uniform(1, 10)
            else:
                # Unusual flag patterns
                value = np.random.uniform(0, 1) if np.random.rand() > 0.5 else np.random.uniform(20, 100)
        
        elif 'time' in feature_lower or 'iat' in feature_lower:
            if is_benign:
                # Normal inter-arrival time
                value = np.random.uniform(1000, 100000)
            else:
                # Very fast or very slow
                if np.random.rand() > 0.5:
                    value = np.random.uniform(1, 100)
                else:
                    value = np.random.uniform(500000, 5000000)
        
        else:
            # Generic feature: use moderate random values
            if is_benign:
                value = np.random.uniform(0, 100)
            else:
                value = np.random.uniform(0, 1000)
        
        sample.append(float(value))
    
    return sample, is_benign

def predict_single_sample(models, features):
    """Make prediction - CatBoost + LOF"""
    features_scaled = models['scaler'].transform([features])
    
    # CatBoost
    catboost_proba = models['catboost_model'].predict_proba(features_scaled)[0, 1]
    catboost_pred = 1 if catboost_proba >= models['catboost_threshold'] else 0
    
    # LOF
    lof_decision = models['lof_model'].predict(features_scaled)[0]
    lof_pred = 1 if lof_decision == -1 else 0
    
    # LOF probability - improved calculation
    # score_samples returns: negative values = outliers (attacks)
    lof_score = models['lof_model'].score_samples(features_scaled)[0]
    
    # More aggressive scaling to get varied probabilities
    # Adjust multiplier based on typical score range
    if lof_score < -2:
        # Very negative = definitely attack
        lof_proba = 0.95
    elif lof_score < -1:
        # Moderately negative = likely attack
        lof_proba = 0.70 + (abs(lof_score) - 1) * 0.25
    elif lof_score < 0:
        # Slightly negative = maybe attack
        lof_proba = 0.50 + abs(lof_score) * 0.20
    elif lof_score < 1:
        # Slightly positive = probably benign
        lof_proba = 0.50 - lof_score * 0.20
    elif lof_score < 2:
        # Moderately positive = likely benign
        lof_proba = 0.30 - (lof_score - 1) * 0.20
    else:
        # Very positive = definitely benign
        lof_proba = 0.05
    
    lof_proba = np.clip(lof_proba, 0.01, 0.99)
    
    # Voting
    voting_pred = max(catboost_pred, lof_pred)
    voting_proba = max(catboost_proba, lof_proba)
    
    return {
        'catboost_pred': catboost_pred,
        'catboost_proba': catboost_proba,
        'lof_pred': lof_pred,
        'lof_proba': lof_proba,
        'lof_score': lof_score,  # For debugging
        'voting_pred': voting_pred,
        'voting_proba': voting_proba
    }
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
st.caption("Fast inference • Google Drive support")
st.markdown("---")

# Load everything
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
    st.markdown("Enter network flow features to predict if traffic is benign or attack")
    
    st.markdown("---")
    
    feature_names = models['feature_names']
    n_features = len(feature_names)
    
    # Random sample button
    if st.button("🎲 Generate Random Sample"):
        # Generate realistic sample
        random_values, expected_class = generate_realistic_sample(feature_names)
        st.session_state.random_values = random_values
        st.session_state.expected_class = expected_class
        st.session_state.random_sample = True
        st.rerun()
    
    # Show hint if random sample was generated
    if st.session_state.get('random_sample', False):
        expected = "🟢 Benign" if st.session_state.get('expected_class', False) else "🔴 Attack"
        st.info(f"💡 Generated sample with {expected}-like pattern")
    
    st.markdown("**Quick Input (Top 10 Features):**")
    
    input_values = []
    col1, col2 = st.columns(2)
    
    # Show first 10 features
    for i, feature in enumerate(feature_names[:10]):
        if i % 2 == 0:
            with col1:
                # Use pre-generated random values if available
                if st.session_state.get('random_sample', False) and 'random_values' in st.session_state:
                    default = st.session_state.random_values[i]
                else:
                    default = 0.0
                
                val = st.number_input(
                    feature, 
                    value=float(default), 
                    format="%.4f", 
                    key=f"feat_{i}"
                )
                input_values.append(val)
        else:
            with col2:
                # Use pre-generated random values if available
                if st.session_state.get('random_sample', False) and 'random_values' in st.session_state:
                    default = st.session_state.random_values[i]
                else:
                    default = 0.0
                
                val = st.number_input(
                    feature, 
                    value=float(default), 
                    format="%.4f", 
                    key=f"feat_{i}"
                )
                input_values.append(val)
    
    # Fill remaining features (hidden) from pre-generated values
    if st.session_state.get('random_sample', False) and 'random_values' in st.session_state:
        input_values.extend(st.session_state.random_values[10:])
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
            st.markdown("### 🎖️ **Voting Ensemble**")
            result = "🔴 **ATTACK**" if prediction['voting_pred'] == 1 else "🟢 **BENIGN**"
            st.markdown(f"## {result}")
            st.metric("Confidence", f"{prediction['voting_proba']:.2%}")
        
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.json({
                'CatBoost': {
                    'probability': f"{prediction['catboost_proba']:.4f}",
                    'prediction': 'ATTACK' if prediction['catboost_pred'] == 1 else 'BENIGN',
                    'threshold': models['catboost_threshold']
                },
                'LOF': {
                    'raw_score': f"{prediction.get('lof_score', 0):.4f}",
                    'probability': f"{prediction['lof_proba']:.4f}",
                    'prediction': 'ATTACK' if prediction['lof_pred'] == 1 else 'BENIGN',
                    'note': 'Negative scores = outliers/attacks'
                },
                'Voting': {
                    'strategy': 'OR logic (max of both)',
                    'final': 'ATTACK' if prediction['voting_pred'] == 1 else 'BENIGN'
                }
            })
        
        # Clear random sample flag
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
        filter_option = st.selectbox(
            "Filter", 
            ["All", "Correct Predictions", "Incorrect Predictions", "Attacks Only", "Benign Only"]
        )
    with col2:
        n_rows = st.slider("Rows to display", 10, 100, 50)
    
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
    
    csv = df_filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Sample Predictions",
        csv,
        "sample_predictions.csv",
        "text/csv"
    )