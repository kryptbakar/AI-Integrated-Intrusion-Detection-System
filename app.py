from __future__ import annotations
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import OptimizedVotingHybridPipeline
from explain import (
    _safe_background, build_tree_explainer, compute_shap_values,
    plot_shap_bar, plot_shap_beeswarm, plot_shap_waterfall_for_row, LimeHelper,
)
from utils import (
    threshold_sweep, plot_confusion_matrix, plot_roc_curve,
    plot_pr_curve, plot_threshold_sweep
)
from gdrive_downloader import download_and_cache_dataset

# Page config
st.set_page_config(
    page_title="Hybrid IDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark theme with black background
st.markdown('''
    <style>
    /* Main background */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333333;
    }
    
    /* Text elements */
    .stMarkdown, p, span, label {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
    }
    
    .stButton > button:hover {
        background-color: #444444;
        border: 1px solid #666666;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #333333;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #333333;
    }
    </style>
''', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Option 1: Upload file
uploaded = st.sidebar.file_uploader(
    "📁 Option 1: Upload CSV (max 200MB)",
    type=["csv"],
    help="For smaller files"
)

st.sidebar.markdown("**OR**")

# Option 2: Google Drive link
use_gdrive = st.sidebar.checkbox("📥 Use Google Drive (700MB OK!)", value=False)

gdrive_url = None
if use_gdrive:
    gdrive_url = st.sidebar.text_input(
        "Paste Google Drive link",
        placeholder="https://drive.google.com/file/d/...",
        help="Share link from Google Drive"
    )
    
    with st.sidebar.expander("❓ How to get link"):
        st.markdown("""
        1. Upload to Google Drive
        2. Right-click → Share
        3. "Anyone with link" → Copy
        4. Paste here ☝️
        """)

st.sidebar.markdown("**OR**")

# Option 3: Local path
default_path = st.sidebar.text_input(
    "📂 Option 3: Local path",
    value="merged_output.csv",
    help="Path to local CSV file"
)

# Show current working directory for debugging
with st.sidebar.expander("🔍 Debug Info"):
    import os
    st.code(f"Current directory: {os.getcwd()}")
    st.code(f"Files in directory: {os.listdir('.')[:10]}")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")

test_size = st.sidebar.slider("Test split ratio", 0.1, 0.5, 0.3, 0.05)
xgb_threshold = st.sidebar.slider("XGBoost threshold", 0.0, 0.9, 0.15, 0.01)
ocsvm_nu = st.sidebar.slider("OCSVM ν (nu)", 0.01, 0.5, 0.2, 0.01)
ocsvm_sample = st.sidebar.slider("OCSVM sample ratio", 0.1, 1.0, 0.3, 0.05)
use_linear = st.sidebar.checkbox("Use Linear kernel (faster)", value=False)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Pipeline", type="primary")

# Helper functions
@st.cache_data(show_spinner=False)
def _persist_upload_to_tmp(bin_data: bytes) -> str:
    """Save uploaded file to temp location"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(tmp.name, "wb") as f:
        f.write(bin_data)
    return tmp.name

@st.cache_resource(show_spinner=False)
def _run_pipeline(
    path: str,
    test_size: float,
    xgb_t: float,
    oc_nu: float,
    oc_ratio: float,
    linear: bool
) -> tuple[OptimizedVotingHybridPipeline, dict]:
    """Run pipeline with caching"""
    pipe = OptimizedVotingHybridPipeline(
        data_path=path,
        xgb_threshold=xgb_t,
        ocsvm_nu=oc_nu,
        ocsvm_sample_ratio=oc_ratio,
        use_linear_kernel=linear,
    )
    pipe.test_size = test_size
    artifacts = pipe.run_pipeline()
    return pipe, artifacts

@st.cache_resource(show_spinner=False)
def _shap_setup(
    xgb_model,
    X_train_scaled: np.ndarray,
    feature_names: np.ndarray
) -> tuple:
    """Setup SHAP explainer with caching"""
    bg = _safe_background(X_train_scaled, max_rows=1000)
    explainer = build_tree_explainer(xgb_model, bg)
    return explainer, bg

# Main title
st.title("🛡️ Hybrid Intrusion Detection System")
st.markdown("**XGBoost + One-Class SVM with SHAP/LIME Explainability**")
st.caption("Temporal split • Aggressive balancing • OR-voting • Model interpretability")
st.markdown("---")

# Check if we should show initial instructions
if not run_btn:
    # Initial state
    st.info("👈 Configure parameters in the sidebar and click **Run Pipeline** to start")

    st.markdown('''### 🎯 What This Dashboard Does''')
    st.markdown('''
    - **Hybrid Detection**: Combines XGBoost (supervised) and One-Class SVM (anomaly detection)
    - **No Data Leakage**: Strict temporal split with proper validation
    - **Explainability**: SHAP and LIME for model interpretation
    - **Interactive Analysis**: Explore predictions, thresholds, and feature importance
    ''')

    st.markdown('''### 📋 Requirements''')
    st.markdown('''
    - CSV file with network traffic data
    - Must contain a 'Label' column (BENIGN or attack types)
    - Numeric features for classification
    ''')
    
    st.markdown('''### ⚠️ Important Notes''')
    st.markdown('''
    - **Default file**: Uses `merged_output.csv` from current directory
    - **Upload**: Or upload your own CSV file in the sidebar
    - **Processing time**: May take 5-15 minutes depending on dataset size
    - **First run**: Will take longer due to model training
    ''')
    
    st.stop()  # Stop execution here if button not pressed

# Main execution
if run_btn:
    import os
    
    # Determine data path - Priority: Upload > Google Drive > Local Path
    path = None
    
    if uploaded:
        # Option 1: File uploaded via sidebar
        path = _persist_upload_to_tmp(uploaded.read())
        st.success(f"✓ Uploaded file: {uploaded.name}")
        
    elif use_gdrive and gdrive_url:
        # Option 2: Download from Google Drive
        path = download_and_cache_dataset(gdrive_url)
        if not path:
            st.error("❌ Failed to download from Google Drive")
            st.info("💡 Try uploading the file directly or use a local path")
            st.stop()
            
    else:
        # Option 3: Local file path
        path = default_path
        
        # Convert Windows backslashes to forward slashes
        path = path.replace('\\', '/')
        
        # Try to resolve relative paths
        if not os.path.isabs(path):
            # Try in current directory
            if os.path.exists(path):
                path = os.path.abspath(path)
            # Try in parent directory
            elif os.path.exists(os.path.join('..', path)):
                path = os.path.abspath(os.path.join('..', path))
            # Try common data directories
            elif os.path.exists(os.path.join('data', path)):
                path = os.path.abspath(os.path.join('data', path))
        
        st.info(f"📂 Using path: `{path}`")
    
    # Check if file exists
    if not os.path.exists(path):
        st.error(f"❌ Error: File not found!")
        st.error(f"**Tried path:** `{path}`")
        st.error("")
        st.markdown("### 💡 How to fix:")
        st.markdown("**Option 1:** Check the **📥 Use Google Drive** box and paste your Google Drive link")
        st.markdown("**Option 2:** Upload your CSV file using the file uploader")
        st.markdown("**Option 3:** Put file in same folder as app.py:")
        st.code("merged_output.csv")
        
        with st.expander("📋 Google Drive Setup Instructions"):
            st.markdown("""
            ### How to use Google Drive for 700MB file:
            
            1. **Upload to Google Drive:**
               - Go to drive.google.com
               - Upload your `merged_output.csv` (700MB)
            
            2. **Get shareable link:**
               - Right-click the file
               - Click "Share"
               - Set to "Anyone with the link" can view
               - Click "Copy link"
            
            3. **Paste in Streamlit:**
               - Check "📥 Use Google Drive" box
               - Paste the link
               - Click "🚀 Run Pipeline"
            
            **First download takes 1-2 min, then cached!**
            """)
        
        with st.expander("🔍 Current Directory Info"):
            st.code(f"Current working directory:\n{os.getcwd()}")
            st.code("Files in current directory:")
            try:
                files = [f for f in os.listdir('.') if f.endswith('.csv')]
                if files:
                    st.code('\n'.join(files))
                else:
                    st.code("No CSV files found in current directory")
            except Exception as e:
                st.code(f"Error listing files: {e}")
        
        st.stop()
    
    # Check file size
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    st.info(f"📊 Dataset size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 500:
        st.warning("⚠️ Large dataset detected! Processing may take 10-20 minutes.")
        st.warning("Consider using a smaller sample for initial testing.")

    # Run pipeline with better progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Step 1/5: Loading dataset...")
        progress_bar.progress(20)
        
        status_text.text("🔄 Step 2/5: Training models...")
        progress_bar.progress(40)
        
        pipe, art = _run_pipeline(
            path, test_size, xgb_threshold, ocsvm_nu, ocsvm_sample, linear
        )
        
        status_text.text("🔄 Step 3/5: Generating predictions...")
        progress_bar.progress(60)
        
        status_text.text("🔄 Step 4/5: Computing metrics...")
        progress_bar.progress(80)
        
        status_text.text("🔄 Step 5/5: Preparing visualizations...")
        progress_bar.progress(100)
        
        status_text.empty()
        progress_bar.empty()
        
    except FileNotFoundError as e:
        st.error(f"❌ File Error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error during pipeline execution:")
        st.exception(e)
        st.stop()

    st.success("✅ Pipeline completed successfully!")

    # ========== METRICS SECTION ==========
    st.header("📊 Performance Metrics")
    res = art["results"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{res['accuracy']:.3f}")
    col2.metric("Precision", f"{res['precision']:.3f}")
    col3.metric("Recall", f"{res['recall']:.3f}")
    col4.metric("F1-Score", f"{res['f1_score']:.3f}")
    col5.metric("ROC-AUC", f"{(res['roc_auc'] or 0):.3f}")

    st.markdown("---")

    # Detailed metrics
    with st.expander("📈 Detailed Confusion Matrix Metrics"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("True Positives", f"{res['true_positives']:,}")
            st.metric("True Negatives", f"{res['true_negatives']:,}")
        with col_b:
            st.metric("False Positives", f"{res['false_positives']:,}")
            st.metric("False Negatives", f"{res['false_negatives']:,}")

        st.metric("False Positive Rate", f"{res['fpr']:.4f}")
        st.metric("False Negative Rate", f"{res['fnr']:.4f}")
        st.metric("Training Time", f"{res['training_time']:.2f}s")

    # ========== VISUALIZATION SECTION ==========
    st.header("📉 Model Visualizations")

    y_true = art["y_test"]
    y_pred = art["y_pred_voting"]
    proba_vote = art["proba_voting"]
    proba_xgb = art["proba_xgb"]

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, y_pred)
        st.pyplot(fig_cm)

    with col_2:
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_true, proba_vote)
        st.pyplot(fig_roc)

    with col_3:
        st.subheader("Precision-Recall Curve")
        fig_pr = plot_pr_curve(y_true, proba_vote)
        st.pyplot(fig_pr)

    st.markdown("---")

    # ========== THRESHOLD ANALYSIS ==========
    st.header("🎯 Threshold Sweep Analysis")
    st.markdown("Analyze how different thresholds affect model performance")

    table = threshold_sweep(y_true, proba_vote, num=50)
    fig_sweep = plot_threshold_sweep(table)
    st.pyplot(fig_sweep)

    sweep_df = pd.DataFrame(table, columns=["threshold", "precision", "recall", "f1"])

    with st.expander("📊 View Threshold Data"):
        st.dataframe(
            sweep_df.style.format("{:.4f}"), use_container_width=True
        )

        csv = sweep_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Threshold Data",
            csv,
            "threshold_sweep.csv",
            "text/csv"
        )

    st.markdown("---")

    # ========== SHAP GLOBAL EXPLANATIONS ==========
    st.header("🔍 Global Model Explanations (SHAP)")
    st.markdown("Understanding which features drive XGBoost predictions globally")

    X_test_s = art["X_test_scaled"]
    
    # Determine safe default for SHAP
    if len(X_test_s) > 10000:
        default_shap = 500
        st.warning("⚠️ Large test set detected. Using 500 samples for SHAP by default to prevent timeout.")
    else:
        default_shap = min(1000, len(X_test_s))
    
    max_rows = st.slider(
        "Number of test samples for SHAP analysis",
        100,
        min(2000, len(X_test_s)),  # Cap at 2000 to prevent hanging
        default_shap,
        100
    )
    
    if max_rows > 1500:
        st.warning("⚠️ Computing SHAP for >1500 samples may take several minutes...")

    explainer, _ = _shap_setup(
        art["xgb_model"],
        art["X_train_scaled"],
        art["feature_names"]
    )

    idx = np.arange(len(X_test_s))[:max_rows]

    with st.spinner(f"Computing SHAP values for {max_rows} samples... This may take a moment."):
        try:
            shap_vals = compute_shap_values(explainer, X_test_s[idx])
        except Exception as e:
            st.error(f"❌ SHAP computation failed: {str(e)}")
            st.error("Try reducing the number of samples or skip SHAP analysis.")
            shap_vals = None

    if shap_vals is not None:
        col_shap1, col_shap2 = st.columns(2)

        with col_shap1:
            st.subheader("Feature Importance (Bar)")
            fig_bar = plot_shap_bar(shap_vals, X_test_s[idx], art["feature_names"])
            st.pyplot(fig_bar)

        with col_shap2:
            st.subheader("Feature Impact (Beeswarm)")
            fig_bee = plot_shap_beeswarm(shap_vals, X_test_s[idx], art["feature_names"])
            st.pyplot(fig_bee)
    else:
        st.info("⚠️ SHAP analysis skipped due to errors. Other features still available.")

    st.markdown("---")

    # ========== LOCAL EXPLANATIONS ==========
    st.header("🎯 Local Explanations (Single Prediction)")
    st.markdown("Understand why the model made a specific prediction")

    row_idx = st.number_input(
        "Select test sample index",
        min_value=0,
        max_value=len(X_test_s) - 1,
        value=0,
        step=1
    )

    # Display prediction info
    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
    col_pred1.metric("True Label", "Attack" if y_true[row_idx] == 1 else "Benign")
    col_pred2.metric("XGB Prediction", "Attack" if art["y_pred_xgb"][row_idx] == 1 else "Benign")
    col_pred3.metric("OCSVM Prediction", "Attack" if art["y_pred_ocsvm"][row_idx] == 1 else "Benign")
    col_pred4.metric("Voting Prediction", "Attack" if y_pred[row_idx] == 1 else "Benign")

    st.markdown("---")

    # SHAP waterfall
    st.subheader("SHAP Waterfall (XGBoost)")
    st.markdown("Shows how each feature pushes the prediction from base value")
    fig_water = plot_shap_waterfall_for_row(
        explainer,
        X_test_s[row_idx],
        art["feature_names"]
    )
    st.pyplot(fig_water)

    st.markdown("---")

    # LIME explanation
    st.subheader("LIME Local Explanation")
    st.markdown("Alternative explanation method showing feature contributions")

    def predict_proba_for_lime(X_batch: np.ndarray) -> np.ndarray:
        p_attack = art["xgb_model"].predict_proba(X_batch)[:, 1]
        p_benign = 1 - p_attack
        return np.vstack([p_benign, p_attack]).T

    lime = LimeHelper(art["X_train_scaled"], art["feature_names"])
    lime_df = lime.explain_row(predict_proba_for_lime, X_test_s[row_idx], top_k=10)

    st.dataframe(
        lime_df.style.format({"weight": "{:.4f}"}), use_container_width=True
    )

    csv_lime = lime_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download LIME Explanation",
        csv_lime,
        f"lime_explanation_row_{row_idx}.csv",
        "text/csv"
    )

    st.markdown("---")

    # ========== PREDICTIONS TABLE ==========
    st.header("📋 Predictions Overview")

    out_df = pd.DataFrame({
        "y_true": y_true,
        "true_label": ["Attack" if y == 1 else "Benign" for y in y_true],
        "proba_xgb": proba_xgb,
        "proba_ocsvm": art["proba_ocsvm"],
        "proba_voting": proba_vote,
        "pred_xgb": ["Attack" if y == 1 else "Benign" for y in art["y_pred_xgb"]],
        "pred_ocsvm": ["Attack" if y == 1 else "Benign" for y in art["y_pred_ocsvm"]],
        "pred_voting": ["Attack" if y == 1 else "Benign" for y in y_pred],
        "correct": y_true == y_pred
    })

    # Filter options
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_option = st.selectbox(
            "Filter predictions",
            ["All", "Correct Only", "Incorrect Only", "Attack Only", "Benign Only"]
        )

    with col_f2:
        show_rows = st.number_input("Rows to display", 10, 500, 100, 10)

    # Apply filters
    if filter_option == "Correct Only":
        out_df_filtered = out_df[out_df["correct"]]
    elif filter_option == "Incorrect Only":
        out_df_filtered = out_df[~out_df["correct"]]
    elif filter_option == "Attack Only":
        out_df_filtered = out_df[out_df["y_true"] == 1]
    elif filter_option == "Benign Only":
        out_df_filtered = out_df[out_df["y_true"] == 0]
    else:
        out_df_filtered = out_df

    st.dataframe(
        out_df_filtered.head(show_rows).style.format({
            "proba_xgb": "{:.4f}",
            "proba_ocsvm": "{:.4f}",
            "proba_voting": "{:.4f}"
        }),
        use_container_width=True
    )

    csv_pred = out_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download All Predictions",
        csv_pred,
        "predictions.csv",
        "text/csv"
    )

else:
    # Initial state
    st.info("👈 Configure parameters in the sidebar and click **Run Pipeline** to start")

    st.markdown('''### 🎯 What This Dashboard Does''')
    st.markdown('''
    - **Hybrid Detection**: Combines XGBoost (supervised) and One-Class SVM (anomaly detection)
    - **No Data Leakage**: Strict temporal split with proper validation
    - **Explainability**: SHAP and LIME for model interpretation
    - **Interactive Analysis**: Explore predictions, thresholds, and feature importance
    ''')

    st.markdown('''### 📋 Requirements''')
    st.markdown('''
    - CSV file with network traffic data
    - Must contain a 'Label' column (BENIGN or attack types)
    - Numeric features for classification
    ''')