# local_train.py - Local training script matching your third file
"""
Local Training Script - Run this on your local machine
Trains models on full 700MB dataset and saves everything for Streamlit deployment
"""

from pipeline import OptimizedVotingHybridPipeline
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime

def train_and_save_models(data_path="merged_output.csv"):
    """Train models locally and save everything needed for Streamlit"""
    
    print("="*80)
    print("LOCAL TRAINING MODE - Full Dataset")
    print("="*80)
    print("This will train on your local machine and save models for Streamlit")
    print("="*80)
    
    # Configuration - MATCHING YOUR THIRD FILE
    XGB_THRESHOLD = 0.2 # Same as third file default
    OCSVM_NU = 0.2  # Same as third file default
    OCSVM_SAMPLE_RATIO = 0.3
    USE_LINEAR_KERNEL = False
    TEST_SIZE = 0.3
    
    print(f"\nConfiguration:")
    print(f"  XGBoost Threshold: {XGB_THRESHOLD}")
    print(f"  OCSVM Nu: {OCSVM_NU}")
    print(f"  OCSVM Sampling: {OCSVM_SAMPLE_RATIO*100:.0f}%")
    print(f"  Kernel: {'Linear' if USE_LINEAR_KERNEL else 'RBF'}")
    
    # Initialize pipeline
    pipeline = OptimizedVotingHybridPipeline(
        data_path=data_path,
        xgb_threshold=XGB_THRESHOLD,
        ocsvm_nu=OCSVM_NU,
        ocsvm_sample_ratio=OCSVM_SAMPLE_RATIO,
        use_linear_kernel=USE_LINEAR_KERNEL
    )
    
    # Run full pipeline
    print("\n🚀 Starting training...")
    artifacts = pipeline.run_pipeline()
    
    # Prepare data to save
    print("\n📦 Preparing files for Streamlit deployment...")
    
    # 1. Save trained models
    models = {
        'xgb_model': pipeline.xgb_model,
        'ocsvm_model': pipeline.ocsvm_model,
        'scaler': pipeline.scaler,
        'feature_names': pipeline.feature_names,
        'xgb_threshold': XGB_THRESHOLD,
        'config': {
            'ocsvm_nu': OCSVM_NU,
            'test_size': TEST_SIZE,
            'use_linear_kernel': USE_LINEAR_KERNEL,
            'training_date': datetime.now().isoformat()
        }
    }
    
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("✓ Saved: trained_models.pkl")
    
    # 2. Save metrics/results
    results_for_streamlit = {
        'results': artifacts['results'],
        'training_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(pipeline.df),
            'train_samples': len(pipeline.y_train),
            'test_samples': len(pipeline.y_test),
            'features': len(pipeline.feature_names),
            'class_distribution': pipeline.df['Label'].value_counts().to_dict()
        }
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(results_for_streamlit, f, indent=2, default=str)
    print("✓ Saved: metrics.json")
    
    # 3. Save sample predictions
    sample_size = min(1000, len(pipeline.y_test))
    sample_indices = np.random.choice(len(pipeline.y_test), sample_size, replace=False)
    
    sample_predictions = {
        'y_true': pipeline.y_test.values[sample_indices].tolist(),
        'y_pred_xgb': pipeline.y_pred_xgb[sample_indices].tolist(),
        'y_pred_ocsvm': pipeline.y_pred_ocsvm[sample_indices].tolist(),
        'y_pred_voting': pipeline.y_pred_voting[sample_indices].tolist(),
        'proba_xgb': pipeline.proba_xgb[sample_indices].tolist(),
        'proba_ocsvm': pipeline.proba_ocsvm[sample_indices].tolist(),
        'proba_voting': pipeline.proba_voting[sample_indices].tolist()
    }
    
    with open('sample_predictions.json', 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    print("✓ Saved: sample_predictions.json")
    
    # 4. Save feature statistics
    X_train_df = pd.DataFrame(pipeline.X_train_scaled, columns=pipeline.feature_names)
    
    feature_stats = {
        'means': X_train_df.mean().to_dict(),
        'stds': X_train_df.std().to_dict(),
        'mins': X_train_df.min().to_dict(),
        'maxs': X_train_df.max().to_dict()
    }
    
    with open('feature_stats.json', 'w') as f:
        json.dump(feature_stats, f, indent=2, default=str)
    print("✓ Saved: feature_stats.json")
    
    # Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE - Files Ready for Streamlit")
    print("="*80)
    print("\nFiles created:")
    print("  1. trained_models.pkl")
    print("  2. metrics.json")
    print("  3. sample_predictions.json")
    print("  4. feature_stats.json")
    
    print("\n📊 Model Performance:")
    print(f"  Accuracy:  {artifacts['results']['accuracy']:.4f}")
    print(f"  Precision: {artifacts['results']['precision']:.4f}")
    print(f"  Recall:    {artifacts['results']['recall']:.4f}")
    print(f"  F1-Score:  {artifacts['results']['f1_score']:.4f}")
    
    print("\n🎉 Ready for deployment!")
    print("="*80)
    
    return artifacts


if __name__ == "__main__":
    import os
    
    data_file = "merged_output.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        print(f"Current directory: {os.getcwd()}")
        exit(1)
    
    train_and_save_models(data_file)