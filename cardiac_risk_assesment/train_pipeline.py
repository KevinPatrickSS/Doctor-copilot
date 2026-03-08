"""
Complete Training Pipeline for Cardiac Risk Assessment System
Trains RCRI + ML models + SHAP explainers
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Import our custom modules
from data_preprocessing import HeartDiseasePreprocessor
from ml_models import CardiacRiskMLModel
from shap_explainer import SHAPExplainer


def train_complete_pipeline(data_filepath, output_dir='models'):
    """
    Train the complete cardiac risk assessment pipeline.
    
    Args:
        data_filepath: Path to UCI heart disease CSV file
        output_dir: Directory to save trained models
    """
    print("\n" + "="*80)
    print("CARDIAC RISK ASSESSMENT - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load and Preprocess Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = HeartDiseasePreprocessor()
    
    # Load data
    df = preprocessor.load_data(data_filepath)
    if df is None:
        print("Error: Could not load data. Exiting.")
        return None
    
    # Prepare data
    (X_train, X_val, X_test, 
     y_train, y_val, y_test,
     X_train_unscaled, X_val_unscaled, X_test_unscaled) = preprocessor.prepare_data(df)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f'{output_dir}/preprocessor.pkl')
    
    # ========================================================================
    # STEP 2: Train ML Models
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: MACHINE LEARNING MODEL TRAINING")
    print("="*80)
    
    ml_model = CardiacRiskMLModel()
    
    # Train Logistic Regression
    ml_model.train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Train XGBoost
    ml_model.train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate ensemble on test set
    ml_model.evaluate_test_set(X_test, y_test)
    
    # Save models
    ml_model.save_models(f'{output_dir}/cardiac_risk_model')
    
    # ========================================================================
    # STEP 3: Feature Importance Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importance = ml_model.get_feature_importance(preprocessor.feature_names)
    
    if 'logistic' in importance:
        print("\nLogistic Regression - Top Features:")
        sorted_log = sorted(importance['logistic'].items(), 
                           key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_log[:10], 1):
            print(f"  {i}. {feat}: {imp:.4f}")
    
    if 'xgboost' in importance:
        print("\nXGBoost - Top Features:")
        sorted_xgb = sorted(importance['xgboost'].items(), 
                           key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_xgb[:10], 1):
            print(f"  {i}. {feat}: {imp:.4f}")
    
    # ========================================================================
    # STEP 4: Initialize SHAP Explainers
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: SHAP EXPLAINER INITIALIZATION")
    print("="*80)
    
    # Create SHAP explainer for XGBoost (tree-based)
    print("\nInitializing SHAP explainer for XGBoost...")
    shap_xgb = SHAPExplainer(
        ml_model.xgb_model, 
        model_type='tree',
        feature_names=preprocessor.feature_names
    )
    shap_xgb.initialize_explainer()
    
    # Create SHAP explainer for Logistic Regression (linear)
    print("\nInitializing SHAP explainer for Logistic Regression...")
    shap_logistic = SHAPExplainer(
        ml_model.logistic_model,
        model_type='linear',
        feature_names=preprocessor.feature_names
    )
    shap_logistic.initialize_explainer(X_train)
    
    # ========================================================================
    # STEP 5: Global Feature Importance from SHAP
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: GLOBAL SHAP FEATURE IMPORTANCE")
    print("="*80)
    
    print("\nComputing global importance for XGBoost model...")
    global_importance_xgb = shap_xgb.get_global_feature_importance(X_test)
    
    print("\nComputing global importance for Logistic model...")
    global_importance_log = shap_logistic.get_global_feature_importance(X_test)
    
    # Save explainers
    shap_xgb.save_explainer(f'{output_dir}/shap_xgb_explainer.pkl')
    shap_logistic.save_explainer(f'{output_dir}/shap_logistic_explainer.pkl')
    
    # ========================================================================
    # STEP 6: Test on Sample Patients
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAMPLE PATIENT PREDICTIONS")
    print("="*80)
    
    # Test on 3 random patients from test set
    sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{'='*60}")
        print(f"Sample Patient {i}")
        print('='*60)
        
        X_patient = X_test[idx:idx+1]
        y_true = y_test[idx]
        
        # ML Prediction
        ml_result = ml_model.predict_single_patient(X_patient)
        
        print(f"\nActual outcome: {'Disease' if y_true == 1 else 'No Disease'}")
        print(f"\nML Model Prediction:")
        print(f"  Ensemble probability: {ml_result['ml_risk_probability']:.3f}")
        print(f"  Risk class: {ml_result['ml_risk_class']}")
        print(f"  Logistic prob: {ml_result['logistic_probability']:.3f}")
        print(f"  XGBoost prob: {ml_result['xgb_probability']:.3f}")
        
        # SHAP Explanation
        explanation = shap_xgb.explain_prediction(X_patient, top_n=5)
        print(f"\nSHAP Explanation:")
        print(explanation['explanation_text'])
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nModels saved in: {output_dir}/")
    print("\nFiles created:")
    print(f"  - preprocessor.pkl")
    print(f"  - cardiac_risk_model_logistic.pkl")
    print(f"  - cardiac_risk_model_xgboost.pkl")
    print(f"  - cardiac_risk_model_thresholds.pkl")
    print(f"  - shap_xgb_explainer.pkl")
    print(f"  - shap_logistic_explainer.pkl")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Use the trained models for predictions")
    print("  2. Combine with RCRI calculations")
    print("  3. Generate patient risk reports")
    print("="*80 + "\n")
    
    return {
        'preprocessor': preprocessor,
        'ml_model': ml_model,
        'shap_xgb': shap_xgb,
        'shap_logistic': shap_logistic,
        'test_data': (X_test, y_test)
    }


if __name__ == "__main__":
    print("\nCardiac Risk Assessment - Training Pipeline")
    print("="*80)
    
    # Check if data file path is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Using data file: {data_file}")
        train_complete_pipeline(data_file)
    else:
        print("\nUsage: python train_pipeline.py <path_to_heart_disease.csv>")
        print("\nExample:")
        print("  python train_pipeline.py heart_disease.csv")
        print("\nThis will:")
        print("  1. Load and preprocess the UCI heart disease dataset")
        print("  2. Train Logistic Regression and XGBoost models")
        print("  3. Initialize SHAP explainers")
        print("  4. Save all trained components")
        print("  5. Test on sample patients")