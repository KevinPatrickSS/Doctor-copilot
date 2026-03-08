"""
Demo Script: Complete Cardiac Risk Assessment
Shows end-to-end usage of the system with multiple patient examples
"""

from cardiac_risk_agent import CardiacRiskAgent
import json


def demo_assessment():
    """Demonstrate the complete risk assessment system."""
    
    print("\n" + "="*80)
    print("CARDIAC RISK ASSESSMENT SYSTEM - DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows the three-layer assessment on different patient scenarios")
    
    # Initialize the agent
    print("\nInitializing risk assessment agent...")
    agent = CardiacRiskAgent(models_dir='models')
    
    # Define test patients with different risk profiles
    patients = [
        {
            'name': 'Low Risk Patient',
            'patient_id': 'DEMO-001',
            'scenario': 'Young patient, no cardiac history, low-risk surgery',
            'data': {
                # RCRI fields
                'surgery_risk_level': 'low',
                'history_ihd': False,
                'history_hf': False,
                'history_cva': False,
                'diabetes_insulin': False,
                'creatinine_mg_dl': 0.9,
                # Clinical fields
                'age': 45,
                'sex': 'Female',
                'trestbps': 120,
                'chol': 200,
                'fbs_binary': 0,
                'thalch': 165,
                'oldpeak': 0.5,
                'exang_binary': 0,
                'lv_hypertrophy': 0
            }
        },
        {
            'name': 'Intermediate Risk Patient',
            'patient_id': 'DEMO-002',
            'scenario': 'Middle-aged patient, diabetes, intermediate-risk surgery',
            'data': {
                # RCRI fields
                'surgery_risk_level': 'intermediate',
                'history_ihd': False,
                'history_hf': False,
                'history_cva': False,
                'diabetes_insulin': True,
                'creatinine_mg_dl': 1.5,
                # Clinical fields
                'age': 62,
                'sex': 'Male',
                'trestbps': 145,
                'chol': 230,
                'fbs_binary': 1,
                'thalch': 135,
                'oldpeak': 1.8,
                'exang_binary': 0,
                'lv_hypertrophy': 0
            }
        },
        {
            'name': 'High Risk Patient',
            'patient_id': 'DEMO-003',
            'scenario': 'Elderly patient, cardiac history, high-risk surgery',
            'data': {
                # RCRI fields
                'surgery_risk_level': 'high',
                'history_ihd': True,
                'history_hf': True,
                'history_cva': False,
                'diabetes_insulin': True,
                'creatinine_mg_dl': 2.3,
                # Clinical fields
                'age': 72,
                'sex': 'Male',
                'trestbps': 160,
                'chol': 270,
                'fbs_binary': 1,
                'thalch': 105,
                'oldpeak': 3.2,
                'exang_binary': 1,
                'lv_hypertrophy': 1
            }
        }
    ]
    
    # Assess each patient
    all_results = []
    
    for i, patient_info in enumerate(patients, 1):
        print("\n" + "="*80)
        print(f"PATIENT {i}/3: {patient_info['name']}")
        print("="*80)
        print(f"Patient ID: {patient_info['patient_id']}")
        print(f"Scenario: {patient_info['scenario']}")
        
        # Add patient_id to data
        patient_info['data']['patient_id'] = patient_info['patient_id']
        
        # Perform assessment
        result = agent.assess_patient(patient_info['data'])
        all_results.append(result)
        
        # Generate and display report
        report = agent.generate_report(result)
        print("\n" + report)
        
        # Wait for user
        if i < len(patients):
            input("\nPress Enter to continue to next patient...")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON OF ALL PATIENTS")
    print("="*80)
    print(f"\n{'Patient ID':<15} {'RCRI':<10} {'ML Risk':<12} {'Overall':<15} {'Agreement':<10}")
    print("-"*80)
    
    for result in all_results:
        patient_id = result['patient_id']
        rcri_class = result['rcri']['rcri_risk_class']
        ml_prob = result['ml_prediction']['ml_risk_probability'] if result['ml_prediction'] else 0
        ml_class = result['ml_prediction']['ml_risk_class'] if result['ml_prediction'] else 'N/A'
        overall = result['combined_assessment']['overall_risk_class']
        agreement = result['combined_assessment']['agreement']
        
        print(f"{patient_id:<15} {rcri_class:<10} {f'{ml_prob:.1%}':<12} {overall:<15} {str(agreement):<10}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. RCRI provides fast, rule-based risk stratification
2. ML models capture complex patterns in patient data
3. SHAP explanations show which features drive each prediction
4. Combined assessment increases confidence when models agree
5. Disagreements between RCRI and ML warrant closer clinical review
    """)
    
    # Export results
    print("\nExporting results to JSON...")
    export_results(all_results)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated reports")
    print("2. Examine the exported JSON for integration")
    print("3. Customize the system for your specific needs")
    print("4. Validate on your institution's data")


def export_results(results):
    """Export assessment results to JSON file."""
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    # Convert results
    export_data = {
        'assessment_results': [convert_to_native(r) for r in results],
        'metadata': {
            'system_version': '1.0',
            'model_type': 'RCRI + ML Ensemble + SHAP',
            'description': 'Cardiac risk assessment demonstration results'
        }
    }
    
    # Save to file
    filename = 'demo_results.json'
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Results exported to: {filename}")


def show_system_info():
    """Display information about the system."""
    print("\n" + "="*80)
    print("SYSTEM ARCHITECTURE")
    print("="*80)
    print("""
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: RCRI Calculator                                    │
│ ✓ 6 binary risk factors                                    │
│ ✓ Evidence-based scoring                                   │
│ ✓ Risk stratification                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: ML Models                                          │
│ ✓ Logistic Regression (interpretable)                      │
│ ✓ XGBoost (high performance)                               │
│ ✓ Ensemble averaging                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: SHAP Explainability                               │
│ ✓ Patient-level explanations                               │
│ ✓ Feature importance                                       │
│ ✓ Human-readable text                                      │
└─────────────────────────────────────────────────────────────┘

RCRI Risk Factors:
  1. High-risk surgery (intraperitoneal, intrathoracic, vascular)
  2. Ischemic heart disease (MI, angina, positive stress test)
  3. Heart failure (CHF, pulmonary edema)
  4. Cerebrovascular disease (stroke, TIA)
  5. Diabetes on insulin
  6. Renal insufficiency (creatinine > 2.0 mg/dL)

ML Features (15 total):
  • Demographics: age, sex
  • Cardiac markers: angina, exercise angina, LV hypertrophy
  • Vital signs: blood pressure, heart rate, ST depression
  • Labs: cholesterol, fasting blood sugar
  • Derived: age>65, hypertension, reduced exercise capacity

Risk Classification:
  • Low: RCRI=0, ML<5%
  • Intermediate: RCRI=1-2, ML=5-15%
  • High: RCRI≥3, ML>15%
    """)


if __name__ == "__main__":
    import sys
    
    # Check if models exist
    from pathlib import Path
    if not Path('models').exists() or not Path('models/cardiac_risk_model_xgboost.pkl').exists():
        print("\n" + "="*80)
        print("ERROR: Trained models not found!")
        print("="*80)
        print("\nPlease train the models first:")
        print("  python train_pipeline.py heart_disease.csv")
        print("\nThis will create the 'models/' directory with trained models.")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Show system info
    show_system_info()
    
    # Wait for user to start
    print("\n" + "="*80)
    input("Press Enter to start the demonstration...")
    
    # Run demo
    try:
        demo_assessment()
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        print("Make sure you have:")
        print("1. Trained the models (python train_pipeline.py heart_disease.csv)")
        print("2. Installed all dependencies (pip install -r requirements.txt)")
        raise