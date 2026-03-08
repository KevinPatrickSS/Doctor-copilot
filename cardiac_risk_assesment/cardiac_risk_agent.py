"""
Integrated Cardiac Risk Assessment Agent
Combines RCRI rules + ML models + SHAP explanations
"""

import numpy as np
import pickle
from .rcri_calculator import RCRICalculator
from .ml_models import CardiacRiskMLModel
from .shap_explainer import SHAPExplainer
from .data_preprocessing import HeartDiseasePreprocessor


class CardiacRiskAgent:
    """
    Complete cardiac risk assessment agent that integrates:
    1. RCRI (Revised Cardiac Risk Index) - deterministic rules
    2. ML models (Logistic + XGBoost ensemble) - data-driven predictions
    3. SHAP explanations - interpretability layer
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the risk agent.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        
        # Initialize components
        self.rcri_calculator = RCRICalculator()
        self.ml_model = CardiacRiskMLModel()
        self.preprocessor = HeartDiseasePreprocessor()
        self.shap_explainer = None
        
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and preprocessors."""
        try:
            # Load preprocessor
            self.preprocessor.load_preprocessor(f'{self.models_dir}/preprocessor.pkl')
            
            # Load ML models
            self.ml_model.load_models(f'{self.models_dir}/cardiac_risk_model')
            
            # Load SHAP explainer (using XGBoost version)
            with open(f'{self.models_dir}/shap_xgb_explainer.pkl', 'rb') as f:
                explainer_data = pickle.load(f)
                self.shap_explainer = SHAPExplainer(
                    self.ml_model.xgb_model,
                    model_type='tree',
                    feature_names=explainer_data['feature_names']
                )
                self.shap_explainer.explainer = explainer_data['explainer']
                self.shap_explainer.base_value = explainer_data['base_value']
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you've run train_pipeline.py first!")
    
    def assess_patient(self, patient_data):
        """
        Perform complete risk assessment for a patient.
        
        Args:
            patient_data: Dictionary containing both:
                - RCRI fields (surgery_risk_level, history_ihd, etc.)
                - Clinical fields (age, sex, cholesterol, etc.)
                
        Returns:
            dict: Complete risk assessment with RCRI, ML, and SHAP results
        """
        print("\n" + "="*80)
        print("CARDIAC RISK ASSESSMENT")
        print("="*80)
        
        # ====================================================================
        # LAYER 1: RCRI Calculation
        # ====================================================================
        print("\n" + "-"*80)
        print("LAYER 1: RCRI (Revised Cardiac Risk Index)")
        print("-"*80)
        
        rcri_result = self.rcri_calculator.calculate_score(patient_data)
        
        print(rcri_result['summary'])
        
        # ====================================================================
        # LAYER 2: ML Model Prediction
        # ====================================================================
        print("\n" + "-"*80)
        print("LAYER 2: Machine Learning Risk Prediction")
        print("-"*80)
        
        # Prepare features for ML model
        X_patient = self._prepare_ml_features(patient_data)
        
        if X_patient is not None:
            # Get ML prediction
            ml_result = self.ml_model.predict_single_patient(X_patient)
            
            print(f"\nEnsemble Model Results:")
            print(f"  Risk Probability: {ml_result['ml_risk_probability']:.1%}")
            print(f"  Risk Class: {ml_result['ml_risk_class'].upper()}")
            print(f"\nModel Breakdown:")
            print(f"  Logistic Regression: {ml_result['logistic_probability']:.1%}")
            print(f"  XGBoost: {ml_result['xgb_probability']:.1%}")
            
            # ================================================================
            # LAYER 3: SHAP Explanation
            # ================================================================
            print("\n" + "-"*80)
            print("LAYER 3: SHAP Explainability")
            print("-"*80)
            
            if self.shap_explainer is not None:
                shap_result = self.shap_explainer.explain_prediction(X_patient, top_n=5)
                
                print(f"\n{shap_result['explanation_text']}")
                
                # Show top feature contributions
                print("\nTop 5 Feature Contributions:")
                for i, (feature, data) in enumerate(list(shap_result['top_features'].items()), 1):
                    shap_val = data['shap_value']
                    direction = "↑" if shap_val > 0 else "↓"
                    print(f"  {i}. {feature}: {shap_val:+.4f} {direction}")
            else:
                shap_result = None
                print("SHAP explainer not available")
        else:
            ml_result = None
            shap_result = None
            print("Insufficient data for ML prediction")
        
        # ====================================================================
        # Combined Assessment
        # ====================================================================
        print("\n" + "="*80)
        print("COMBINED RISK ASSESSMENT")
        print("="*80)
        
        combined_risk = self._combine_risk_assessments(rcri_result, ml_result)
        
        print(f"\nRCRI Score: {rcri_result['rcri_score']}/6 ({rcri_result['rcri_risk_class']})")
        if ml_result:
            print(f"ML Model: {ml_result['ml_risk_probability']:.1%} probability ({ml_result['ml_risk_class']})")
        print(f"\nOverall Assessment: {combined_risk['overall_risk_class'].upper()}")
        print(f"Recommendation: {combined_risk['recommendation']}")
        
        # ====================================================================
        # Return structured results
        # ====================================================================
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'rcri': rcri_result,
            'ml_prediction': ml_result,
            'shap_explanation': shap_result,
            'combined_assessment': combined_risk,
            'timestamp': self._get_timestamp()
        }
    
    def _prepare_ml_features(self, patient_data):
        """
        Convert patient data to ML feature vector.
        
        Args:
            patient_data: Raw patient data dictionary
            
        Returns:
            numpy array: Scaled feature vector
        """
        try:
            # Extract features in the correct order
            feature_values = []
            for feature_name in self.preprocessor.feature_names:
                # Map feature names to patient data fields
                value = self._get_feature_value(patient_data, feature_name)
                feature_values.append(value)
            
            # Convert to numpy array
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            X_scaled = self.preprocessor.scaler.transform(X)
            
            return X_scaled
            
        except Exception as e:
            print(f"Error preparing ML features: {e}")
            return None
    
    def _get_feature_value(self, patient_data, feature_name):
        """Map feature names to patient data fields."""
        # Direct mappings
        if feature_name in patient_data:
            return patient_data[feature_name]
        
        # Derived features
        if feature_name == 'age_over_65':
            return 1 if patient_data.get('age', 0) >= 65 else 0
        elif feature_name == 'is_male':
            sex = patient_data.get('sex', '').lower()
            return 1 if sex in ['male', 'm', '1'] else 0
        elif feature_name == 'has_angina':
            return 1 if patient_data.get('history_ihd', False) else 0
        elif feature_name == 'high_cholesterol':
            return 1 if patient_data.get('chol', 0) > 240 else 0
        elif feature_name == 'hypertension':
            return 1 if patient_data.get('trestbps', 0) > 140 else 0
        
        # Default to 0 if not found
        return 0
    
    def _combine_risk_assessments(self, rcri_result, ml_result):
        """
        Combine RCRI and ML assessments into overall risk.
        
        Args:
            rcri_result: RCRI calculation results
            ml_result: ML prediction results
            
        Returns:
            dict: Combined risk assessment
        """
        rcri_class = rcri_result['rcri_risk_class']
        
        if ml_result is None:
            # Only RCRI available
            overall_class = rcri_class
            confidence = "moderate (RCRI only)"
        else:
            ml_class = ml_result['ml_risk_class']
            
            # Combine the assessments
            if rcri_class == 'high' or ml_class == 'high':
                overall_class = 'high'
                confidence = "high" if rcri_class == ml_class else "moderate"
            elif rcri_class == 'low' and ml_class == 'low':
                overall_class = 'low'
                confidence = "high"
            else:
                overall_class = 'intermediate'
                confidence = "moderate"
        
        # Generate recommendation
        recommendations = {
            'low': "Patient is suitable for surgery with standard perioperative care.",
            'intermediate': "Consider additional cardiac evaluation. Optimize medical management before surgery.",
            'high': "Strongly consider cardiology consultation. May need stress testing or coronary angiography before high-risk surgery."
        }
        
        return {
            'overall_risk_class': overall_class,
            'confidence': confidence,
            'recommendation': recommendations[overall_class],
            'agreement': rcri_class == ml_result['ml_risk_class'] if ml_result else "N/A"
        }
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_report(self, assessment_result):
        """
        Generate a formatted text report from assessment results.
        
        Args:
            assessment_result: Output from assess_patient()
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("="*80)
        report.append("PERIOPERATIVE CARDIAC RISK ASSESSMENT REPORT")
        report.append("="*80)
        report.append(f"\nPatient ID: {assessment_result['patient_id']}")
        report.append(f"Assessment Date: {assessment_result['timestamp']}")
        
        # RCRI Section
        report.append("\n" + "-"*80)
        report.append("REVISED CARDIAC RISK INDEX (RCRI)")
        report.append("-"*80)
        rcri = assessment_result['rcri']
        report.append(f"Score: {rcri['rcri_score']}/6")
        report.append(f"Classification: {rcri['rcri_risk_class'].upper()}")
        report.append(f"Estimated Event Rate: {rcri['rcri_event_rate']}%")
        
        if rcri['factors_present']:
            report.append(f"\nRisk Factors Present ({len(rcri['factors_present'])}):")
            for i, factor in enumerate(rcri['factors_present'], 1):
                report.append(f"  {i}. {factor['factor']}")
        
        # ML Section
        if assessment_result['ml_prediction']:
            report.append("\n" + "-"*80)
            report.append("MACHINE LEARNING RISK PREDICTION")
            report.append("-"*80)
            ml = assessment_result['ml_prediction']
            report.append(f"Risk Probability: {ml['ml_risk_probability']:.1%}")
            report.append(f"Risk Classification: {ml['ml_risk_class'].upper()}")
        
        # SHAP Section
        if assessment_result['shap_explanation']:
            report.append("\n" + "-"*80)
            report.append("KEY RISK DRIVERS (SHAP Analysis)")
            report.append("-"*80)
            report.append(assessment_result['shap_explanation']['explanation_text'])
        
        # Combined Assessment
        report.append("\n" + "="*80)
        report.append("OVERALL ASSESSMENT")
        report.append("="*80)
        combined = assessment_result['combined_assessment']
        report.append(f"Risk Level: {combined['overall_risk_class'].upper()}")
        report.append(f"Confidence: {combined['confidence']}")
        report.append(f"\nRecommendation:")
        report.append(f"{combined['recommendation']}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


def create_sample_patient():
    """Create a sample patient for demonstration."""
    return {
        'patient_id': 'DEMO-001',
        
        # RCRI fields
        'surgery_risk_level': 'high',
        'history_ihd': True,
        'history_hf': False,
        'history_cva': False,
        'diabetes_insulin': True,
        'creatinine_mg_dl': 2.1,
        
        # Clinical fields for ML
        'age': 68,
        'sex': 'Male',
        'trestbps': 150,
        'chol': 250,
        'fbs_binary': 1,
        'thalch': 110,
        'oldpeak': 2.5,
        'exang_binary': 1,
        'lv_hypertrophy': 1
    }


if __name__ == "__main__":
    print("\nCardiac Risk Assessment Agent")
    print("="*80)
    
    # Initialize agent
    agent = CardiacRiskAgent(models_dir='models')
    
    # Create sample patient
    patient = create_sample_patient()
    
    print("\nAssessing sample patient...")
    
    # Perform assessment
    result = agent.assess_patient(patient)
    
    # Generate report
    print("\n\n")
    report = agent.generate_report(result)
    print(report)