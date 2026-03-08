"""
RCRI (Revised Cardiac Risk Index) Calculator
Implements deterministic rules for cardiac risk assessment
"""

class RCRICalculator:
    """
    Revised Cardiac Risk Index (RCRI) calculator for perioperative cardiac risk.
    
    The RCRI uses 6 binary factors to predict major cardiac complications:
    1. High-risk surgery
    2. Ischemic heart disease
    3. Heart failure
    4. Cerebrovascular disease
    5. Diabetes on insulin
    6. Creatinine > 2.0 mg/dL
    """
    
    # Risk class thresholds and event rates based on published literature
    RISK_CLASSES = {
        0: {
            "class": "low",
            "event_rate": 0.4,
            "description": "Very low risk"
        },
        1: {
            "class": "intermediate",
            "event_rate": 0.9,
            "description": "Low risk"
        },
        2: {
            "class": "intermediate",
            "event_rate": 6.6,
            "description": "Intermediate risk"
        },
        3: {
            "class": "high",
            "event_rate": 11.0,
            "description": "High risk"
        }
    }
    
    def __init__(self):
        self.factors = []
        
    def calculate_score(self, patient_data):
        """
        Calculate RCRI score from patient data.
        
        Args:
            patient_data (dict): Dictionary containing:
                - surgery_risk_level: "high", "intermediate", or "low"
                - history_ihd: bool (ischemic heart disease)
                - history_hf: bool (heart failure)
                - history_cva: bool (cerebrovascular disease)
                - diabetes_insulin: bool (diabetes on insulin)
                - creatinine_mg_dl: float (serum creatinine)
                
        Returns:
            dict: Contains score, risk_class, event_rate, and factors
        """
        factors_present = []
        score = 0
        
        # Factor 1: High-risk surgery
        if patient_data.get('surgery_risk_level', '').lower() == 'high':
            score += 1
            factors_present.append({
                "factor": "High-risk surgery",
                "description": "Intraperitoneal, intrathoracic, or suprainguinal vascular"
            })
        
        # Factor 2: Ischemic heart disease
        if patient_data.get('history_ihd', False):
            score += 1
            factors_present.append({
                "factor": "Ischemic heart disease",
                "description": "History of MI, positive stress test, or angina"
            })
        
        # Factor 3: Heart failure
        if patient_data.get('history_hf', False):
            score += 1
            factors_present.append({
                "factor": "Heart failure",
                "description": "History of CHF, pulmonary edema, or PND"
            })
        
        # Factor 4: Cerebrovascular disease
        if patient_data.get('history_cva', False):
            score += 1
            factors_present.append({
                "factor": "Cerebrovascular disease",
                "description": "History of stroke or TIA"
            })
        
        # Factor 5: Diabetes on insulin
        if patient_data.get('diabetes_insulin', False):
            score += 1
            factors_present.append({
                "factor": "Diabetes on insulin",
                "description": "Preoperative insulin therapy"
            })
        
        # Factor 6: Creatinine > 2.0 mg/dL
        creatinine = patient_data.get('creatinine_mg_dl', 0)
        if creatinine > 2.0:
            score += 1
            factors_present.append({
                "factor": "Renal insufficiency",
                "description": f"Creatinine {creatinine:.1f} mg/dL (> 2.0)"
            })
        
        # Map score to risk class
        risk_info = self._get_risk_class(score)
        
        return {
            "rcri_score": score,
            "rcri_risk_class": risk_info["class"],
            "rcri_event_rate": risk_info["event_rate"],
            "rcri_description": risk_info["description"],
            "factors_present": factors_present,
            "total_factors": len(factors_present),
            "summary": self._generate_summary(score, risk_info, factors_present)
        }
    
    def _get_risk_class(self, score):
        """Map RCRI score to risk classification."""
        if score >= 3:
            return self.RISK_CLASSES[3]
        else:
            return self.RISK_CLASSES.get(score, self.RISK_CLASSES[0])
    
    def _generate_summary(self, score, risk_info, factors):
        """Generate human-readable summary."""
        summary = f"RCRI score: {score}/6 ({risk_info['description']})\n"
        summary += f"Estimated cardiac event rate: {risk_info['event_rate']}%\n"
        
        if factors:
            summary += f"\nRisk factors present ({len(factors)}):\n"
            for i, factor in enumerate(factors, 1):
                summary += f"{i}. {factor['factor']} - {factor['description']}\n"
        else:
            summary += "\nNo RCRI risk factors present."
        
        return summary


def create_sample_patient_data():
    """Create sample patient data for testing."""
    return {
        "surgery_risk_level": "high",  # High-risk vascular surgery
        "history_ihd": True,            # Has history of MI
        "history_hf": False,            # No heart failure
        "history_cva": False,           # No stroke
        "diabetes_insulin": True,       # On insulin
        "creatinine_mg_dl": 2.3         # Elevated creatinine
    }


if __name__ == "__main__":
    # Test the calculator
    calculator = RCRICalculator()
    sample_patient = create_sample_patient_data()
    
    result = calculator.calculate_score(sample_patient)
    
    print("=" * 60)
    print("RCRI CARDIAC RISK ASSESSMENT")
    print("=" * 60)
    print(result["summary"])
    print("=" * 60)
    print(f"\nRisk Classification: {result['rcri_risk_class'].upper()}")