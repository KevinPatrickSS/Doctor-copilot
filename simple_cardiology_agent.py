"""
Simple Cardiology NLP Agent - Standalone Version
=================================================
A simplified version that works without BioBERT for quick testing.
This can be used if the full BioBERT version has issues.

Place this file as: simple_cardiology_agent.py
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone


class SimpleCardiologyIngestionAgent:
    """
    Simplified cardiology NLP agent using rule-based extraction only.
    No BioBERT required - works out of the box.
    """
    
    def __init__(self):
        print("🔵 Initializing Simple Cardiology Agent (Rule-based)")
        self.name = "Simple Cardiology Ingestion Agent"
    
    def process(self, raw_text: str) -> Dict[str, Any]:
        """
        Process clinical text and extract structured information.
        
        Args:
            raw_text: Unstructured clinical note
            
        Returns:
            Structured dictionary with patient info, symptoms, etc.
        """
        text = raw_text.lower()
        
        # Extract patient demographics
        age = self._extract_age(raw_text)
        sex = self._extract_sex(raw_text)
        encounter_type = self._extract_encounter_type(text)
        
        # Extract symptoms
        symptoms = self._extract_symptoms(raw_text)
        
        # Extract risk factors
        risk_factors = self._extract_risk_factors(text)
        
        # Extract investigations
        investigations = self._extract_investigations(raw_text)
        
        # Extract medications
        medications = self._extract_medications(raw_text)
        
        # Extract entities
        extracted_entities = self._extract_entities(raw_text)
        
        # Check cardiology relevance
        is_relevant, relevance_score = self._check_cardiology_relevance(text)
        
        return {
            'patient': {
                'age': age,
                'sex': sex,
                'encounter_type': encounter_type,
                'encounter_date': None
            },
            'cardiology_relevant': is_relevant,
            'cardiology_relevance_score': relevance_score,
            'symptoms': symptoms,
            'risk_factors': risk_factors,
            'investigations': investigations,
            'medications': medications,
            'extracted_entities': extracted_entities,
            'metadata': {
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'agent_version': '1.0.0-simple',
                'model': 'Rule-based (no BioBERT)',
                'safety_notice': 'This output is for information extraction only.'
            }
        }
    
    def _extract_age(self, text: str) -> Optional[int]:
        """Extract patient age."""
        match = re.search(r'(\d{1,3})[\s-]?(?:year|yr|yo|y/o)[\s-]?old', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_sex(self, text: str) -> Optional[str]:
        """Extract patient sex."""
        if re.search(r'\b(?:male|man|gentleman)\b', text, re.IGNORECASE):
            if not re.search(r'\bfemale\b', text, re.IGNORECASE):
                return 'M'
        if re.search(r'\b(?:female|woman|lady)\b', text, re.IGNORECASE):
            return 'F'
        return None
    
    def _extract_encounter_type(self, text: str) -> Optional[str]:
        """Extract encounter type."""
        if re.search(r'\b(?:ed|emergency department|emergency room)\b', text):
            return 'ED'
        if re.search(r'\b(?:admission|admitted|inpatient)\b', text):
            return 'Inpatient'
        if re.search(r'\b(?:clinic|outpatient|office visit)\b', text):
            return 'Outpatient'
        return None
    
    def _extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptoms."""
        symptoms = []
        
        # Chest pain
        if re.search(r'\bchest pain\b', text, re.IGNORECASE):
            symptom = {'symptom': 'chest pain'}
            
            # Character
            if re.search(r'pressure|crushing', text, re.IGNORECASE):
                symptom['character'] = 'pressure'
            elif re.search(r'sharp', text, re.IGNORECASE):
                symptom['character'] = 'sharp'
            
            # Radiation
            if re.search(r'left arm', text, re.IGNORECASE):
                symptom['radiation'] = 'left arm'
            elif re.search(r'jaw', text, re.IGNORECASE):
                symptom['radiation'] = 'jaw'
            
            # Onset
            if re.search(r'acute|sudden', text, re.IGNORECASE):
                symptom['onset'] = 'acute'
            
            symptoms.append(symptom)
        
        # Dyspnea
        if re.search(r'\b(?:dyspnea|shortness of breath|short of breath|SOB)\b', text, re.IGNORECASE):
            symptoms.append({'symptom': 'dyspnea'})
        
        # Other symptoms
        for symptom_name in ['palpitations', 'syncope', 'edema', 'orthopnea', 'fatigue', 'nausea', 'diaphoresis']:
            if re.search(rf'\b{symptom_name}\b', text, re.IGNORECASE):
                symptoms.append({'symptom': symptom_name})
        
        return symptoms
    
    def _extract_risk_factors(self, text: str) -> List[Dict[str, Any]]:
        """Extract cardiovascular risk factors."""
        risk_factors = []
        
        # Hypertension
        if re.search(r'\b(?:hypertension|HTN|high blood pressure)\b', text, re.IGNORECASE):
            controlled = 'controlled' in text or 'well-controlled' in text
            risk_factors.append({'factor': 'hypertension', 'controlled': controlled})
        
        # Diabetes
        if re.search(r'\b(?:diabetes|DM|diabetic)\b', text, re.IGNORECASE):
            insulin_treated = 'insulin' in text
            risk_factors.append({'factor': 'diabetes', 'insulin_treated': insulin_treated})
        
        # Other risk factors
        for factor in ['smoking', 'smoker', 'hyperlipidemia', 'high cholesterol']:
            if re.search(rf'\b{factor}\b', text, re.IGNORECASE):
                risk_factors.append({'factor': factor})
        
        return risk_factors
    
    def _extract_investigations(self, text: str) -> List[Dict[str, Any]]:
        """Extract investigation results."""
        investigations = []
        
        # Troponin
        match = re.search(r'troponin[:\s]+([0-9.]+)\s*(?:ng/mL)?', text, re.IGNORECASE)
        if match:
            investigations.append({
                'test_type': 'troponin',
                'value': float(match.group(1)),
                'unit': 'ng/mL',
                'ref_upper_limit': 0.04
            })
        
        # BNP
        match = re.search(r'BNP[:\s]+([0-9.]+)\s*(?:pg/mL)?', text, re.IGNORECASE)
        if match:
            investigations.append({
                'test_type': 'BNP',
                'value': float(match.group(1)),
                'unit': 'pg/mL',
                'ref_upper_limit': 100
            })
        
        # Creatinine
        match = re.search(r'creatinine[:\s]+([0-9.]+)\s*(?:mg/dL)?', text, re.IGNORECASE)
        if match:
            investigations.append({
                'test_type': 'creatinine',
                'value': float(match.group(1)),
                'unit': 'mg/dL',
                'ref_upper_limit': 1.3
            })
        
        # Blood Pressure
        match = re.search(r'BP[:\s]+(\d+)[/-](\d+)\s*mmHg', text, re.IGNORECASE)
        if match:
            investigations.append({
                'test_type': 'blood_pressure',
                'systolic': int(match.group(1)),
                'diastolic': int(match.group(2)),
                'unit': 'mmHg'
            })
        
        # ECG findings
        if re.search(r'\bECG\b', text, re.IGNORECASE):
            ecg_findings = []
            if re.search(r'ST elevation', text, re.IGNORECASE):
                match = re.search(r'ST elevation.*?(?:leads?\s+)?([\w\d,\s-]+)', text, re.IGNORECASE)
                if match:
                    ecg_findings.append(f'ST elevation {match.group(1)}')
            
            if re.search(r'LBBB|left bundle branch block', text, re.IGNORECASE):
                ecg_findings.append('LBBB' if 'new' not in text.lower() else 'new LBBB')
            
            if ecg_findings:
                investigations.append({
                    'test_type': 'ecg',
                    'findings': ecg_findings
                })
        
        # Echo findings
        if re.search(r'\b(?:echo|echocardiography)\b', text, re.IGNORECASE):
            echo_data = {'test_type': 'echo'}
            
            # EF
            match = re.search(r'(?:ejection fraction|EF)[:\s]+(\d+)%?', text, re.IGNORECASE)
            if match:
                ef_value = int(match.group(1))
                echo_data['ejection_fraction'] = ef_value / 100 if ef_value > 1 else ef_value
            
            # Wall motion
            if re.search(r'hypokinesis', text, re.IGNORECASE):
                echo_data['wall_motion'] = 'global hypokinesis' if 'global' in text.lower() else 'hypokinesis'
            
            if len(echo_data) > 1:
                investigations.append(echo_data)
        
        return investigations
    
    def _extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """Extract medications."""
        medications = []
        
        med_patterns = {
            'aspirin': r'\baspirin\b',
            'atorvastatin': r'\batorvastatin\b',
            'lisinopril': r'\blisinopril\b',
            'metoprolol': r'\bmetoprolol\b',
            'metformin': r'\bmetformin\b',
            'furosemide': r'\bfurosemide\b',
            'insulin': r'\binsulin\b'
        }
        
        for med_name, pattern in med_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                med_data = {'drug': med_name}
                
                # Try to extract dose
                dose_match = re.search(rf'{pattern}[:\s]+(\d+)\s*(mg|units?)', text, re.IGNORECASE)
                if dose_match:
                    med_data['dose'] = f"{dose_match.group(1)} {dose_match.group(2)}"
                
                medications.append(med_data)
        
        return medications
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract key clinical entities."""
        entities = []
        
        # Diseases
        diseases = ['STEMI', 'NSTEMI', 'MI', 'myocardial infarction', 'heart failure', 
                   'atrial fibrillation', 'AF', 'CAD', 'ACS']
        
        for disease in diseases:
            if re.search(rf'\b{disease}\b', text, re.IGNORECASE):
                entities.append({
                    'entity': disease,
                    'type': 'DISEASE',
                    'confidence': 0.85
                })
        
        return entities
    
    def _check_cardiology_relevance(self, text: str) -> tuple:
        """Check if text is cardiology-relevant."""
        cardiac_keywords = [
            'chest pain', 'MI', 'STEMI', 'NSTEMI', 'heart failure', 'dyspnea',
            'troponin', 'BNP', 'ECG', 'echo', 'ejection fraction', 'cardiac',
            'cardiology', 'heart', 'angina', 'palpitations'
        ]
        
        matches = sum(1 for kw in cardiac_keywords if kw.lower() in text)
        score = min(matches / 5.0, 1.0)
        
        return score > 0.3, round(score, 2)


# For compatibility with the orchestrator
CardiologyIngestionAgent = SimpleCardiologyIngestionAgent


if __name__ == "__main__":
    # Quick test
    sample_text = """
    65-year-old male with acute chest pain.
    Troponin 2.45 ng/mL. BP 150/95 mmHg.
    ECG shows ST elevation in V2-V4.
    History of hypertension and diabetes on insulin.
    """
    
    agent = SimpleCardiologyIngestionAgent()
    result = agent.process(sample_text)
    
    import json
    print(json.dumps(result, indent=2))