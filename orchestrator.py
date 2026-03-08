"""
Doctor Copilot - Risk Assessment Agent Integration
Enhanced orchestrator with cardiac risk assessment
Uses RCRI + ML models + SHAP for comprehensive risk evaluation
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

# Import your agents
from cardiology_nlp_agent import CardiologyIngestionAgent
from RagSystem.rag_system import RAGSystem
from cardiac_risk_assesment.cardiac_risk_agent import CardiacRiskAgent  # Risk Assessment Agent
from dotenv import load_dotenv

load_dotenv()

class DoctorCopilotOrchestrator:
    """
    Enhanced orchestrator that includes Risk Assessment Agent.
    
    Pipeline:
    1. Ingestion & NLP → Extract patient data
    2. Risk Assessment → RCRI + ML scoring + SHAP
    3. RAG Retrieval → Get clinical guidelines
    4. Clinical Reasoning → Gemini insights
    5. Safety Checks → Deterministic rules
    6. Report Generation → Final output
    """
    
    def __init__(self, gemini_api_key: str, chroma_db_path: str = "D:\GUIDELINES_mod\RagSystem\chroma_db", 
                 models_dir: str = "D:\GUIDELINES_mod\cardiac_risk_assesment\models"):
        """Initialize all components including Risk Assessment Agent."""
        print("\n" + "="*80)
        print("🏥 DOCTOR COPILOT - ORCHESTRATOR WITH RISK ASSESSMENT")
        print("="*80)
        
        # Configure Gemini
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        # Initialize agents
        print("\n[1/4] Initializing Ingestion & NLP Agent...")
        try:
            self.ingestion_agent = CardiologyIngestionAgent(use_biobert=True, use_gpu=False)
            print("✅ Ingestion Agent ready")
        except Exception as e:
            print(f"⚠️  Ingestion Agent error: {e}")
            self.ingestion_agent = None
        
        print("\n[2/4] Initializing Risk Assessment Agent...")
        try:
            self.risk_agent = CardiacRiskAgent(models_dir=models_dir)
            print("✅ Risk Assessment Agent ready (RCRI + ML + SHAP)")
        except Exception as e:
            print(f"⚠️  Risk Assessment Agent error: {e}")
            print("   Continuing without risk assessment...")
            self.risk_agent = None
        
        print("\n[3/4] Initializing RAG System...")
        try:
            self.rag_system = RAGSystem(persist_directory=chroma_db_path)
            if self.rag_system.load_existing_vectorstore():
                print("✅ RAG System ready")
            else:
                print("⚠️  No existing RAG vectorstore found")
                self.rag_system = None
        except Exception as e:
            print(f"⚠️  RAG System error: {e}")
            self.rag_system = None
        
        print("\n[4/4] Gemini API ready for clinical reasoning")
        print("✅ Orchestrator initialization complete")
        print("="*80 + "\n")
    
    def process_patient_data(self, raw_patient_text: str) -> Dict[str, Any]:
        """
        Main orchestration pipeline with risk assessment.
        
        Args:
            raw_patient_text: Raw clinical note or patient data
            
        Returns:
            Comprehensive clinical decision support report with risk assessment
        """
        print("\n" + "="*80)
        print("🔄 PROCESSING PATIENT DATA WITH RISK ASSESSMENT")
        print("="*80)
        
        # Step 1: Ingestion & NLP
        print("\n[Step 1] Ingesting and parsing patient data...")
        ingested_data = self._step_ingestion(raw_patient_text)
        
        # Step 2: Risk Assessment (NEW)
        print("\n[Step 2] Performing cardiac risk assessment...")
        risk_assessment = self._step_risk_assessment(ingested_data)
        
        # Step 3: RAG Retrieval
        print("\n[Step 3] Retrieving relevant clinical guidelines...")
        guideline_context = self._step_rag_retrieval(ingested_data, risk_assessment)
        
        # Step 4: Clinical Reasoning
        print("\n[Step 4] Generating clinical insights...")
        clinical_insights = self._step_clinical_reasoning(ingested_data, risk_assessment, guideline_context)
        
        # Step 5: Safety & Contraindication Checks
        print("\n[Step 5] Running safety checks...")
        safety_checks = self._step_safety_checks(ingested_data, clinical_insights, risk_assessment)
        
        # Step 6: Generate Final Report
        print("\n[Step 6] Compiling final report...")
        final_report = self._step_report_generation(
            ingested_data, 
            risk_assessment,
            clinical_insights, 
            safety_checks,
            guideline_context
        )
        
        return final_report
    
    def _step_ingestion(self, raw_text: str) -> Dict[str, Any]:
        """Step 1: Parse and normalize patient data."""
        if not self.ingestion_agent:
            return {"raw_text": raw_text, "error": "Ingestion agent not available"}
        
        try:
            result = self.ingestion_agent.process(raw_text)
            print("   ✅ Extracted: demographics, symptoms, investigations, medications")
            return result
        except Exception as e:
            print(f"   ⚠️  Ingestion error: {e}")
            return {"raw_text": raw_text, "error": str(e)}
    
    def _step_risk_assessment(self, ingested_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Perform comprehensive cardiac risk assessment.
        Includes RCRI scoring, ML models, and SHAP explanations.
        """
        if not self.risk_agent:
            print("   ⚠️  Risk Assessment Agent not available")
            return {"status": "unavailable", "reason": "Risk agent not initialized"}
        
        try:
            # Convert ingested data to risk agent format
            patient_data = self._convert_to_risk_format(ingested_data)
            
            # Perform assessment
            risk_result = self.risk_agent.assess_patient(patient_data)
            
            print("   ✅ Risk Assessment Complete:")
            print(f"      • RCRI Score: {risk_result['rcri'].get('rcri_score', 'N/A')}/6")
            print(f"      • RCRI Risk Class: {risk_result['rcri'].get('rcri_risk_class', 'N/A')}")
            if risk_result.get('ml_prediction'):
                print(f"      • ML Risk Probability: {risk_result['ml_prediction'].get('ml_risk_probability', 'N/A'):.1%}")
            print(f"      • Overall Risk: {risk_result['combined_assessment'].get('overall_risk_class', 'N/A').upper()}")
            
            return risk_result
        
        except Exception as e:
            print(f"   ⚠️  Risk Assessment error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _convert_to_risk_format(self, ingested_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ingested data to risk agent input format."""
        
        patient = ingested_data.get('patient', {})
        investigations = ingested_data.get('investigations', [])
        risk_factors = ingested_data.get('risk_factors', [])
        
        risk_input = {
            'patient_id': 'PATIENT-' + datetime.now().strftime('%Y%m%d%H%M%S'),
            
            # Demographics
            'age': patient.get('age'),
            'sex': patient.get('sex'),
            
            # RCRI fields
            'surgery_risk_level': 'high',  # Default, can be inferred
            'history_ihd': self._check_risk_factor(risk_factors, 'cardiac'),
            'history_hf': self._check_risk_factor(risk_factors, 'heart failure'),
            'history_cva': self._check_risk_factor(risk_factors, 'stroke'),
            'diabetes_insulin': self._check_risk_factor(risk_factors, 'diabetes'),
            
            # Extract lab values
            'creatinine_mg_dl': self._extract_lab_value(investigations, 'creatinine'),
        }
        
        # Extract vital signs for ML features
        for inv in investigations:
            if isinstance(inv, dict):
                test_type = inv.get('test_type', '').lower()
                
                if test_type == 'blood_pressure':
                    risk_input['trestbps'] = inv.get('systolic', 120)
                
                if test_type == 'troponin':
                    # High troponin indicates acute event
                    risk_input['exang_binary'] = 1 if inv.get('value', 0) > 0.04 else 0
        
        # Set defaults for missing ML features
        risk_input['trestbps'] = risk_input.get('trestbps', 140)
        risk_input['chol'] = self._extract_lab_value(investigations, 'cholesterol') or 200
        risk_input['fbs_binary'] = 1 if self._check_risk_factor(risk_factors, 'diabetes') else 0
        risk_input['thalch'] = 80  # Default heart rate
        risk_input['oldpeak'] = 0  # Default ST depression
        risk_input['exang_binary'] = risk_input.get('exang_binary', 0)
        risk_input['lv_hypertrophy'] = 0  # Would need echo data
        
        return risk_input
    
    def _check_risk_factor(self, risk_factors: List[Dict], keyword: str) -> bool:
        """Check if specific risk factor is present."""
        rf_text = ' '.join([str(rf.get('factor', '')).lower() for rf in risk_factors])
        return keyword.lower() in rf_text
    
    def _extract_lab_value(self, investigations: List[Dict], test_type: str) -> Optional[float]:
        """Extract numeric value from investigation results."""
        for inv in investigations:
            if isinstance(inv, dict):
                if inv.get('test_type', '').lower() == test_type.lower():
                    return inv.get('value')
        return None
    
    def _step_rag_retrieval(self, ingested_data: Dict[str, Any], 
                           risk_assessment: Dict[str, Any]) -> str:
        """Step 3: Retrieve relevant guidelines considering risk level."""
        if not self.rag_system:
            return "RAG system not available"
        
        try:
            # Build query incorporating risk assessment
            symptoms = ingested_data.get('symptoms', [])
            risk_factors = ingested_data.get('risk_factors', [])
            
            # Add risk level to query
            risk_level = "high risk" if risk_assessment.get('combined_assessment', {}).get('overall_risk_class') == 'high' else "risk assessment"
            
            query_terms = [risk_level]
            for sym in symptoms[:2]:
                if isinstance(sym, dict):
                    query_terms.append(sym.get('symptom', ''))
            for rf in risk_factors[:2]:
                if isinstance(rf, dict):
                    query_terms.append(rf.get('factor', ''))
            
            query = "clinical guidelines " + " ".join(filter(None, query_terms))
            
            context = self.rag_system.query_with_context(query, k=5)
            print(f"   ✅ Retrieved clinical context based on risk profile")
            return context
        
        except Exception as e:
            print(f"   ⚠️  RAG error: {e}")
            return "No guidelines retrieved"
    
    def _step_clinical_reasoning(self, ingested_data: Dict[str, Any], 
                                risk_assessment: Dict[str, Any],
                                guideline_context: str) -> Dict[str, Any]:
        """Step 4: Use Gemini for clinical reasoning informed by risk assessment."""
        
        patient_summary = self._format_patient_summary(ingested_data)
        risk_summary = self._format_risk_summary(risk_assessment)
        
        prompt = f"""
You are an experienced cardiologist providing PRELIMINARY clinical decision support.

IMPORTANT DISCLAIMERS:
- This is NOT a final diagnosis or treatment recommendation
- Always recommend human physician review
- Flag any concerning findings for immediate attention
- Explain reasoning with probabilities, not certainties

PATIENT DATA:
{patient_summary}

RISK ASSESSMENT:
{risk_summary}

RELEVANT CLINICAL GUIDELINES:
{guideline_context}

Based on the patient data and RISK ASSESSMENT above, please provide:

1. PROBABLE DIAGNOSES (with confidence levels, NOT definitive)
   - List 2-3 most likely conditions
   - Explain reasoning for each
   - State uncertainty clearly
   - Consider the risk assessment findings

2. RISK-STRATIFIED RECOMMENDATIONS
   - Based on the identified risk level ({risk_assessment.get('combined_assessment', {}).get('overall_risk_class', 'unknown')}):
     * For HIGH risk: What additional evaluations/interventions?
     * For INTERMEDIATE risk: What consultations needed?
     * For LOW risk: What monitoring?

3. PERIOPERATIVE CONSIDERATIONS (if applicable)
   - Impact of risk assessment on surgical/procedural planning
   - Guideline-based pre-operative optimization

4. RECOMMENDED INVESTIGATIONS
   - Any missing critical tests
   - Priority ordering based on risk level
   - Expected turnaround

5. IMMEDIATE CONCERNS/RED FLAGS
   - Any life-threatening conditions to rule out
   - Contraindications based on risk profile
   - Time-sensitive interventions needed

6. NEXT STEPS
   - Recommended specialist consultation (cardiology? surgery?)
   - Timeline for decision-making considering risk level
   - Monitoring plan

CRITICAL: Use phrases like "may suggest", "could indicate", "warrants investigation" - NEVER "is" or "definitively"
INCORPORATE RISK DATA: Ensure recommendations align with identified cardiac risk level
"""
        
        try:
            response = self.model.generate_content(prompt)
            insights = {
                "reasoning": response.text,
                "risk_aware": True,
                "timestamp": datetime.now().isoformat(),
                "model": "gemini-2.0-flash"
            }
            print("   ✅ Clinical reasoning complete (risk-informed)")
            return insights
        except Exception as e:
            print(f"   ⚠️  Gemini error: {e}")
            return {"error": str(e), "reasoning": "Could not generate reasoning"}
    
    def _step_safety_checks(self, ingested_data: Dict[str, Any], 
                           clinical_insights: Dict[str, Any],
                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Enhanced safety checks incorporating risk assessment."""
        
        red_flags = []
        missing_evals = []
        warnings = []
        
        # Risk-based red flags
        risk_level = risk_assessment.get('combined_assessment', {}).get('overall_risk_class', 'unknown')
        if risk_level == 'high':
            red_flags.append("HIGH PERIOPERATIVE RISK IDENTIFIED - Cardiology consultation strongly recommended")
            rcri_score = risk_assessment.get('rcri', {}).get('rcri_score', 0)
            if rcri_score >= 3:
                red_flags.append(f"RCRI score: {rcri_score}/6 (≥3 points indicates increased risk)")
        
        # Check for critical lab values
        investigations = ingested_data.get('investigations', [])
        for inv in investigations:
            if isinstance(inv, dict):
                test_type = inv.get('test_type', '').lower()
                value = inv.get('value')
                
                # Troponin elevation
                if test_type == 'troponin' and value and value > 0.04:
                    red_flags.append(f"Elevated troponin ({value} ng/mL) - acute coronary syndrome concern")
                
                # Critical BP
                if test_type == 'blood_pressure':
                    sys = inv.get('systolic', 0)
                    dias = inv.get('diastolic', 0)
                    if sys > 180 or dias > 120:
                        red_flags.append(f"Severe hypertension ({sys}/{dias} mmHg) - contraindication to elective procedures")
                
                # Low ejection fraction
                if test_type == 'echo':
                    ef = inv.get('ejection_fraction')
                    if ef and ef < 0.35:
                        warnings.append(f"Severely reduced EF ({ef*100:.0f}%) - very high perioperative risk")
                        red_flags.append("EF <35% - High risk for perioperative complications")
                    elif ef and ef < 0.40:
                        warnings.append(f"Reduced EF ({ef*100:.0f}%) - elevated perioperative risk")
        
        # Missing critical investigations based on risk
        inv_types = [i.get('test_type', '').lower() for i in investigations if isinstance(i, dict)]
        
        if risk_level in ['intermediate', 'high']:
            if 'echo' not in inv_types:
                missing_evals.append("Echocardiography (critical for risk stratification)")
            if 'troponin' not in inv_types:
                missing_evals.append("Troponin level (rule out acute coronary syndrome)")
            if 'blood_pressure' not in inv_types:
                missing_evals.append("Blood pressure (baseline for risk assessment)")
        
        # Risk factor assessment
        risk_factors = ingested_data.get('risk_factors', [])
        rf_text = str(risk_factors).lower()
        
        if 'hypertension' in rf_text and 'diabetes' in rf_text:
            warnings.append("Multiple major cardiovascular risk factors - high event probability")
        
        return {
            "red_flags": red_flags,
            "missing_evaluations": missing_evals,
            "warnings": warnings,
            "safety_status": "CLEAR" if not red_flags else "⚠️ REVIEW NEEDED",
            "risk_level": risk_level
        }
    
    def _step_report_generation(self, ingested_data: Dict[str, Any], 
                               risk_assessment: Dict[str, Any],
                               clinical_insights: Dict[str, Any],
                               safety_checks: Dict[str, Any], 
                               guideline_context: str) -> Dict[str, Any]:
        """Step 6: Compile final structured report with risk assessment."""
        
        report = {
            "report_id": f"DCR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "PRELIMINARY CLINICAL DECISION SUPPORT - NOT A FINAL DIAGNOSIS. Requires physician review and confirmation. Risk assessment is probabilistic and should be interpreted by qualified clinicians.",
            
            # Patient Information
            "patient_summary": {
                "age": ingested_data.get('patient', {}).get('age'),
                "sex": ingested_data.get('patient', {}).get('sex'),
                "encounter_type": ingested_data.get('patient', {}).get('encounter_type'),
                "cardiology_relevant": ingested_data.get('cardiology_relevant', False),
                "relevance_score": ingested_data.get('cardiology_relevance_score', 0)
            },
            
            # Risk Assessment (NEW)
            "risk_assessment": {
                "status": risk_assessment.get('status', 'completed'),
                "rcri_score": risk_assessment.get('rcri', {}).get('rcri_score'),
                "rcri_risk_class": risk_assessment.get('rcri', {}).get('rcri_risk_class'),
                "rcri_event_rate": risk_assessment.get('rcri', {}).get('rcri_event_rate'),
                "ml_risk_probability": risk_assessment.get('ml_prediction', {}).get('ml_risk_probability'),
                "ml_risk_class": risk_assessment.get('ml_prediction', {}).get('ml_risk_class'),
                "overall_risk_level": risk_assessment.get('combined_assessment', {}).get('overall_risk_class'),
                "confidence": risk_assessment.get('combined_assessment', {}).get('confidence'),
                "recommendation": risk_assessment.get('combined_assessment', {}).get('recommendation'),
                "rcri_factors": risk_assessment.get('rcri', {}).get('factors_present', []),
                "shap_explanation": risk_assessment.get('shap_explanation', {}).get('explanation_text') if risk_assessment.get('shap_explanation') else None
            },
            
            # Extracted Clinical Data
            "clinical_data": {
                "symptoms": ingested_data.get('symptoms', []),
                "risk_factors": ingested_data.get('risk_factors', []),
                "investigations": ingested_data.get('investigations', []),
                "medications": ingested_data.get('medications', [])
            },
            
            # Clinical Reasoning
            "clinical_assessment": clinical_insights.get('reasoning', ''),
            
            # Safety Layer
            "safety_assessment": {
                "red_flags": safety_checks.get('red_flags', []),
                "missing_evaluations": safety_checks.get('missing_evaluations', []),
                "warnings": safety_checks.get('warnings', []),
                "status": safety_checks.get('safety_status', 'UNKNOWN'),
                "risk_informed": True
            },
            
            # Risk-Based Recommendations
            "physician_actions": self._generate_risk_based_actions(risk_assessment, safety_checks),
            
            "metadata": {
                "system_version": "Doctor Copilot v1.1 (with Risk Assessment)",
                "components_used": ["BioBERT NER", "RCRI Scoring", "ML Models", "SHAP Explainability", "RAG Guidelines", "Gemini-2.0-Flash"],
                "quality_notes": "This system provides probabilistic, explainable support informed by cardiac risk assessment - not definitive conclusions"
            }
        }
        
        return report
    
    def _format_patient_summary(self, ingested_data: Dict[str, Any]) -> str:
        """Format ingested data into readable summary."""
        summary = []
        
        patient = ingested_data.get('patient', {})
        if patient:
            summary.append(f"Demographics: {patient.get('age', 'Unknown')} yo {patient.get('sex', 'Unknown')}")
        
        symptoms = ingested_data.get('symptoms', [])
        if symptoms:
            summary.append(f"Symptoms: {', '.join([s.get('symptom', '') for s in symptoms if isinstance(s, dict)])}")
        
        risk_factors = ingested_data.get('risk_factors', [])
        if risk_factors:
            summary.append(f"Risk Factors: {', '.join([rf.get('factor', '') for rf in risk_factors if isinstance(rf, dict)])}")
        
        investigations = ingested_data.get('investigations', [])
        if investigations:
            inv_summary = []
            for inv in investigations:
                if isinstance(inv, dict):
                    test = inv.get('test_type', '')
                    if inv.get('value'):
                        inv_summary.append(f"{test}: {inv.get('value')} {inv.get('unit', '')}")
                    elif inv.get('findings'):
                        inv_summary.append(f"{test}: {', '.join(inv.get('findings', []))}")
            if inv_summary:
                summary.append(f"Investigations: {'; '.join(inv_summary)}")
        
        medications = ingested_data.get('medications', [])
        if medications:
            meds_str = ', '.join([m.get('drug', '') for m in medications if isinstance(m, dict)])
            if meds_str:
                summary.append(f"Medications: {meds_str}")
        
        return '\n'.join(summary)
    
    def _format_risk_summary(self, risk_assessment: Dict[str, Any]) -> str:
        """Format risk assessment into readable summary."""
        summary = []
        
        if risk_assessment.get('status') == 'unavailable':
            return "Risk assessment data not available"
        
        if risk_assessment.get('status') == 'error':
            return f"Risk assessment error: {risk_assessment.get('error')}"
        
        rcri = risk_assessment.get('rcri', {})
        summary.append(f"RCRI Score: {rcri.get('rcri_score', 'N/A')}/6 ({rcri.get('rcri_risk_class', 'Unknown')})")
        summary.append(f"Estimated Event Rate: {rcri.get('rcri_event_rate', 'N/A')}%")
        
        ml = risk_assessment.get('ml_prediction', {})
        if ml and ml.get('ml_risk_probability'):
            summary.append(f"ML Risk Probability: {ml.get('ml_risk_probability'):.1%} ({ml.get('ml_risk_class', 'Unknown')})")
        
        combined = risk_assessment.get('combined_assessment', {})
        summary.append(f"Overall Risk Level: {combined.get('overall_risk_class', 'Unknown').upper()}")
        summary.append(f"Confidence: {combined.get('confidence', 'Unknown')}")
        summary.append(f"Recommendation: {combined.get('recommendation', 'N/A')}")
        
        if rcri.get('factors_present'):
            summary.append(f"\nRCRI Risk Factors Present:")
            for factor in rcri.get('factors_present', []):
                summary.append(f"  • {factor.get('factor', 'Unknown')}")
        
        shap = risk_assessment.get('shap_explanation', {})
        if shap and shap.get('explanation_text'):
            summary.append(f"\nKey Risk Drivers (SHAP):")
            summary.append(shap.get('explanation_text', ''))
        
        return '\n'.join(summary)
    
    def _generate_risk_based_actions(self, risk_assessment: Dict[str, Any], 
                                    safety_checks: Dict[str, Any]) -> List[str]:
        """Generate physician actions tailored to risk level."""
        
        actions = []
        risk_level = risk_assessment.get('combined_assessment', {}).get('overall_risk_class', 'unknown')
        
        # Risk-based action hierarchy
        if risk_level == 'high':
            actions.append("1. ⚠️ HIGH RISK IDENTIFIED - Urgent cardiology consultation recommended")
            actions.append("2. Perform comprehensive preoperative cardiac evaluation if surgery planned")
            actions.append("3. Consider non-invasive stress testing or coronary angiography")
            actions.append("4. Optimize cardiac medications before elective procedures")
            actions.append("5. Plan for intensive perioperative monitoring")
        
        elif risk_level == 'intermediate':
            actions.append("1. Request cardiology consultation for risk stratification")
            actions.append("2. Consider non-invasive testing (stress test, CTA)")
            actions.append("3. Optimize medical management (blood pressure, heart rate control)")
            actions.append("4. Plan moderate level of perioperative monitoring")
            actions.append("5. Document risk-benefit discussion with patient")
        
        else:  # low risk
            actions.append("1. Standard perioperative evaluation and monitoring")
            actions.append("2. Continue current cardiac medications")
            actions.append("3. Routine preoperative testing per guidelines")
            actions.append("4. Proceed with planned procedures/surgery as scheduled")
            actions.append("5. Standard postoperative care")
        
        # Add safety-specific actions
        if safety_checks.get('red_flags'):
            actions.append(f"\n6. 🚩 CRITICAL: Address {len(safety_checks.get('red_flags', []))} red flag(s) before proceeding")
        
        if safety_checks.get('missing_evaluations'):
            actions.append(f"\n7. 📋 Obtain {len(safety_checks.get('missing_evaluations', []))} missing evaluation(s)")
        
        actions.append("\n8. Review all risk assessment details and SHAP explanations")
        actions.append("9. Make final diagnosis and treatment decisions based on clinical judgment")
        actions.append("10. Document risk assessment results and decision-making rationale")
        
        return actions


def main():
    """Demo usage with Risk Assessment."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    orchestrator = DoctorCopilotOrchestrator(api_key)
    
    sample_text = """
    EMERGENCY DEPARTMENT NOTE
    
    65-year-old male presenting to ED with acute chest pain.
    Pain described as pressure-like, radiating to left arm, started 30 minutes ago.
    Patient has history of hypertension and diabetes, on insulin.
    
    VITAL SIGNS:
    BP: 165/95 mmHg
    HR: 102
    
    LABS:
    Troponin: 0.85 ng/mL (reference <0.04)
    BNP: 450 pg/mL
    Creatinine: 1.2 mg/dL
    
    ECG:
    ST elevation in leads V1-V4, new LBBB
    
    ECHO:
    Ejection fraction 28%, global hypokinesis
    
    MEDICATIONS:
    Atorvastatin 80 mg daily
    Lisinopril 20 mg daily
    """
    
    report = orchestrator.process_patient_data(sample_text)
    
    print("\n" + "="*80)
    print("FINAL REPORT WITH RISK ASSESSMENT")
    print("="*80)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()