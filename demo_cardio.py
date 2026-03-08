"""
demo_cardio.py
Doctor Copilot - Quick Demo Script
===================================
Demonstrates the complete multi-agent system with a sample cardiology case.
"""

import json
import sys
from pathlib import Path

# Add the project root to path if needed
# sys.path.insert(0, str(Path(__file__).parent))

from cardiology_orchestrator import DoctorCopilotOrchestrator


def run_cardiology_case():
    """Run a complete cardiology case through the system."""
    
    print("\n" + "=" * 80)
    print("DOCTOR COPILOT - CARDIOLOGY CASE DEMONSTRATION")
    print("=" * 80)
    
    # Sample STEMI case
    clinical_note = """
    EMERGENCY DEPARTMENT NOTE
    Date: 2024-01-15
    
    HISTORY OF PRESENT ILLNESS:
    65-year-old male presents with acute onset chest pain that began 45 minutes ago while
    shoveling snow. Pain is described as severe pressure-like sensation, 8/10 intensity,
    radiating to left arm and jaw. Associated with diaphoresis, nausea, and shortness of breath.
    Pain partially relieved with rest but has not completely resolved. Denies prior episodes.
    
    PAST MEDICAL HISTORY:
    - Hypertension (poorly controlled)
    - Type 2 Diabetes Mellitus on insulin
    - Hyperlipidemia
    - Former smoker (quit 2 years ago, 30 pack-year history)
    
    MEDICATIONS:
    - Lisinopril 20 mg daily
    - Metformin 1000 mg twice daily
    - Insulin glargine 40 units nightly
    - Atorvastatin 40 mg daily
    
    PHYSICAL EXAMINATION:
    Vitals: BP 155/92 mmHg, HR 95 bpm, RR 22, SpO2 94% on room air, Temp 98.4°F
    General: Diaphoretic, in moderate distress
    Cardiovascular: Regular rhythm, no murmurs
    Respiratory: Clear bilaterally, increased work of breathing
    
    LABORATORY RESULTS:
    Troponin I: 2.45 ng/mL (reference <0.04 ng/mL) - CRITICAL HIGH
    BNP: 680 pg/mL (reference <100 pg/mL)
    Creatinine: 1.4 mg/dL
    Glucose: 245 mg/dL
    WBC: 12.5 K/uL
    
    ECG FINDINGS:
    Sinus rhythm at 95 bpm
    ST elevation 3-4 mm in leads V2-V4
    ST elevation 2 mm in leads I, aVL
    New left bundle branch block
    Compared to prior ECG from 6 months ago showing normal sinus rhythm
    
    ECHOCARDIOGRAPHY (PORTABLE):
    Left ventricular ejection fraction: 32% (severely reduced)
    Global hypokinesis with regional wall motion abnormalities in anterior wall
    Mild mitral regurgitation
    No pericardial effusion
    
    IMPRESSION:
    Acute ST-elevation myocardial infarction (anterior STEMI)
    Cardiogenic shock risk
    
    PLAN:
    Activate cardiac catheterization lab
    Cardiology consultation - STAT
    """
    
    # Initialize the orchestrator
    orchestrator = DoctorCopilotOrchestrator()
    
    # Process the case
    print("\nProcessing clinical case...")
    report = orchestrator.process_patient(clinical_note)
    
    # Save the detailed JSON report
    json_path = 'D:\GUIDELINES_mod\output\stemi_case_report.json'
    orchestrator.save_report(report, json_path)
    
    # Print formatted report to console
    orchestrator.print_formatted_report(report)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Probable Diagnoses: {len(report['clinical_assessment']['probable_diagnoses'])}")
    print(f"Risk Level: {report['risk_stratification']['overall_risk']}")
    print(f"Risk Score: {report['risk_stratification']['risk_score']}")
    print(f"Red Flags: {len(report['safety_assessment']['red_flags'])}")
    print(f"Critical Alerts: {len(report['safety_assessment']['critical_alerts'])}")
    print(f"Recommendations: {len(report['recommendations'])}")
    print("=" * 80)
    
    return report


def run_non_cardiac_case():
    """Run a non-cardiac case to test filtering."""
    
    print("\n" + "=" * 80)
    print("TESTING NON-CARDIAC CASE (SHOULD FILTER OUT)")
    print("=" * 80)
    
    clinical_note = """
    ORTHOPEDIC CLINIC NOTE
    
    CHIEF COMPLAINT:
    Right knee pain for 3 weeks
    
    HPI:
    42-year-old female with right knee pain following jogging injury.
    Sharp pain with weight bearing, swelling noted.
    No chest pain, no shortness of breath.
    
    EXAM:
    Right knee: effusion present, tenderness medial joint line
    McMurray test positive
    
    X-RAY:
    Mild degenerative changes, no fracture
    
    ASSESSMENT:
    Likely meniscal tear
    
    PLAN:
    MRI knee
    Orthopedic referral
    """
    
    orchestrator = DoctorCopilotOrchestrator()
    report = orchestrator.process_patient(clinical_note)
    
    print(f"\nCardiology Relevant: {report['patient_summary']['cardiology_relevant']}")
    print(f"Relevance Score: {report['patient_summary']['relevance_score']}")
    
    if not report['patient_summary']['cardiology_relevant']:
        print("✓ Correctly identified as non-cardiac case")
    
    return report


def run_custom_case(clinical_text: str, output_file: str = None):
    """
    Run any custom clinical case through the system.
    
    Args:
        clinical_text: Raw clinical note text
        output_file: Optional path to save JSON report
    """
    orchestrator = DoctorCopilotOrchestrator()
    report = orchestrator.process_patient(clinical_text)
    
    if output_file:
        orchestrator.save_report(report, output_file)
    else:
        orchestrator.print_formatted_report(report)
    
    return report


if __name__ == "__main__":
    # Run the main cardiology demonstration
    report = run_cardiology_case()
    
    # Optionally test non-cardiac filtering
    # run_non_cardiac_case()
    
    print("\n✅ Demo complete! Check the generated JSON files for detailed outputs.")