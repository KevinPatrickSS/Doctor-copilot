#!/usr/bin/env python3
"""
Doctor Copilot - Demo Script
Demonstrates system usage without the web UI
"""

import os
import json
from orchestrator import DoctorCopilotOrchestrator

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def demo_patient_1():
    """Demo patient case 1: Acute Coronary Syndrome"""
    return """
EMERGENCY DEPARTMENT NOTE
Date: 01/30/2024

65-year-old male presenting to ED with acute onset chest pain.

HPI:
Patient awoke at 6 AM with crushing substernal chest pain radiating to left arm 
and jaw. Associated with dyspnea and diaphoresis. Immediate onset, lasted 
30 minutes. Patient denies prior episodes.

PMHx:
- Hypertension (treated with lisinopril)
- Type 2 Diabetes (insulin-dependent)
- Hyperlipidemia (on atorvastatin)
- Smoking history: 40 pack-years, quit 5 years ago

Physical Examination:
BP: 165/95 mmHg
HR: 102 bpm
RR: 22
O2 sat: 94% on room air
JVD: 2 cm
Lung auscultation: clear bilaterally
Heart: regular rate and rhythm, S4 gallop noted

LABS:
Troponin I: 0.85 ng/mL (reference <0.04) - HIGH
BNP: 450 pg/mL (reference <100) - ELEVATED
Creatinine: 1.2 mg/dL (baseline ~1.0)
WBC: 13.2 K/uL

IMAGING:
ECG:
- ST elevation in leads V1-V4 (2-3 mm)
- New LBBB pattern
- T wave inversion in aVL

Chest X-ray:
- Mild pulmonary edema
- No pneumothorax

ECHO:
- Ejection fraction: 28% (severely reduced)
- Global hypokinesis with anteroseptal wall motion abnormality
- Mild mitral regurgitation
- Normal pericardium

MEDICATIONS:
- Atorvastatin 80 mg daily
- Lisinopril 20 mg daily
- Aspirin 81 mg daily
- No beta-blockers currently

ALLERGIES: NKDA

ASSESSMENT:
Acute anteroseptal STEMI with cardiogenic shock
"""

def demo_patient_2():
    """Demo patient case 2: Heart Failure Exacerbation"""
    return """
INPATIENT HOSPITAL NOTE
Date: 01/30/2024

72-year-old female admitted with dyspnea and lower extremity edema

HPI:
Patient reports progressive dyspnea over 3 days, orthopnea, PND.
Denies chest pain. Weight gain of 4 lbs in past week.
Known history of heart failure with preserved ejection fraction.

PMHx:
- Heart Failure (HFpEF) - EF 55% on last echo 6 months ago
- Hypertension
- Atrial Fibrillation on rate control
- Chronic Kidney Disease Stage 3 (CKD3)
- Hypothyroidism

Vitals:
BP: 152/88 mmHg
HR: 88 bpm (irregular)
RR: 24
O2 sat: 89% on room air, improves to 94% on 2L NC
Weight: 78.5 kg (up from 74.8 kg baseline)

PE:
Elevated JVP (5 cm)
Lung auscultation: crackles at bilateral bases
Abdomen: hepatomegaly, ascites present
Extremities: 2+ pitting edema bilateral ankles

LABS:
BNP: 580 pg/mL
NT-proBNP: 2400 pg/mL
Creatinine: 1.8 mg/dL (baseline 1.5)
eGFR: 35 mL/min/1.73m2
K+: 5.2 mEq/L (high-normal)
Troponin I: 0.02 ng/mL (normal)

IMAGING:
Chest X-ray:
- Bilateral pleural effusions (moderate)
- Interstitial edema pattern
- Cardiomegaly noted

ECHO (Today):
- EF: 58% (preserved)
- LA enlargement
- Elevated E/e' ratio suggesting impaired relaxation
- Moderate mitral regurgitation
- No structural abnormalities

ECG:
- Atrial fibrillation with RVR (rate 88)
- RBBB pattern
- No acute ST changes

CURRENT MEDS:
- Metoprolol 50 mg BID
- Lisinopril 10 mg daily
- Furosemide 40 mg daily (taking)
- Spironolactone 25 mg daily
- Apixaban 5 mg BID
- Levothyroxine 75 mcg daily
- Simvastatin 20 mg daily

ASSESSMENT:
Acute decompensated heart failure HFpEF with elevated filling pressures
Worsening renal function
"""

def demo_patient_3():
    """Demo patient case 3: Asymptomatic with risk factors"""
    return """
OUTPATIENT CARDIOLOGY NOTE
Date: 01/30/2024

58-year-old male follow-up for cardiac risk stratification

HPI:
Patient presenting for routine cardiology evaluation.
No current symptoms. Reports good functional capacity.
Concerned about family history of premature CAD.

FHx:
Father: MI at age 62
Brother: Hypertension, no CAD

PMHx:
- Hypertension (newly diagnosed 6 months ago)
- Hyperlipidemia (on statin)
- Sedentary lifestyle
- BMI: 28.5 kg/m2 (overweight)
- Never smoker

Social:
Married, works in IT, sedentary job
Alcohol: occasional
Stress: reports moderate work stress

Vitals:
BP: 138/88 mmHg
HR: 72 bpm
Weight: 92 kg, Height: 180 cm

Exam:
General: well-appearing
HEENT: unremarkable
Cardiac: regular rate/rhythm, no murmurs
Lungs: clear
Extremities: no edema

LABS:
Total Cholesterol: 248 mg/dL
LDL: 165 mg/dL
HDL: 38 mg/dL
Triglycerides: 185 mg/dL
Fasting glucose: 112 mg/dL (impaired fasting glucose)
Troponin I: <0.01 ng/mL (normal)
hs-CRP: 2.8 mg/L (elevated)

ECG:
Normal sinus rhythm
No ST changes
Normal QTc
No LVH

Stress Test (6 months ago):
Treadmill exercise test: Negative for inducible ischemia
Achieved 11 METs

ASSESSMENT:
Asymptomatic with multiple cardiovascular risk factors:
- Family history of premature CAD
- Hypertension
- Dyslipidemia (elevated LDL, low HDL, elevated TG)
- Impaired fasting glucose
- Elevated hs-CRP
- Sedentary lifestyle
- Overweight
"""

def main():
    """Run demo."""
    print_section("DOCTOR COPILOT - DEMONSTRATION")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not set")
        print("\nSet it with:")
        print('  export GEMINI_API_KEY="your_api_key_here"')
        print("\nGet a free key from: https://makersuite.google.com/app/apikey")
        return
    
    print("✅ Gemini API key configured")
    
    # Initialize orchestrator
    print("\n📚 Initializing Doctor Copilot orchestrator...")
    try:
        orchestrator = DoctorCopilotOrchestrator(api_key)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Choose demo case
    print_section("AVAILABLE DEMO CASES")
    print("1. Acute Coronary Syndrome (STEMI)")
    print("2. Heart Failure Exacerbation (HFpEF)")
    print("3. Asymptomatic with Risk Factors")
    print("4. Exit")
    
    choice = input("\nSelect case (1-4): ").strip()
    
    demo_cases = {
        '1': ('Acute Coronary Syndrome', demo_patient_1()),
        '2': ('Heart Failure Exacerbation', demo_patient_2()),
        '3': ('Asymptomatic with Risk Factors', demo_patient_3()),
    }
    
    if choice not in demo_cases or choice == '4':
        print("\n👋 Exiting demo")
        return
    
    case_name, patient_data = demo_cases[choice]
    
    print_section(f"ANALYZING: {case_name}")
    print("Processing patient data through all agents...\n")
    
    # Process
    try:
        report = orchestrator.process_patient_data(patient_data)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print_section("CLINICAL DECISION SUPPORT REPORT")
    
    # Disclaimer
    print(f"⚠️  {report.get('disclaimer', '')}\n")
    
    # Report ID
    print(f"Report ID: {report.get('report_id')}")
    print(f"Timestamp: {report.get('timestamp')}\n")
    
    # Patient Summary
    patient = report.get('patient_summary', {})
    print("PATIENT INFORMATION:")
    print(f"  Age: {patient.get('age', 'N/A')}")
    print(f"  Sex: {patient.get('sex', 'N/A')}")
    print(f"  Encounter Type: {patient.get('encounter_type', 'N/A')}")
    print(f"  Cardiology Relevant: {patient.get('cardiology_relevant')}")
    print(f"  Relevance Score: {patient.get('relevance_score')}\n")
    
    # Safety Assessment
    safety = report.get('safety_assessment', {})
    print("SAFETY ASSESSMENT:")
    print(f"  Status: {safety.get('status', 'UNKNOWN')}")
    
    if safety.get('red_flags'):
        print(f"\n  🚩 Red Flags ({len(safety.get('red_flags'))}):")
        for flag in safety.get('red_flags', []):
            print(f"    • {flag}")
    
    if safety.get('warnings'):
        print(f"\n  ⚠️ Warnings ({len(safety.get('warnings'))}):")
        for warn in safety.get('warnings', []):
            print(f"    • {warn}")
    
    if safety.get('missing_evaluations'):
        print(f"\n  📋 Missing Evaluations ({len(safety.get('missing_evaluations'))}):")
        for missing in safety.get('missing_evaluations', []):
            print(f"    • {missing}")
    
    # Clinical Assessment (truncated)
    print("\nCLINICAL ASSESSMENT:")
    assessment = report.get('clinical_assessment', '')
    lines = assessment.split('\n')
    for line in lines[:15]:  # Show first 15 lines
        if line.strip():
            print(f"  {line}")
    if len(lines) > 15:
        print(f"  ... ({len(lines) - 15} more lines)")
    
    # Recommendations
    print("\nRECOMMENDED PHYSICIAN ACTIONS:")
    for action in report.get('physician_actions', []):
        print(f"  {action}")
    
    # Save report
    print_section("SAVING REPORT")
    
    report_id = report.get('report_id', 'unknown')
    json_file = f"report_{report_id}.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Full report saved to: {json_file}\n")
    
    # Show how to export
    print("To export this report:")
    print(f"  1. Use the web UI at http://localhost:3000")
    print(f"  2. Or manually: python -c \"import json; r=json.load(open('{json_file}')); print(json.dumps(r, indent=2))\"")
    
    print_section("DEMO COMPLETE")
    print("✅ Doctor Copilot successfully analyzed patient data!")
    print("\nFor production use:")
    print("  1. Start backend: python app.py")
    print("  2. Start frontend: npm start")
    print("  3. Open: http://localhost:3000")


if __name__ == "__main__":
    main()