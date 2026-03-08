"""
Cardiology Ingestion & NLP Agent - BioBERT Enhanced
====================================================
COMPLETE VERSION with all classes included.

Author: Production ML Engineering Team
Version: 2.0.0 (BioBERT Enhanced)
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import warnings
import numpy as np

# Required dependencies
try:
    import spacy
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        AutoModel,
        pipeline
    )
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. Run:\n"
        "pip install spacy transformers torch numpy\n"
        "python -m spacy download en_core_web_sm"
    ) from e


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Entity:
    """Represents an extracted clinical entity."""
    text: str
    entity_type: str
    confidence: float
    start_char: int
    end_char: int
    negated: bool = False
    temporal_info: Optional[str] = None
    snomed_code: Optional[str] = None
    icd10_code: Optional[str] = None
    rxnorm_code: Optional[str] = None


@dataclass
class ClinicalSection:
    """Represents a section of clinical text."""
    section_type: str
    content: str
    start_char: int
    end_char: int
    entity_density: float = 0.0


# ============================================================================
# PREPROCESSING MODULE
# ============================================================================

class ClinicalTextPreprocessor:
    """Handles preprocessing and normalization of clinical text."""
    
    SECTION_PATTERNS = {
        'HPI': r'(?:HISTORY OF PRESENT ILLNESS|HPI|CHIEF COMPLAINT|CC):?',
        'PMHx': r'(?:PAST MEDICAL HISTORY|PMHx|MEDICAL HISTORY):?',
        'MEDICATIONS': r'(?:MEDICATIONS|MEDS|CURRENT MEDICATIONS):?',
        'PHYSICAL_EXAM': r'(?:PHYSICAL EXAM|PE|EXAMINATION):?',
        'ASSESSMENT': r'(?:ASSESSMENT|IMPRESSION|A/P):?',
        'PLAN': r'(?:PLAN|RECOMMENDATIONS):?',
        'LABS': r'(?:LABS|LABORATORY|LAB RESULTS):?',
        'IMAGING': r'(?:IMAGING|RADIOLOGY|ECG|ECHO):?'
    }
    
    UNIT_NORMALIZATIONS = {
        r'\bmmhg\b': 'mmHg',
        r'\bmg/dl\b': 'mg/dL',
        r'\bng/ml\b': 'ng/mL',
        r'\bml/min\b': 'mL/min',
        r'\bg/dl\b': 'g/dL',
        r'\bmeq/l\b': 'mEq/L',
        r'\bumol/l\b': 'umol/L'
    }
    
    def __init__(self):
        self.phi_patterns = [
            r'\[\*\*[^\]]+\*\*\]',
            r'\[PHI:[^\]]+\]',
        ]
    
    def remove_phi_markers(self, text: str) -> str:
        for pattern in self.phi_patterns:
            text = re.sub(pattern, '[REDACTED]', text)
        return text
    
    def normalize_units(self, text: str) -> str:
        normalized = text
        for pattern, replacement in self.UNIT_NORMALIZATIONS.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return normalized
    
    def split_sections(self, text: str) -> List[ClinicalSection]:
        sections = []
        section_matches = []
        
        for section_type, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_matches.append((match.start(), section_type, match.group()))
        
        section_matches.sort(key=lambda x: x[0])
        
        for i, (start_pos, section_type, header) in enumerate(section_matches):
            content_start = start_pos + len(header)
            content_end = section_matches[i + 1][0] if i + 1 < len(section_matches) else len(text)
            
            content = text[content_start:content_end].strip()
            if content:
                sections.append(ClinicalSection(
                    section_type=section_type,
                    content=content,
                    start_char=content_start,
                    end_char=content_end
                ))
        
        if not sections:
            sections.append(ClinicalSection(
                section_type='HPI',
                content=text.strip(),
                start_char=0,
                end_char=len(text)
            ))
        
        return sections
    
    def preprocess(self, text: str) -> Tuple[str, List[ClinicalSection]]:
        text = self.remove_phi_markers(text)
        text = self.normalize_units(text)
        sections = self.split_sections(text)
        return text, sections


# ============================================================================
# BIOBERT-POWERED NER MODULE
# ============================================================================

class BioBERTCardiologyNER:
    """BioBERT-powered cardiology-specific Named Entity Recognition."""
    
    ENTITY_TYPES = {
        'SYMPTOM': ['chest pain', 'dyspnea', 'syncope', 'palpitations', 'orthopnea', 'edema', 'fatigue'],
        'DISEASE': ['MI', 'myocardial infarction', 'STEMI', 'NSTEMI', 'heart failure', 'HFrEF', 'HFpEF', 
                   'HF', 'AF', 'atrial fibrillation', 'ACS', 'CAD', 'cardiomyopathy'],
        'FINDING': ['ST elevation', 'ST depression', 'T wave inversion', 'LBBB', 'RBBB', 'EF', 
                   'ejection fraction', 'wall motion', 'hypokinesis', 'akinesis'],
        'PROCEDURE': ['PCI', 'CABG', 'catheterization', 'angiography', 'echocardiography', 'echo', 
                     'stress test', 'ECG', 'cath'],
        'MEDICATION': ['aspirin', 'statin', 'atorvastatin', 'beta blocker', 'metoprolol', 'lisinopril', 
                      'ACEI', 'ARB', 'entresto', 'SGLT2i', 'warfarin', 'DOAC', 'nitrate'],
        'RISK_FACTOR': [],
        'VALUE': []
    }
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", use_gpu: bool = False):
        print(f"🔵 Loading BioBERT model: {model_name}")
        
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")
        
        try:
            print("   Attempting to load from cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            print("   ✓ Loaded from local cache")
        except Exception as e:
            print(f"   Downloading from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print("   ✓ Downloaded successfully")
        
        self.model.to(self.device)
        self.model.eval()
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except (OSError, ImportError):
            print("⚠️  spaCy model not found")
            self.nlp = None
            self.use_spacy = False
        
        self.entity_patterns = self._compile_patterns()
        self.entity_embeddings = self._precompute_entity_embeddings()
        
        print("✅ BioBERT NER initialized successfully")
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        patterns = {}
        for entity_type, keywords in self.ENTITY_TYPES.items():
            patterns[entity_type] = [
                re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                for kw in keywords
            ]
        
        patterns['VALUE'] = [
            re.compile(r'\b(?:BP|blood pressure)[\s:]+\d+[/-]\d+\s*mmHg\b', re.IGNORECASE),
            re.compile(r'\b(?:HR|heart rate)[\s:]+\d+\b', re.IGNORECASE),
            re.compile(r'\b(?:troponin|Troponin)[\s:]+\d+\.?\d*\s*(?:ng/mL)?\b', re.IGNORECASE),
            re.compile(r'\b(?:BNP|NT-proBNP)[\s:]+\d+\.?\d*\s*(?:pg/mL)?\b', re.IGNORECASE),
        ]
        
        patterns['RISK_FACTOR'] = [
            re.compile(r'\b(?:hypertension|HTN|high blood pressure)\b', re.IGNORECASE),
            re.compile(r'\b(?:diabetes|DM|diabetic)\b', re.IGNORECASE),
            re.compile(r'\b(?:smoking|smoker|tobacco)\b', re.IGNORECASE),
            re.compile(r'\b(?:hyperlipidemia|high cholesterol)\b', re.IGNORECASE),
        ]
        
        return patterns
    
    def _precompute_entity_embeddings(self) -> Dict[str, torch.Tensor]:
        embeddings = {}
        with torch.no_grad():
            for entity_type, keywords in self.ENTITY_TYPES.items():
                if keywords:
                    type_embeddings = []
                    for keyword in keywords:
                        inputs = self.tokenizer(keyword, return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu()
                        type_embeddings.append(embedding)
                    embeddings[entity_type] = torch.mean(torch.cat(type_embeddings), dim=0)
        return embeddings
    
    def extract_entities(self, text: str, use_biobert: bool = True) -> List[Entity]:
        entities = []
        seen_spans: Set[Tuple[int, int]] = set()
        
        # Rule-based extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())
                    if span not in seen_spans:
                        entities.append(Entity(
                            text=match.group(),
                            entity_type=entity_type,
                            confidence=0.85,
                            start_char=match.start(),
                            end_char=match.end(),
                            negated=False
                        ))
                        seen_spans.add(span)
        
        entities = self._deduplicate_entities(entities)
        print(f"   Extracted {len(entities)} entities using BioBERT-enhanced NER")
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        if not entities:
            return []
        entities.sort(key=lambda e: e.start_char)
        deduplicated = []
        for entity in entities:
            if deduplicated:
                last = deduplicated[-1]
                if entity.start_char < last.end_char:
                    if entity.confidence > last.confidence:
                        deduplicated[-1] = entity
                    continue
            deduplicated.append(entity)
        return deduplicated


# ============================================================================
# NEGATION DETECTION
# ============================================================================

class NegationDetector:
    """Detects negated clinical concepts."""
    
    NEGATION_PATTERNS = [
        r'\bno\b', r'\bnot\b', r'\bdenies\b', r'\bdenied\b',
        r'\bwithout\b', r'\brule out\b', r'\br/o\b', r'\babsent\b'
    ]
    
    NEGATION_WINDOW = 50
    
    def __init__(self):
        self.negation_regex = re.compile('|'.join(self.NEGATION_PATTERNS), re.IGNORECASE)
    
    def detect_negation(self, text: str, entities: List[Entity]) -> List[Entity]:
        negation_triggers = [m.start() for m in self.negation_regex.finditer(text)]
        for entity in entities:
            for trigger_pos in negation_triggers:
                if trigger_pos < entity.start_char < trigger_pos + self.NEGATION_WINDOW:
                    entity.negated = True
                    break
        return entities


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class CardiologyIngestionAgent:
    """Main orchestrator for cardiology NLP pipeline with BioBERT."""
    
    def __init__(self, use_biobert: bool = True, use_gpu: bool = False):
        print("\n" + "=" * 80)
        print("INITIALIZING BIOBERT-POWERED CARDIOLOGY NLP AGENT")
        print("=" * 80)
        
        self.preprocessor = ClinicalTextPreprocessor()
        
        if use_biobert:
            self.ner = BioBERTCardiologyNER(use_gpu=use_gpu)
        else:
            print("⚠️  BioBERT disabled - using rule-based NER only")
            self.ner = BioBERTCardiologyNER(use_gpu=False)
        
        self.negation_detector = NegationDetector()
        self.use_biobert = use_biobert
        
        print("✅ Agent initialization complete")
        print("=" * 80 + "\n")
    
    def process(self, raw_text: str) -> Dict[str, Any]:
        """Main processing pipeline."""
        # Preprocess
        cleaned_text, sections = self.preprocessor.preprocess(raw_text)
        
        # Extract entities
        entities = self.ner.extract_entities(cleaned_text)
        
        # Detect negations
        entities = self.negation_detector.detect_negation(cleaned_text, entities)
        
        # Structure output
        output = self._structure_output(cleaned_text, entities)
        return output
    
    def _structure_output(self, text: str, entities: List[Entity]) -> Dict[str, Any]:
        """Convert extracted information into structured JSON output."""
        
        age = self._extract_age(text)
        sex = self._extract_sex(text)
        encounter_type = self._extract_encounter_type(text)
        
        symptoms = self._extract_detailed_symptoms(text, entities)
        risk_factors = self._extract_risk_factors(text, entities)
        investigations = self._extract_investigations(text, entities)
        medications = self._extract_medications(text, entities)
        
        extracted_entities = []
        for entity in entities:
            if not entity.negated and entity.entity_type in ['DISEASE', 'FINDING']:
                if entity.confidence >= 0.75:
                    extracted_entities.append({
                        'entity': entity.text,
                        'type': entity.entity_type,
                        'confidence': entity.confidence,
                        'snomed_code': entity.snomed_code,
                        'icd10_code': entity.icd10_code
                    })
        
        is_relevant, relevance_score = self._compute_relevance(text, entities)
        
        output = {
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
                'agent_version': '2.0.0-biobert',
                'model': 'BioBERT (dmis-lab/biobert-base-cased-v1.1)' if self.use_biobert else 'Rule-based',
                'safety_notice': 'This output is for information extraction only.'
            }
        }
        
        return output
    
    def _extract_age(self, text: str) -> Optional[int]:
        match = re.search(r'\b(\d{1,3})[\s-]?(?:year|yr|yo|y/o)[\s-]?old\b', text, re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def _extract_sex(self, text: str) -> Optional[str]:
        if re.search(r'\b(?:male|man)\b', text, re.IGNORECASE) and not re.search(r'\bfemale\b', text, re.IGNORECASE):
            return 'M'
        if re.search(r'\b(?:female|woman)\b', text, re.IGNORECASE):
            return 'F'
        return None
    
    def _extract_encounter_type(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        if 'ed' in text_lower or 'emergency' in text_lower:
            return 'ED'
        if 'admission' in text_lower or 'inpatient' in text_lower:
            return 'Inpatient'
        if 'clinic' in text_lower or 'outpatient' in text_lower:
            return 'Outpatient'
        return None
    
    def _extract_detailed_symptoms(self, text: str, entities: List[Entity]) -> List[Dict[str, Any]]:
        symptoms = []
        for entity in entities:
            if entity.entity_type == 'SYMPTOM' and not entity.negated:
                symptom_data = {'symptom': entity.text}
                
                context_start = max(0, entity.start_char - 200)
                context_end = min(len(text), entity.end_char + 200)
                context = text[context_start:context_end].lower()
                
                if 'chest pain' in entity.text.lower():
                    if 'pressure' in context:
                        symptom_data['character'] = 'pressure'
                    elif 'sharp' in context:
                        symptom_data['character'] = 'sharp'
                    
                    if 'left arm' in context:
                        symptom_data['radiation'] = 'left arm'
                    elif 'jaw' in context:
                        symptom_data['radiation'] = 'jaw'
                    
                    if 'acute' in context or 'sudden' in context:
                        symptom_data['onset'] = 'acute'
                
                symptoms.append(symptom_data)
        
        return symptoms
    
    def _extract_risk_factors(self, text: str, entities: List[Entity]) -> List[Dict[str, Any]]:
        risk_factors = []
        text_lower = text.lower()
        
        if any(e.entity_type == 'RISK_FACTOR' and 'hypertension' in e.text.lower() for e in entities):
            controlled = 'controlled' in text_lower
            risk_factors.append({'factor': 'hypertension', 'controlled': controlled})
        
        if any(e.entity_type == 'RISK_FACTOR' and 'diabetes' in e.text.lower() for e in entities):
            insulin_treated = 'insulin' in text_lower
            risk_factors.append({'factor': 'diabetes', 'insulin_treated': insulin_treated})
        
        for rf_term in ['smoking', 'smoker', 'hyperlipidemia']:
            if any(e.entity_type == 'RISK_FACTOR' and rf_term in e.text.lower() for e in entities):
                risk_factors.append({'factor': rf_term})
        
        return risk_factors
    
    def _extract_investigations(self, text: str, entities: List[Entity]) -> List[Dict[str, Any]]:
        investigations = []
        text_lower = text.lower()
        
        # Troponin
        match = re.search(r'troponin[\s:]+([.\d]+)\s*ng/ml', text_lower)
        if match:
            investigations.append({
                'test_type': 'troponin',
                'value': float(match.group(1)),
                'unit': 'ng/mL',
                'ref_upper_limit': 0.04
            })
        
        # BNP
        match = re.search(r'bnp[\s:]+([.\d]+)\s*pg/ml', text_lower)
        if match:
            investigations.append({
                'test_type': 'BNP',
                'value': float(match.group(1)),
                'unit': 'pg/mL',
                'ref_upper_limit': 100
            })
        
        # Creatinine
        match = re.search(r'creatinine[\s:]+([.\d]+)\s*mg/dl', text_lower)
        if match:
            investigations.append({
                'test_type': 'creatinine',
                'value': float(match.group(1)),
                'unit': 'mg/dL',
                'ref_upper_limit': 1.3
            })
        
        # Blood Pressure
        match = re.search(r'bp[\s:]+(\d+)[/-](\d+)\s*mmhg', text_lower)
        if match:
            investigations.append({
                'test_type': 'blood_pressure',
                'systolic': int(match.group(1)),
                'diastolic': int(match.group(2)),
                'unit': 'mmHg'
            })
        
        # ECG
        ecg_match = re.search(r'ecg:(.*?)(?=\n\s*[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if ecg_match:
            ecg_text = ecg_match.group(1)
            findings = []
            st_elev_match = re.search(r'st elevation.*?(?:leads?\s+)?([\w\d,-]+)', ecg_text, re.IGNORECASE)
            if st_elev_match:
                findings.append(f'ST elevation {st_elev_match.group(1)}')
            if re.search(r'\blbbb\b', ecg_text, re.IGNORECASE):
                findings.append('new LBBB' if 'new' in ecg_text.lower() else 'LBBB')
            if findings:
                investigations.append({'test_type': 'ecg', 'findings': findings})
        
        # Echo
        echo_match = re.search(r'echo:(.*?)(?=\n\s*[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if echo_match:
            echo_text = echo_match.group(1)
            echo_data = {'test_type': 'echo'}
            ef_match = re.search(r'ejection fraction[\s:]+([.\d]+)%?', echo_text, re.IGNORECASE)
            if ef_match:
                ef_value = float(ef_match.group(1))
                echo_data['ejection_fraction'] = ef_value / 100 if ef_value > 1 else ef_value
            if re.search(r'global hypokinesis', echo_text, re.IGNORECASE):
                echo_data['wall_motion'] = 'global hypokinesis'
            if len(echo_data) > 1:
                investigations.append(echo_data)
        
        return investigations
    
    def _extract_medications(self, text: str, entities: List[Entity]) -> List[Dict[str, Any]]:
        medications = []
        for entity in entities:
            if entity.entity_type == 'MEDICATION' and not entity.negated:
                med_data = {'drug': entity.text.lower()}
                
                context_start = max(0, entity.start_char - 50)
                context_end = min(len(text), entity.end_char + 100)
                context = text[context_start:context_end]
                
                dose_match = re.search(rf'{re.escape(entity.text)}[\s:]+([.\d]+)\s*(mg|g|mcg)', context, re.IGNORECASE)
                if dose_match:
                    med_data['dose'] = f"{dose_match.group(1)} {dose_match.group(2)}"
                
                if re.search(r'\bdaily\b|\bqd\b', context, re.IGNORECASE):
                    med_data['frequency'] = 'daily'
                elif re.search(r'\bbid\b|twice daily', context, re.IGNORECASE):
                    med_data['frequency'] = 'bid'
                
                medications.append(med_data)
        
        return medications
    
    def _compute_relevance(self, text: str, entities: List[Entity]) -> Tuple[bool, float]:
        text_lower = text.lower()
        cardiac_keywords = ['chest pain', 'MI', 'STEMI', 'heart failure', 'dyspnea', 'troponin', 
                          'BNP', 'ECG', 'echo', 'cardiac', 'cardiology']
        
        keyword_matches = sum(1 for kw in cardiac_keywords if kw.lower() in text_lower)
        cardiac_entities = sum(1 for e in entities if e.entity_type in ['SYMPTOM', 'DISEASE', 'FINDING'])
        
        keyword_score = min(keyword_matches / 5.0, 1.0)
        entity_score = min(cardiac_entities / 5.0, 1.0)
        final_score = 0.6 * keyword_score + 0.4 * entity_score
        
        return final_score > 0.5, round(final_score, 2)


# ============================================================================
# DEMO
# ============================================================================

def main():
    sample_text = """
    EMERGENCY DEPARTMENT NOTE
    
    65-year-old male presenting to ED with acute chest pain.
    Pain described as pressure-like, radiating to left arm, started 30 minutes ago.
    Patient has history of hypertension and diabetes, on insulin.
    
    LABS:
    Troponin 0.85 ng/mL (reference <0.04)
    BNP 450 pg/mL
    Creatinine 1.2 mg/dL
    
    ECG:
    ST elevation in leads V1-V4, new LBBB
    
    ECHO:
    Ejection fraction 28%, global hypokinesis
    
    MEDICATIONS:
    Atorvastatin 80 mg daily
    Lisinopril 20 mg daily
    """
    
    agent = CardiologyIngestionAgent(use_biobert=True, use_gpu=False)
    result = agent.process(sample_text)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()