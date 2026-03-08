"""
diagnose_biobert.py
Diagnostic script to check BioBERT setup
Run this in your virtual environment to see what's missing
"""

import sys
import importlib

print("=" * 80)
print("BIOBERT DEPENDENCY DIAGNOSTIC")
print("=" * 80)
print()

# Check Python version
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print()

# Required packages
required_packages = {
    'torch': 'PyTorch',
    'transformers': 'Hugging Face Transformers',
    'spacy': 'spaCy NLP',
    'numpy': 'NumPy'
}

print("Checking required packages:")
print("-" * 80)

all_available = True

for package, name in required_packages.items():
    try:
        module = importlib.import_module(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {name:30s} - Installed (v{version})")
    except ImportError as e:
        print(f"❌ {name:30s} - MISSING")
        all_available = False
        print(f"   Error: {e}")

print()

# Check spaCy model
print("Checking spaCy language model:")
print("-" * 80)

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ en_core_web_sm - Installed")
    except OSError:
        print("❌ en_core_web_sm - MISSING")
        print("   Run: python -m spacy download en_core_web_sm")
        all_available = False
except ImportError:
    print("❌ Cannot check - spaCy not installed")
    all_available = False

print()

# Try to import BioBERT agent
print("Checking BioBERT agent import:")
print("-" * 80)

try:
    from cardiology_nlp_agent import BioBERTCardiologyAgent
    print("✅ BioBERTCardiologyAgent - Importable")
    
    # Try to initialize
    try:
        print("   Attempting to initialize BioBERT agent...")
        agent = BioBERTCardiologyAgent()
        print("   ✅ BioBERT agent initialized successfully!")
        
        # Test with sample text
        print()
        print("Testing BioBERT agent:")
        print("-" * 80)
        test_text = "65 year old male with chest pain and elevated troponin"
        result = agent.extract_entities(test_text)
        print(f"   Test text: '{test_text}'")
        print(f"   Entities extracted: {len(result.get('entities', []))}")
        
        if result.get('entities'):
            print("   Sample entities:")
            for entity in result['entities'][:3]:
                print(f"      - {entity}")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        all_available = False
        
except ImportError as e:
    print(f"❌ BioBERTCardiologyAgent - Cannot import")
    print(f"   Error: {e}")
    all_available = False

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if all_available:
    print("✅ All BioBERT dependencies are properly installed!")
    print("   BioBERT should work in Streamlit.")
else:
    print("❌ Some dependencies are missing.")
    print()
    print("INSTALLATION COMMANDS:")
    print("-" * 80)
    print("# Activate your virtual environment first:")
    print("# On Windows PowerShell:")
    print(".\\chroma_legacy\\Scripts\\Activate.ps1")
    print()
    print("# Then install missing packages:")
    print("pip install torch transformers spacy numpy")
    print("python -m spacy download en_core_web_sm")
    print()
    print("# Verify installation:")
    print("python diagnose_biobert.py")

print()
print("After fixing dependencies, restart Streamlit:")
print("streamlit run app.py")
print("=" * 80)