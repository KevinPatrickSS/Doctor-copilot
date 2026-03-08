# 🏥 Doctor Copilot

**AI-Powered Clinical Decision Support System**

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Input Format](#input-format)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimers](#disclaimers)

---

## 🎯 Overview

Doctor Copilot is a **production-ready clinical decision support system** that combines:

- **AI/ML Models**: BioBERT NLP, Ensemble ML (Logistic Regression + XGBoost), SHAP explainability
- **Clinical Scoring**: RCRI (Revised Cardiac Risk Index) 0-6 scale
- **Knowledge Systems**: RAG (Retrieval-Augmented Generation) with ChromaDB
- **LLM Integration**: Google Gemini API for probabilistic diagnoses
- **Safety Layer**: Deterministic rule-based red flag detection
- **Professional UI**: React + Tailwind CSS responsive interface

The system processes **raw clinical notes** and generates **risk-stratified clinical recommendations** with:
- Explicit uncertainty quantification
- Explainable AI (SHAP)
- Rule-based safety checks
- Human-in-the-loop positioning

---

## ✨ Features

### Core Capabilities

✅ **6-Agent Modular Architecture**
- Ingestion & NLP (BioBERT)
- Risk Assessment (RCRI + ML + SHAP)
- RAG Retrieval (Clinical Guidelines)
- Clinical Reasoning (Gemini API)
- Safety Checks (Deterministic Rules)
- Report Generation (PDF/JSON)

✅ **Clinical Features**
- Cardiac risk assessment (RCRI 0-6 scoring)
- ML ensemble risk prediction
- SHAP feature importance explanations
- Automatic red flag detection
- Missing evaluation flagging
- Risk-stratified recommendations

✅ **Professional Interface**
- Modern React UI with Tailwind CSS
- Real-time analysis progress
- Multi-tab report viewing (Patient Data, Report, Details)
- Safety assessment visualization
- One-click PDF/JSON export

✅ **Production Quality**
- REST API with proper error handling
- CORS support for cross-origin requests
- Comprehensive logging
- Health check endpoints
- Structured JSON outputs

✅ **Safety-First**
- Non-final diagnosis positioning
- Explicit uncertainty quantification
- Deterministic safety layer
- Critical finding alerts
- Audit trail in all reports

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────┐
│         DOCTOR COPILOT PIPELINE             │
└─────────────────────────────────────────────┘

FRONTEND (React + Tailwind CSS)
  ├─ Patient Data Input
  ├─ Real-time Progress
  └─ Report Visualization

         ↓ HTTP/JSON ↓

BACKEND (Flask REST API)
  ├─ POST /api/analyze
  ├─ POST /api/export-pdf
  ├─ POST /api/export-json
  └─ GET /api/system-status

         ↓ Orchestration ↓

[AGENT 1] Ingestion & NLP (BioBERT)
  └─ Extracts structured data from raw text

[AGENT 2] Risk Assessment (RCRI + ML + SHAP)
  └─ Cardiac risk scoring and explainability

[AGENT 3] RAG Retrieval (ChromaDB)
  └─ Retrieves clinical guidelines

[AGENT 4] Clinical Reasoning (Gemini)
  └─ Generates probabilistic diagnoses

[AGENT 5] Safety Checks (Rules)
  └─ Detects red flags and missing data

[AGENT 6] Report Generation
  └─ Outputs structured JSON/PDF

         ↓ Output ↓

CLINICAL REPORT
  ├─ Patient Summary
  ├─ Risk Assessment
  ├─ Clinical Assessment
  ├─ Safety Assessment
  └─ Physician Actions
```

---

## 📦 Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Node.js**: 16.x or higher (for frontend)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB for dependencies and models
- **OS**: Windows, macOS, or Linux

### Required Services

- **Google Gemini API**: Sign up at https://makersuite.google.com/app/apikey
- **Internet**: For Gemini API calls (other components are local)

### Optional

- **Docker**: For containerized deployment
- **PostgreSQL**: For production data storage
- **Redis**: For caching (production)

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/doctor-copilot.git
cd doctor-copilot
```

### Step 2: Create Virtual Environment (Backend)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Install Frontend Dependencies

```bash
# Navigate to frontend directory (if separate)
cd frontend  # or just stay in root if using monorepo structure

# Install npm dependencies
npm install

# Or with legacy peer deps if issues occur
npm install --legacy-peer-deps
```

### Step 5: Set Environment Variables

Create `.env` file in project root:

```bash
# Google Gemini API
GEMINI_API_KEY=your_api_key_here

# Backend configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000

# Frontend configuration
REACT_APP_API_BASE_URL=http://localhost:5000

# Logging
LOG_LEVEL=INFO
```

### Step 6: Verify Installation

```bash
# Check Python installation
python --version  # Should be 3.10+

# Check pip packages
pip list | grep -E "flask|react|transformers"

# Test backend startup (from backend directory)
python app.py

# Test frontend (from frontend directory, in another terminal)
npm start
```

---

## ⚡ Quick Start (5 Minutes)

### 1. Get Gemini API Key

```
1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Set environment: export GEMINI_API_KEY="your_key"
```

### 2. Install & Setup

```bash
# Backend setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Frontend setup
npm install
```

### 3. Start Backend (Terminal 1)

```bash
export GEMINI_API_KEY="your_api_key"
python app.py
# Should show: "Running on http://localhost:5000"
```

### 4. Start Frontend (Terminal 2)

```bash
npm start
# Should open http://localhost:3000 in browser
```

### 5. Test System

```bash
# In browser, go to: http://localhost:3000
# Paste sample patient data:

65-year-old male with acute chest pain.
BP: 165/95 mmHg, HR: 102 bpm
History of hypertension and diabetes
Troponin: 0.85 ng/mL (HIGH)
ECG: ST elevation V1-V4

# Click "Analyze Patient Data"
# Wait 10-15 seconds for analysis
# Review report with risk assessment
```

---

## 📁 Project Structure

```
doctor-copilot/
├── backend/
│   ├── app.py                          # Flask REST API
│   ├── orchestrator.py                 # Main orchestrator (5 agents)
│   ├── orchestrator_with_risk_assessment.py  # Enhanced (6 agents)
│   ├── cardiology_ingestion_agent.py   # NLP agent
│   ├── rag_system.py                   # RAG + ChromaDB
│   ├── cardiac_risk_agent.py           # Risk assessment agent
│   ├── requirements.txt                # Python dependencies
│   └── chroma_db/                      # Vector database (auto-created)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # Main React component
│   │   ├── main.jsx                    # React entry point
│   │   └── index.css                   # Tailwind CSS
│   ├── public/
│   │   └── index.html                  # HTML entry
│   ├── vite.config.js                  # Vite configuration
│   ├── tailwind.config.js              # Tailwind configuration
│   ├── postcss.config.js               # PostCSS configuration
│   ├── package.json                    # npm dependencies
│   └── package-lock.json
│
├── documentation/
│   ├── README.md                       # This file
│   ├── SETUP.md                        # Detailed setup guide
│   ├── API.md                          # API reference
│   ├── INPUT_FORMAT.md                 # Input specification
│   ├── RESUME_DESCRIPTIONS.md          # Project descriptions
│   └── INTEGRATION_GUIDE.md            # Agent integration
│
├── scripts/
│   ├── setup.sh                        # Automated setup
│   ├── demo.py                         # Demo with sample cases
│   └── verify.sh                       # Verification script
│
├── .env.example                        # Environment template
├── .gitignore                          # Git ignore rules
└── LICENSE                             # MIT License
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Analyze Patient Data
```http
POST /api/analyze
Content-Type: application/json

{
  "patient_data": "raw clinical text here"
}

Response:
{
  "success": true,
  "report": {
    "report_id": "DCR-20240130120000",
    "patient_summary": {...},
    "risk_assessment": {...},
    "clinical_assessment": "...",
    "safety_assessment": {...},
    "physician_actions": [...]
  }
}
```

#### 2. Export to PDF
```http
POST /api/export-pdf
Content-Type: application/json

{
  "report": {report object}
}

Response: Binary PDF file
```

#### 3. Export to JSON
```http
POST /api/export-json
Content-Type: application/json

{
  "report": {report object}
}

Response: JSON file
```

#### 4. System Status
```http
GET /api/system-status

Response:
{
  "status": "ready",
  "components": {
    "ingestion_agent": "ready",
    "risk_assessment_agent": "ready",
    "rag_system": "ready",
    "gemini_api": "ready",
    "database": "ready"
  }
}
```

#### 5. Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-30T12:00:00Z"
}
```

---

## 📥 Input Format

The system accepts **any clinical text format**:

### Minimal Input
```
65-year-old male with chest pain.
BP: 165/95, HR: 102.
History of hypertension and diabetes.
Troponin: 0.85 (HIGH).
```

### Optimal Input
```
EMERGENCY DEPARTMENT NOTE

DEMOGRAPHICS:
Age: 65, Sex: Male

CHIEF COMPLAINT:
Acute chest pain for 30 minutes

HISTORY OF PRESENT ILLNESS:
Pressure-like chest pain radiating to left arm.
Associated with dyspnea.

RISK FACTORS:
- Hypertension
- Type 2 Diabetes (on insulin)
- Hyperlipidemia
- Previous MI 5 years ago

VITAL SIGNS:
BP: 165/95 mmHg
HR: 102 bpm
RR: 18
O2 Saturation: 97% RA

LABS:
Troponin I: 0.85 ng/mL [HIGH]
BNP: 450 pg/mL [HIGH]
Creatinine: 1.2 mg/dL

ECG:
ST elevation V1-V4
New LBBB pattern

ECHOCARDIOGRAPHY:
Ejection Fraction: 28% [REDUCED]
Global hypokinesis

MEDICATIONS:
- Atorvastatin 80 mg daily
- Lisinopril 20 mg daily
- Metoprolol 50 mg BID
- Aspirin 325 mg daily
```

**Supported formats**:
- ✅ Unstructured narrative
- ✅ Semi-structured with headers
- ✅ Key-value pairs
- ✅ Bullet points
- ✅ Tables
- ✅ JSON
- ✅ XML
- ✅ Medical abbreviations (HTN, DM, MI, STEMI, LBBB, etc.)

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```bash
# Gemini API
GEMINI_API_KEY=your_api_key_here

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Frontend
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_API_TIMEOUT=30000

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log

# Database (if using production)
DATABASE_URL=postgresql://user:password@localhost/doctor_copilot

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379
```

### Backend Configuration

Edit `app.py` to customize:

```python
# Line 15: Change orchestrator version
from orchestrator_with_risk_assessment import DoctorCopilotOrchestratorWithRiskAssessment

# Line 315: Change Flask port
app.run(debug=True, port=5000, host='0.0.0.0')

# Line 20: Change CORS origins
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}})
```

### Frontend Configuration

Edit `vite.config.js` to customize:

```javascript
// Port configuration
server: {
  port: 3000,
  strictPort: false,
  host: '127.0.0.1'
}
```

---

## 📊 Usage Examples

### Python

```python
import requests
import json

# Prepare patient data
patient_data = """
65-year-old male with acute chest pain.
BP: 165/95, HR: 102.
History of hypertension and diabetes.
Troponin: 0.85 (HIGH).
ECG: ST elevation V1-V4.
"""

# Send to API
response = requests.post(
    'http://localhost:5000/api/analyze',
    json={'patient_data': patient_data}
)

# Parse response
report = response.json()['report']

# Print risk assessment
print(f"Risk Level: {report['risk_assessment']['overall_risk_level']}")
print(f"RCRI Score: {report['risk_assessment']['rcri_score']}/6")
print(f"Red Flags: {report['safety_assessment']['red_flags']}")

# Save report
with open('report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

### JavaScript/React

```javascript
const analyzePatient = async (patientData) => {
  try {
    const response = await fetch(
      'http://localhost:5000/api/analyze',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_data: patientData })
      }
    );
    
    const result = await response.json();
    
    if (result.success) {
      console.log('Risk Level:', result.report.risk_assessment.overall_risk_level);
      console.log('RCRI Score:', result.report.risk_assessment.rcri_score);
      return result.report;
    } else {
      console.error('Analysis failed:', result.error);
    }
  } catch (error) {
    console.error('API Error:', error);
  }
};

// Usage
const report = await analyzePatient("65-year-old male with chest pain...");
```

### cURL

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": "65-year-old male with chest pain. BP: 165/95, HR: 102. History of hypertension and diabetes. Troponin: 0.85."
  }' | json_pp
```

---

## 🧪 Testing

### Run Demo

```bash
# Test with 3 sample cases (STEMI, HF, Low-risk)
python scripts/demo.py

# Output shows:
# - Full analysis for each case
# - Risk assessment results
# - Generated reports
# - Saved to output/
```

### Verification Checklist

```bash
# Run automated verification
bash scripts/verify.sh

# Checks:
# ✓ Python version >= 3.10
# ✓ All dependencies installed
# ✓ Gemini API key set
# ✓ Backend starts successfully
# ✓ Frontend builds
# ✓ Health endpoints respond
```

### Manual Testing

```bash
# Test backend health
curl http://localhost:5000/health

# Test system status
curl http://localhost:5000/api/system-status

# Test analysis
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"patient_data": "Test patient data"}'
```

---

## 🔧 Troubleshooting

### Common Issues

#### Backend Won't Start

```bash
# Check Python version
python --version  # Must be 3.10+

# Check dependencies
pip list | grep flask

# Reinstall
pip install --force-reinstall -r requirements.txt

# Check port 5000 available
lsof -i :5000  # Kill if needed: kill -9 <PID>
```

#### Frontend Styles Not Showing

```bash
# Clean install
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --legacy-peer-deps
npm start

# Hard refresh browser: Ctrl+Shift+R
```

#### Gemini API Errors

```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Check API key validity at:
# https://makersuite.google.com/app/apikey

# Ensure key has correct permissions
```

#### Missing Modules

```bash
# Reinstall all dependencies
pip install -r requirements.txt --no-cache-dir

# Download spaCy model
python -m spacy download en_core_web_sm

# Check ChromaDB
pip install chromadb --upgrade
```

#### Port Already in Use

```bash
# Find process using port
lsof -i :5000

# Kill process
kill -9 <PID>

# Or change port in code
# app.py line 315: app.run(port=5001)
```

See [SETUP.md](documentation/SETUP.md) for more detailed troubleshooting.

---

## 📚 Documentation

- **[SETUP.md](documentation/SETUP.md)** - Detailed setup guide with configuration
- **[API.md](documentation/API.md)** - Complete API reference
- **[INPUT_FORMAT.md](documentation/INPUT_FORMAT.md)** - Input specification with examples
- **[INTEGRATION_GUIDE.md](documentation/INTEGRATION_GUIDE.md)** - Integrating custom agents
- **[RESUME_DESCRIPTIONS.md](documentation/RESUME_DESCRIPTIONS.md)** - Project descriptions

---

## 🔐 Security

### Best Practices

- ✅ Never commit `.env` file
- ✅ Use strong API keys
- ✅ Enable HTTPS in production
- ✅ Validate all inputs
- ✅ Use rate limiting
- ✅ Keep dependencies updated

### Production Deployment

```bash
# Update packages
pip install --upgrade pip
pip install -r requirements.txt

# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Enable HTTPS with nginx
# (See documentation for details)

# Use PostgreSQL instead of SQLite
# (For data persistence)
```

---

## 🤝 Contributing

### Getting Started

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Code Style

- Python: PEP 8 with 88-char lines (Black formatter)
- JavaScript: Prettier with 2-space indentation
- Commit messages: Conventional commits

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
black . --check
pylint src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimers

### Important Legal Notice

**🚨 NOT FOR CLINICAL USE**

This system is provided **AS-IS** for:
- Research purposes
- Educational demonstrations
- Proof-of-concept development
- System prototyping

**NOT suitable for clinical use without:**
- ✅ Physician validation and oversight
- ✅ IRB (Institutional Review Board) approval
- ✅ Regulatory compliance (FDA, etc.)
- ✅ Healthcare professional integration
- ✅ Formal validation studies

### System Limitations

- ⚠️ Provides **PRELIMINARY** clinical support only
- ⚠️ **NOT** a substitute for professional medical judgment
- ⚠️ All recommendations must be reviewed by qualified physicians
- ⚠️ Risk assessments are **probabilistic**, not definitive
- ⚠️ System may miss important diagnoses
- ⚠️ Not trained on all possible clinical scenarios

### Data Privacy

- ✅ All processing is local (no cloud storage)
- ✅ Data not stored permanently
- ✅ Only Gemini API calls use external service
- ⚠️ Use with de-identified data only
- ⚠️ HIPAA compliant architecture but not certified

### Liability

By using this system, you acknowledge:
- System is provided without warranty
- Authors not liable for clinical decisions
- You assume all responsibility for use
- User must obtain proper authorization

---

## 📞 Support

### Getting Help

- **Issues**: GitHub Issues tab
- **Documentation**: See [documentation/](documentation/) folder
- **Email**: support@doctorcopilot.dev
- **Community**: GitHub Discussions

### Resources

- [Gemini API Documentation](https://ai.google.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
- [BioBERT Paper](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)

---

## 🎉 Acknowledgments

Built with:
- [Google Gemini API](https://ai.google.dev/) - LLM reasoning
- [BioBERT](https://github.com/dmis-lab/biobert) - Clinical NLP
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [React](https://react.dev/) - Frontend framework
- [Flask](https://flask.palletsprojects.com/) - Backend framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [SHAP](https://shap.readthedocs.io/) - ML explainability

---

## 📈 Roadmap

Future enhancements:

- [ ] Multi-language support
- [ ] Additional medical domains (neurology, oncology)
- [ ] Database integration for data persistence
- [ ] User authentication and role-based access
- [ ] Advanced visualization dashboards
- [ ] Real-time collaboration features
- [ ] Mobile app
- [ ] Integration with EHR systems

---

## 📊 Project Stats

- **Lines of Code**: ~6,100
- **Files**: 16 production-ready
- **Test Cases**: 3 validated samples
- **Documentation**: 5 comprehensive guides
- **Languages**: Python, JavaScript/React
- **Dependencies**: 30+
- **Processing Time**: 10-15 seconds per analysis

---

## 🚀 Getting Started Now

1. **Clone**: `git clone https://github.com/yourusername/doctor-copilot.git`
2. **Install**: Follow [Installation](#installation) section
3. **Configure**: Set `GEMINI_API_KEY` in `.env`
4. **Start**: `python app.py` (Terminal 1) + `npm start` (Terminal 2)
5. **Test**: Go to http://localhost:3000
6. **Read**: Check [documentation/](documentation/) for details

---

**Version**: 1.1.0  
**Last Updated**: March 30, 2026  
**Status**: ✅ Production Ready  


---

**Happy analyzing! 🏥** 

*Remember: Always defer final clinical decisions to qualified physicians.*
