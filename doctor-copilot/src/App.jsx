import React, { useState, useRef } from 'react';
import { AlertCircle, CheckCircle, AlertTriangle, Download, Copy, RefreshCw } from 'lucide-react';

export default function DoctorCopilot() {
  const [patientData, setPatientData] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('input');
  const textAreaRef = useRef(null);

  const handleAnalyze = async () => {
    if (!patientData.trim()) {
      setError('Please enter patient data');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_data: patientData })
      });

      const result = await response.json();

      if (result.success) {
        setReport(result.report);
        setActiveTab('report');
      } else {
        setError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setError(`Connection error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleExportPDF = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/export-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report })
      });

      if (!response.ok) throw new Error('PDF export failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${report.report_id}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Export failed: ${err.message}`);
    }
  };

  const handleExportJSON = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/export-json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report })
      });

      if (!response.ok) throw new Error('JSON export failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${report.report_id}.json`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Export failed: ${err.message}`);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
  };

  const handleClear = () => {
    setPatientData('');
    setReport(null);
    setError(null);
    setActiveTab('input');
    textAreaRef.current?.focus();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-blue-700">
      {/* Header */}
      <header className="bg-white shadow-lg border-b-4 border-blue-600">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">🏥</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Doctor Copilot</h1>
                <p className="text-sm text-gray-600">AI-Powered Clinical Decision Support</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-500">v1.0.0</p>
              <p className="text-xs text-green-600 font-semibold">● System Ready</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Disclaimer Banner */}
        <div className="mb-6 bg-red-50 border-2 border-red-400 rounded-lg p-4 flex gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-900">⚠️ Important Disclaimer</p>
            <p className="text-sm text-red-800 mt-1">
              This system provides PRELIMINARY clinical support only. It is NOT a substitute for professional medical judgment. All recommendations must be reviewed and confirmed by qualified physicians.
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab('input')}
            className={`px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'input'
                ? 'bg-white text-blue-600 shadow-lg'
                : 'bg-blue-700 text-white hover:bg-blue-600'
            }`}
          >
            📋 Patient Data
          </button>
          <button
            onClick={() => setActiveTab('report')}
            disabled={!report}
            className={`px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'report'
                ? 'bg-white text-blue-600 shadow-lg'
                : report
                ? 'bg-blue-700 text-white hover:bg-blue-600 cursor-pointer'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }`}
          >
            📊 Report
          </button>
          <button
            onClick={() => setActiveTab('details')}
            disabled={!report}
            className={`px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'details'
                ? 'bg-white text-blue-600 shadow-lg'
                : report
                ? 'bg-blue-700 text-white hover:bg-blue-600 cursor-pointer'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }`}
          >
            📈 Details
          </button>
        </div>

        {/* Input Tab */}
        {activeTab === 'input' && (
          <div className="bg-white rounded-lg shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Enter Patient Data</h2>
            <p className="text-gray-600 mb-6">
              Paste clinical notes, lab results, and patient information. The system will extract key data and provide clinical insights.
            </p>

            <textarea
              ref={textAreaRef}
              value={patientData}
              onChange={(e) => setPatientData(e.target.value)}
              placeholder="EMERGENCY DEPARTMENT NOTE&#10;&#10;65-year-old male presenting with acute chest pain...&#10;&#10;VITAL SIGNS:&#10;BP: 165/95 mmHg&#10;HR: 102&#10;&#10;LABS:&#10;Troponin: 0.85 ng/mL&#10;ECG: ST elevation in V1-V4..."
              className="w-full h-64 p-4 border-2 border-gray-300 rounded-lg font-mono text-sm focus:outline-none focus:border-blue-500 resize-none"
            />

            {error && (
              <div className="mt-4 bg-red-50 border border-red-300 rounded-lg p-4 flex gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <p className="text-red-800">{error}</p>
              </div>
            )}

            <div className="mt-6 flex gap-4">
              <button
                onClick={handleAnalyze}
                disabled={loading || !patientData.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg transition flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span>🔍 Analyze Patient Data</span>
                  </>
                )}
              </button>
              <button
                onClick={handleClear}
                className="px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-lg transition"
              >
                Clear
              </button>
            </div>

            {loading && (
              <div className="mt-8 p-6 bg-blue-50 border border-blue-300 rounded-lg">
                <p className="text-blue-900 font-semibold mb-3">Processing...</p>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                    <span className="text-sm text-blue-800">Ingesting and parsing patient data</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                    <span className="text-sm text-blue-800">Retrieving clinical guidelines</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    <span className="text-sm text-blue-800">Generating clinical insights</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.6s' }}></div>
                    <span className="text-sm text-blue-800">Running safety checks</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Report Tab */}
        {activeTab === 'report' && report && (
          <div className="bg-white rounded-lg shadow-xl p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Clinical Report</h2>
              <div className="flex gap-3">
                <button
                  onClick={handleExportPDF}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition"
                >
                  <Download className="w-4 h-4" />
                  PDF
                </button>
                <button
                  onClick={handleExportJSON}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition"
                >
                  <Download className="w-4 h-4" />
                  JSON
                </button>
              </div>
            </div>

            {/* Disclaimer */}
            <div className="mb-6 p-4 bg-yellow-50 border-2 border-yellow-400 rounded-lg">
              <p className="text-sm text-yellow-900">{report.disclaimer}</p>
            </div>

            {/* Report ID and Timestamp */}
            <div className="mb-6 grid grid-cols-2 gap-4 text-sm text-gray-600">
              <div>
                <p className="font-semibold text-gray-900">Report ID</p>
                <p className="font-mono">{report.report_id}</p>
              </div>
              <div>
                <p className="font-semibold text-gray-900">Generated</p>
                <p>{new Date(report.timestamp).toLocaleString()}</p>
              </div>
            </div>

            {/* Patient Summary */}
            <div className="mb-6">
              <h3 className="text-lg font-bold text-gray-900 mb-3">Patient Information</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold">Age</p>
                  <p className="text-lg font-bold text-gray-900">{report.patient_summary.age || 'N/A'}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold">Sex</p>
                  <p className="text-lg font-bold text-gray-900">{report.patient_summary.sex || 'N/A'}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold">Encounter Type</p>
                  <p className="text-lg font-bold text-gray-900">{report.patient_summary.encounter_type || 'N/A'}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold">Cardiology Relevant</p>
                  <p className="text-lg font-bold text-gray-900">{report.patient_summary.cardiology_relevant ? 'Yes' : 'No'}</p>
                </div>
              </div>
            </div>

            {/* Safety Assessment */}
            <div className="mb-6">
              <h3 className="text-lg font-bold text-gray-900 mb-3">Safety Assessment</h3>
              <div className={`p-4 rounded-lg border-2 ${
                report.safety_assessment.status === 'CLEAR'
                  ? 'bg-green-50 border-green-300'
                  : 'bg-red-50 border-red-300'
              }`}>
                <div className="flex items-center gap-2 mb-4">
                  {report.safety_assessment.status === 'CLEAR' ? (
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  ) : (
                    <AlertTriangle className="w-6 h-6 text-red-600" />
                  )}
                  <span className={`font-bold ${
                    report.safety_assessment.status === 'CLEAR'
                      ? 'text-green-900'
                      : 'text-red-900'
                  }`}>
                    Status: {report.safety_assessment.status}
                  </span>
                </div>

                {report.safety_assessment.red_flags.length > 0 && (
                  <div className="mb-4">
                    <p className="font-semibold text-red-900 mb-2">🚩 Red Flags</p>
                    <ul className="space-y-1">
                      {report.safety_assessment.red_flags.map((flag, idx) => (
                        <li key={idx} className="text-sm text-red-800">• {flag}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {report.safety_assessment.warnings.length > 0 && (
                  <div className="mb-4">
                    <p className="font-semibold text-orange-900 mb-2">⚠️ Warnings</p>
                    <ul className="space-y-1">
                      {report.safety_assessment.warnings.map((warn, idx) => (
                        <li key={idx} className="text-sm text-orange-800">• {warn}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {report.safety_assessment.missing_evaluations.length > 0 && (
                  <div>
                    <p className="font-semibold text-yellow-900 mb-2">📋 Missing Evaluations</p>
                    <ul className="space-y-1">
                      {report.safety_assessment.missing_evaluations.map((missing, idx) => (
                        <li key={idx} className="text-sm text-yellow-800">• {missing}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Clinical Assessment */}
            <div className="mb-6">
              <h3 className="text-lg font-bold text-gray-900 mb-3">Clinical Assessment</h3>
              <div className="bg-blue-50 p-6 rounded-lg border border-blue-300">
                <div className="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap font-sans">
                  {report.clinical_assessment}
                </div>
              </div>
            </div>

            {/* Physician Actions */}
            <div>
              <h3 className="text-lg font-bold text-gray-900 mb-3">Recommended Physician Actions</h3>
              <div className="space-y-2">
                {report.physician_actions.map((action, idx) => (
                  <div key={idx} className="flex gap-3 p-3 bg-gray-50 rounded-lg">
                    <span className="text-blue-600 font-bold flex-shrink-0">{idx + 1}</span>
                    <p className="text-gray-800">{action}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Details Tab */}
        {activeTab === 'details' && report && (
          <div className="bg-white rounded-lg shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Detailed Patient Data</h2>

            {/* Symptoms */}
            {report.clinical_data.symptoms.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-bold text-gray-900 mb-3">Symptoms</h3>
                <div className="space-y-2">
                  {report.clinical_data.symptoms.map((sym, idx) => (
                    <div key={idx} className="p-3 bg-blue-50 rounded-lg border border-blue-300">
                      <p className="font-semibold text-blue-900">{sym.symptom || 'N/A'}</p>
                      {(sym.character || sym.radiation || sym.onset) && (
                        <p className="text-sm text-blue-800 mt-1">
                          {[sym.character, sym.radiation, sym.onset].filter(Boolean).join(' · ')}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Risk Factors */}
            {report.clinical_data.risk_factors.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-bold text-gray-900 mb-3">Risk Factors</h3>
                <div className="space-y-2">
                  {report.clinical_data.risk_factors.map((rf, idx) => (
                    <div key={idx} className="p-3 bg-orange-50 rounded-lg border border-orange-300">
                      <p className="font-semibold text-orange-900">{rf.factor}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Investigations */}
            {report.clinical_data.investigations.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-bold text-gray-900 mb-3">Investigations & Results</h3>
                <div className="space-y-3">
                  {report.clinical_data.investigations.map((inv, idx) => (
                    <div key={idx} className="p-4 bg-green-50 rounded-lg border border-green-300">
                      <p className="font-bold text-green-900 mb-2">{inv.test_type?.toUpperCase() || 'Test'}</p>
                      <p className="text-sm text-green-800">
                        {Object.entries(inv)
                          .filter(([k]) => k !== 'test_type')
                          .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(', ') : v}`)
                          .join(' | ')}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medications */}
            {report.clinical_data.medications.length > 0 && (
              <div>
                <h3 className="text-lg font-bold text-gray-900 mb-3">Current Medications</h3>
                <div className="space-y-2">
                  {report.clinical_data.medications.map((med, idx) => (
                    <div key={idx} className="p-3 bg-purple-50 rounded-lg border border-purple-300">
                      <p className="font-semibold text-purple-900">{med.drug}</p>
                      {(med.dose || med.frequency) && (
                        <p className="text-sm text-purple-800 mt-1">
                          {[med.dose, med.frequency].filter(Boolean).join(' · ')}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Raw JSON */}
            <div className="mt-8 pt-6 border-t-2 border-gray-300">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-bold text-gray-900">Raw Report Data</h3>
                <button
                  onClick={copyToClipboard}
                  className="flex items-center gap-2 px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded transition"
                >
                  <Copy className="w-4 h-4" />
                  Copy
                </button>
              </div>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto text-xs font-mono max-h-96">
                {JSON.stringify(report, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 bg-gray-900 text-gray-300 py-6 text-center text-sm">
        <p>© 2024 Doctor Copilot | Clinical Decision Support System</p>
        <p className="mt-2 text-xs">⚠️ For research and educational purposes only. Not intended for clinical use without physician validation.</p>
      </footer>
    </div>
  );
}