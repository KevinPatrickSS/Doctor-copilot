"""
Doctor Copilot - Flask Backend API
Provides REST endpoints for the frontend UI
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import traceback

from orchestrator import DoctorCopilotOrchestrator

app = Flask(__name__)
CORS(app)

# Initialize orchestrator
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not set. Set with: export GEMINI_API_KEY=your_key")

orchestrator = None

def init_orchestrator():
    """Initialize orchestrator on first use."""
    global orchestrator
    if orchestrator is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        orchestrator = DoctorCopilotOrchestrator(GEMINI_API_KEY)
    return orchestrator


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_patient():
    """
    Main analysis endpoint.
    
    Request body:
    {
        "patient_data": "raw clinical text..."
    }
    
    Response:
    {
        "success": true/false,
        "report": {...},
        "error": "error message if failed"
    }
    """
    try:
        data = request.get_json()
        if not data or 'patient_data' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'patient_data' field in request"
            }), 400
        
        patient_text = data['patient_data'].strip()
        if not patient_text:
            return jsonify({
                "success": False,
                "error": "Patient data cannot be empty"
            }), 400
        
        print(f"\n📋 Received patient data ({len(patient_text)} chars)")
        
        # Initialize if needed
        orch = init_orchestrator()
        
        # Process
        print("🔄 Starting analysis...")
        report = orch.process_patient_data(patient_text)
        
        return jsonify({
            "success": True,
            "report": report
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n❌ Error: {error_trace}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": error_trace
        }), 500


@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    """
    Export report as PDF with improved formatting.
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from io import BytesIO
        from flask import send_file
        import re
        
        data = request.get_json()
        report = data.get('report', {})
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=16,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        h1_style = ParagraphStyle(
            'CustomH1',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#1e40af'),
            borderPadding=8,
            backColor=colors.HexColor('#eff6ff')
        )
        
        h2_style = ParagraphStyle(
            'CustomH2',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        h3_style = ParagraphStyle(
            'CustomH3',
            parent=styles['Heading3'],
            fontSize=10,
            textColor=colors.HexColor('#374151'),
            spaceAfter=6,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#1f2937'),
            leftIndent=20,
            spaceAfter=6,
            leading=14
        )
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.red,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            borderWidth=2,
            borderColor=colors.red,
            borderPadding=10,
            backColor=colors.HexColor('#fef2f2')
        )
        
        # Title
        elements.append(Paragraph("CLINICAL DECISION SUPPORT REPORT", title_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Disclaimer
        disclaimer_text = "⚠️ " + report.get('disclaimer', 'PRELIMINARY CLINICAL DECISION SUPPORT - NOT A FINAL DIAGNOSIS. Requires physician review and confirmation.')
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Summary
        elements.append(Paragraph("PATIENT INFORMATION", h1_style))
        patient = report.get('patient_summary', {})
        patient_data = [
            ['Age:', str(patient.get('age', 'N/A'))],
            ['Sex:', str(patient.get('sex', 'N/A'))],
            ['Encounter Type:', str(patient.get('encounter_type', 'N/A'))],
            ['Cardiology Relevant:', str(patient.get('cardiology_relevant', False))],
        ]
        t = Table(patient_data, colWidths=[2*inch, 4.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.2*inch))
        
        # Clinical Assessment - Parse markdown-like formatting
        elements.append(Paragraph("CLINICAL ASSESSMENT", h1_style))
        assessment = report.get('clinical_assessment', 'No assessment available')
        
        # Function to parse and format the assessment text
        def parse_markdown_text(text):
            """Parse markdown-like text into formatted paragraphs"""
            lines = text.split('\n')
            formatted_elements = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    formatted_elements.append(Spacer(1, 0.05*inch))
                    continue
                
                # Check for different markdown patterns
                if line.startswith('###'):
                    # H3 heading
                    heading_text = line.replace('###', '').strip()
                    formatted_elements.append(Paragraph(heading_text, h2_style))
                elif line.startswith('##'):
                    # H2 heading
                    heading_text = line.replace('##', '').strip()
                    formatted_elements.append(Paragraph(heading_text, h2_style))
                elif line.startswith('**') and line.endswith('**'):
                    # Bold text on its own line
                    bold_text = line.replace('**', '').strip()
                    formatted_elements.append(Paragraph(f"<b>{bold_text}</b>", body_style))
                elif line.startswith('* **') or line.startswith('- **'):
                    # Bullet with bold
                    bullet_text = re.sub(r'^[\*\-]\s*', '', line)
                    bullet_text = bullet_text.replace('**', '<b>', 1).replace('**', '</b>', 1)
                    formatted_elements.append(Paragraph(f"• {bullet_text}", bullet_style))
                elif line.startswith('* ') or line.startswith('- '):
                    # Regular bullet
                    bullet_text = re.sub(r'^[\*\-]\s*', '', line)
                    formatted_elements.append(Paragraph(f"• {bullet_text}", bullet_style))
                elif '**' in line:
                    # Inline bold formatting
                    formatted_line = line.replace('**', '<b>', 1)
                    count = formatted_line.count('**')
                    for i in range(count):
                        if i % 2 == 0:
                            formatted_line = formatted_line.replace('**', '</b>', 1)
                        else:
                            formatted_line = formatted_line.replace('**', '<b>', 1)
                    formatted_elements.append(Paragraph(formatted_line, body_style))
                else:
                    # Regular paragraph
                    formatted_elements.append(Paragraph(line, body_style))
            
            return formatted_elements
        
        # Add formatted assessment
        elements.extend(parse_markdown_text(assessment))
        elements.append(Spacer(1, 0.2*inch))
        
        # Safety Assessment
        safety = report.get('safety_assessment', {})
        elements.append(Paragraph("SAFETY ASSESSMENT", h1_style))
        
        red_flags = safety.get('red_flags', [])
        if red_flags:
            elements.append(Paragraph(f"🚩 <b>Red Flags ({len(red_flags)}):</b>", h3_style))
            for flag in red_flags:
                elements.append(Paragraph(f"• {flag}", bullet_style))
            elements.append(Spacer(1, 0.1*inch))
        
        warnings = safety.get('warnings', [])
        if warnings:
            elements.append(Paragraph(f"⚠️ <b>Warnings ({len(warnings)}):</b>", h3_style))
            for warning in warnings:
                elements.append(Paragraph(f"• {warning}", bullet_style))
            elements.append(Spacer(1, 0.1*inch))
        
        missing = safety.get('missing_evaluations', [])
        if missing:
            elements.append(Paragraph(f"📋 <b>Missing Evaluations ({len(missing)}):</b>", h3_style))
            for m in missing:
                elements.append(Paragraph(f"• {m}", bullet_style))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=6,
            backColor=colors.HexColor('#f9fafb')
        )
        footer_text = f"Generated: {report.get('timestamp', 'Unknown')} | Report ID: {report.get('report_id', 'N/A')}"
        elements.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"clinical_report_{report.get('report_id', 'unknown')}.pdf"
        )
    
    except Exception as e:
        import traceback
        print(f"PDF Error: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"PDF generation failed: {str(e)}"
        }), 500


@app.route('/api/export-json', methods=['POST'])
def export_json():
    """Export report as JSON file."""
    try:
        data = request.get_json()
        report = data.get('report', {})
        
        from flask import send_file
        from io import BytesIO
        
        json_bytes = json.dumps(report, indent=2).encode('utf-8')
        buffer = BytesIO(json_bytes)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"clinical_report_{report.get('report_id', 'unknown')}.json"
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"JSON export failed: {str(e)}"
        }), 500


@app.route('/api/system-status', methods=['GET'])
def system_status():
    """Get system status and available components."""
    try:
        orch = init_orchestrator()
        
        status = {
            "status": "ready",
            "components": {
                "ingestion_agent": orch.ingestion_agent is not None,
                "rag_system": orch.rag_system is not None,
                "gemini_api": True
            },
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏥 DOCTOR COPILOT - FLASK API SERVER")
    print("="*80)
    print("\nStarting Flask development server...")
    print("API will be available at: http://localhost:5000")
    print("Frontend will be available at: http://localhost:3000")
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /api/analyze         - Analyze patient data")
    print("  POST /api/export-pdf      - Export report as PDF")
    print("  POST /api/export-json     - Export report as JSON")
    print("  GET  /api/system-status   - System status")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)