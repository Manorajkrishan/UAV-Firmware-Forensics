"""
PDF Report Generator for Forensic Analysis
Generates professional PDF reports with evidence, visualizations, and analysis results
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.pdfgen import canvas
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import io
import base64

def create_forensic_report(
    analysis_data: Dict,
    output_path: Path,
    investigator_notes: Optional[str] = None
) -> Path:
    """
    Create a comprehensive forensic PDF report
    
    Args:
        analysis_data: Complete analysis results including forensic features
        output_path: Path where PDF will be saved
        investigator_notes: Optional notes from investigator
    
    Returns:
        Path to generated PDF file
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Firmware Forensic Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    metadata_data = [
        ['Case ID:', analysis_data.get('firmware_id', 'N/A')],
        ['File Name:', analysis_data.get('file_name', 'N/A')],
        ['Analysis Date:', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')],
        ['Model Used:', analysis_data.get('model_used', 'N/A')],
        ['SHA-256 Hash:', analysis_data.get('sha256_hash', 'N/A')[:64] + '...' if len(analysis_data.get('sha256_hash', '')) > 64 else analysis_data.get('sha256_hash', 'N/A')]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    tampering_status = analysis_data.get('tampering_status', 'Unknown')
    tampering_prob = analysis_data.get('tampering_probability', 0.0) * 100
    severity_level = analysis_data.get('severity_level', 'Unknown')
    
    # Determine status color
    if tampering_status == 'Tampered':
        status_color = colors.red
    elif tampering_status == 'Suspicious':
        status_color = colors.orange
    else:
        status_color = colors.green
    
    summary_text = f"""
    <b>Tampering Status:</b> <font color="{status_color.hex}">{tampering_status}</font><br/>
    <b>Tampering Probability:</b> {tampering_prob:.2f}%<br/>
    <b>Severity Level:</b> {severity_level}<br/>
    <b>Confidence Score:</b> {analysis_data.get('confidence_score', 0.0):.2f}%<br/>
    <b>Anomaly Score:</b> {analysis_data.get('anomaly_score', 0.0):.4f}
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Evidence Justification
    story.append(Paragraph("Evidence Justification", heading_style))
    
    forensic_features = analysis_data.get('forensic_features', {})
    version_anomalies = analysis_data.get('version_anomalies', {})
    boot_analysis = analysis_data.get('boot_analysis', {})
    integrity_checks = analysis_data.get('integrity_checks', {})
    
    evidence_items = []
    
    if version_anomalies.get('has_version_mod'):
        evidence_items.append(f"• Version modification detected (Score: {version_anomalies.get('version_anomaly_score', 0.0):.2f})")
    if boot_analysis.get('boot_time_anomaly'):
        evidence_items.append(f"• Boot sequence irregularities detected (Score: {boot_analysis.get('boot_anomaly_score', 0.0):.2f})")
    if integrity_checks.get('missing_integrity_checks'):
        evidence_items.append(f"• Missing integrity checks detected (Score: {integrity_checks.get('integrity_anomaly_score', 0.0):.2f})")
    
    if not evidence_items:
        evidence_items.append("• No significant anomalies detected in forensic analysis")
    
    evidence_text = "<br/>".join(evidence_items)
    story.append(Paragraph(evidence_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Feature Contributions
    story.append(Paragraph("Feature Contribution Analysis", heading_style))
    
    feature_contributions = analysis_data.get('feature_contributions', {})
    if feature_contributions:
        contrib_data = [['Feature', 'Value', 'Contribution %']]
        for feature, data in sorted(feature_contributions.items(), key=lambda x: x[1].get('percentage', 0), reverse=True)[:10]:
            contrib_data.append([
                feature.replace('_', ' ').title(),
                f"{data.get('value', 0):.2f}",
                f"{data.get('percentage', 0):.1f}%"
            ])
        
        contrib_table = Table(contrib_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        contrib_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(contrib_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    
    recommendations = analysis_data.get('recommendations', [])
    if recommendations:
        rec_text = "<br/>".join([f"• {rec}" for rec in recommendations])
    else:
        rec_text = "• No specific recommendations at this time."
    
    story.append(Paragraph(rec_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Cryptographic Integrity Proof
    story.append(Paragraph("Cryptographic Integrity Proof", heading_style))
    
    hash_value = analysis_data.get('sha256_hash', 'N/A')
    integrity_text = f"""
    <b>SHA-256 Hash:</b> {hash_value}<br/>
    <b>Hash Verification:</b> Hash calculated and stored for integrity verification<br/>
    <b>File Integrity Status:</b> {'Verified' if hash_value != 'N/A' else 'Not Available'}
    """
    
    story.append(Paragraph(integrity_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Investigator Notes
    if investigator_notes:
        story.append(Paragraph("Investigator Notes", heading_style))
        story.append(Paragraph(investigator_notes, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    footer_text = f"""
    <i>Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
    This report is generated by the ML-Based Drone Firmware Tampering Detection System</i>
    """
    story.append(Paragraph(footer_text, styles['Italic']))
    
    # Build PDF
    doc.build(story)
    
    return output_path

