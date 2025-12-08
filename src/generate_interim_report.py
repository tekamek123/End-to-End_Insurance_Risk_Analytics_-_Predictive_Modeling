"""
Generate Interim Report for Insurance Risk Analytics Project
Covers Task 1 (EDA) and Task 2 (DVC Setup) - Enhanced version with visualizations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from datetime import datetime
from pathlib import Path
import os


def create_interim_report(output_path="outputs/reports/interim_report.pdf"):
    """Create enhanced 3-4 page interim report PDF with visualizations"""
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=50, bottomMargin=50)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles - more compact
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#283593'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#3949ab'),
        spaceAfter=6,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        leftIndent=15,
        bulletIndent=8,
        spaceAfter=4
    )
    
    # Title Page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Interim Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Insurance Risk Analytics & Predictive Modeling", 
                          ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                                       fontSize=14, alignment=TA_CENTER,
                                       textColor=colors.HexColor('#5c6bc0'))))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("AlphaCare Insurance Solutions (ACIS)", 
                          ParagraphStyle('Company', parent=styles['Normal'], 
                                       fontSize=12, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('Date', parent=styles['Normal'], 
                                       fontSize=11, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Covering Task 1 (EDA) and Task 2 (DVC Setup)", 
                          ParagraphStyle('Coverage', parent=styles['Normal'], 
                                       fontSize=10, alignment=TA_CENTER,
                                       fontStyle='italic')))
    
    # 1. Executive Summary
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("1. Executive Summary", heading1_style))
    
    exec_summary = """
    This interim report presents progress on the Insurance Risk Analytics project for ACIS, 
    covering comprehensive Exploratory Data Analysis (EDA) and Data Version Control (DVC) 
    infrastructure setup. Analysis of 1,000,098 insurance records (Feb 2014 - Aug 2015) 
    reveals critical risk segmentation opportunities. The overall portfolio loss ratio of 
    104.77% indicates claims exceed premiums, requiring immediate risk-based pricing optimization. 
    DVC infrastructure ensures reproducible and auditable data pipelines for regulatory compliance.
    """
    story.append(Paragraph(exec_summary, body_style))
    
    # 2. Business Objective
    story.append(Paragraph("2. Business Objective", heading1_style))
    
    objective_text = """
    <b>Primary Goal:</b> Identify low-risk customer segments for premium reduction opportunities 
    and develop predictive models for optimal premium pricing. <b>Key Objectives:</b> (1) Discover 
    low-risk targets for marketing, (2) Build ML models for premium optimization, (3) Support 
    data-driven marketing strategies, (4) Ensure regulatory compliance through auditable analytics.
    """
    story.append(Paragraph(objective_text, body_style))
    
    # 3. Task 1: EDA Findings
    story.append(Paragraph("3. Task 1: Exploratory Data Analysis - Key Findings", heading1_style))
    
    story.append(Paragraph("3.1 Data Quality Assessment", heading2_style))
    
    quality_text = """
    Comprehensive data quality analysis was conducted to ensure reliable insights. The dataset 
    contains 1,000,098 records with 52 original features. Data quality assessment revealed:
    """
    story.append(Paragraph(quality_text, body_style))
    
    story.append(Paragraph("3.1.1 Missing Value Analysis", heading2_style))
    
    missing_data = [
        ['Column', 'Missing Count', 'Missing %'],
        ['NumberOfVehiclesInFleet', '1,000,098', '100.00%'],
        ['CrossBorder', '999,400', '99.93%'],
        ['CustomValueEstimate', '779,642', '77.96%'],
        ['Converted', '641,901', '64.18%'],
        ['Rebuilt', '641,901', '64.18%'],
        ['WrittenOff', '641,901', '64.18%'],
        ['LossRatio', '381,634', '38.16%'],
        ['NewVehicle', '153,295', '15.33%'],
        ['Bank', '145,961', '14.59%'],
        ['AccountType', '40,232', '4.02%']
    ]
    
    missing_table = Table(missing_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    missing_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(missing_table)
    story.append(Spacer(1, 0.1*inch))
    
    missing_analysis = """
    <b>Key Findings:</b> Fleet-related fields show 100% missing values, indicating these features 
    are not applicable to most policies. CustomValueEstimate missing in 77.96% of records requires 
    careful handling in modeling. LossRatio missing in 38.16% of records (where TotalPremium = 0). 
    No duplicate rows detected, indicating clean data ingestion. Missing values in demographic fields 
    (Bank: 14.59%, AccountType: 4.02%) are manageable and can be handled through imputation or 
    exclusion strategies.
    """
    story.append(Paragraph(missing_analysis, body_style))
    
    story.append(Paragraph("3.1.2 Outlier Detection", heading2_style))
    
    outlier_text = """
    Outlier detection using Interquartile Range (IQR) method with 1.5x multiplier identified:
    """
    story.append(Paragraph(outlier_text, body_style))
    
    outlier_findings = [
        "TotalClaims: 2,793 outliers (0.28% of records) - extreme claim values requiring investigation",
        "CustomValueEstimate: 1,785 outliers (0.81% of non-missing records) - high-value vehicles",
        "TotalPremium: Negative values detected (min: -782.58) - data quality issue requiring correction",
        "SumInsured: Wide distribution with high variability (CV: 249.65%)"
    ]
    
    for finding in outlier_findings:
        story.append(Paragraph(f"• {finding}", bullet_style))
    
    # Add outlier visualization if available
    outlier_img_path = "outputs/figures/outlier_detection_boxplots.png"
    if os.path.exists(outlier_img_path):
        story.append(Spacer(1, 0.1*inch))
        img = Image(outlier_img_path, width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<i>Figure 1: Box plots showing outlier detection for key financial variables</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, 
                                           alignment=TA_CENTER, fontStyle='italic')))
    
    story.append(Paragraph("3.2 Overall Portfolio Metrics", heading2_style))
    
    # Compact metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Premium', 'ZAR 61.9M'],
        ['Total Claims', 'ZAR 64.9M'],
        ['Loss Ratio', '104.77%'],
        ['Records Analyzed', '1,000,098'],
        ['Period', 'Feb 2014 - Aug 2015']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWPADDING', (0, 1), (-1, -1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("3.2.1 Descriptive Statistics for Key Financial Variables", heading2_style))
    
    desc_stats_text = """
    Comprehensive descriptive statistics for the primary financial variables reveal the distribution 
    characteristics and variability in the dataset:
    """
    story.append(Paragraph(desc_stats_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Descriptive statistics table
    desc_stats_data = [
        ['Statistic', 'TotalPremium (ZAR)', 'TotalClaims (ZAR)'],
        ['Mean (Average)', '61.91', '64.86'],
        ['Median', '2.18', '0.00'],
        ['Standard Deviation', '230.28', '2,384.07'],
        ['Minimum', '-782.58', '-12,002.41'],
        ['Maximum', '65,282.60', '393,092.11'],
        ['25th Percentile (Q1)', '0.00', '0.00'],
        ['75th Percentile (Q3)', '21.93', '0.00'],
        ['Coefficient of Variation', '371.99%', '3,675.66%']
    ]
    
    desc_stats_table = Table(desc_stats_data, colWidths=[2*inch, 2*inch, 2*inch])
    desc_stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWPADDING', (0, 1), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(desc_stats_table)
    story.append(Spacer(1, 0.1*inch))
    
    stats_interpretation = """
    <b>Key Observations:</b> (1) <b>High Variability:</b> TotalClaims shows extreme variability (CV: 3,675.66%) 
    compared to TotalPremium (CV: 371.99%), indicating highly skewed distributions. (2) <b>Median vs Mean:</b> 
    Median TotalPremium (ZAR 2.18) is much lower than mean (ZAR 61.91), indicating right-skewed distribution 
    with high-value outliers. (3) <b>Zero Claims:</b> Median TotalClaims is ZAR 0.00, with 75% of records 
    having zero claims, indicating most policies have no claims. (4) <b>Data Quality Issues:</b> Negative 
    minimum values (-782.58 for Premium, -12,002.41 for Claims) indicate data quality issues requiring 
    correction. (5) <b>Outliers:</b> Maximum values (ZAR 65,282.60 for Premium, ZAR 393,092.11 for Claims) 
    represent extreme cases requiring investigation.
    """
    story.append(Paragraph(stats_interpretation, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("3.3 Loss Ratio by Segment", heading2_style))
    
    # Add correlation matrix visualization
    corr_img_path = "outputs/figures/correlation_matrix.png"
    if os.path.exists(corr_img_path):
        img = Image(corr_img_path, width=4.5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("<i>Figure 2: Correlation matrix of key financial variables</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, 
                                           alignment=TA_CENTER, fontStyle='italic')))
        story.append(Spacer(1, 0.1*inch))
    
    # Compact province table
    province_data = [
        ['Province', 'Loss Ratio', 'Risk'],
        ['Gauteng', '122.20%', 'High'],
        ['KwaZulu-Natal', '108.27%', 'High'],
        ['Western Cape', '105.95%', 'High'],
        ['North West', '79.04%', 'Moderate'],
        ['Mpumalanga', '72.09%', 'Low'],
        ['Free State', '68.08%', 'Low'],
        ['Limpopo', '66.12%', 'Low'],
        ['Eastern Cape', '63.38%', 'Low'],
        ['Northern Cape', '28.27%', 'Very Low']
    ]
    
    province_table = Table(province_data, colWidths=[2*inch, 1.2*inch, 1.3*inch])
    province_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(province_table)
    story.append(Spacer(1, 0.1*inch))
    
    # Add loss ratio heatmap visualization
    heatmap_img_path = "outputs/figures/creative_viz1_loss_ratio_heatmap.png"
    if os.path.exists(heatmap_img_path):
        story.append(Spacer(1, 0.1*inch))
        img = Image(heatmap_img_path, width=5.5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("<i>Figure 3: Loss ratio heatmap by Province and Vehicle Type (lower is better)</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, 
                                           alignment=TA_CENTER, fontStyle='italic')))
        story.append(Spacer(1, 0.1*inch))
    
    findings_text = """
    <b>Key Insights:</b> (1) <b>Geographic Risk:</b> 4.3x difference between highest (Gauteng: 122.20%) 
    and lowest (Northern Cape: 28.27%) risk provinces. (2) <b>Vehicle Type:</b> Heavy Commercial shows 
    162.81% loss ratio vs Bus at 13.73% (12x difference). (3) <b>Gender:</b> Female drivers (82.19%) 
    lower risk than males (88.39%). (4) <b>Vehicle Makes:</b> AUDI (271%) and HYUNDAI (399%) show 
    extreme risk requiring immediate attention.
    """
    story.append(Paragraph(findings_text, body_style))
    
    story.append(Paragraph("3.4 Temporal Trends", heading2_style))
    
    # Add temporal visualization
    temporal_img_path = "outputs/figures/creative_viz3_temporal_evolution.png"
    if os.path.exists(temporal_img_path):
        img = Image(temporal_img_path, width=5.5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("<i>Figure 4: Temporal evolution of key insurance metrics (Feb 2014 - Aug 2015)</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, 
                                           alignment=TA_CENTER, fontStyle='italic')))
        story.append(Spacer(1, 0.1*inch))
    
    temporal_text = """
    Monthly analysis reveals significant volatility: Loss ratio peaked at 139.64% in April 2015, 
    with August 2015 showing unusually low 14.01% (requires investigation). Premium collection 
    remained stable while claims showed high variability. The temporal dashboard (Figure 4) shows 
    clear patterns in policy count, premium, claims, and loss ratio trends over the 18-month period.
    """
    story.append(Paragraph(temporal_text, body_style))
    
    # 4. Task 2: DVC Setup
    story.append(Paragraph("4. Task 2: Data Version Control (DVC) Setup", heading1_style))
    
    dvc_text = """
    <b>Implementation:</b> DVC repository initialized with local remote storage at './dvc_storage/'. 
    Primary data file (MachineLearningRating_v3.txt, ~529 MB) added to DVC tracking. Data metadata 
    committed to Git while actual data stored in DVC remote. <b>Benefits:</b> (1) Complete 
    reproducibility for regulatory audits, (2) Version control for datasets, (3) Storage efficiency 
    (large files outside Git), (4) Team collaboration with specific data versions.
    """
    story.append(Paragraph(dvc_text, body_style))
    
    # 5. Key Findings & Opportunities
    story.append(Paragraph("5. Key Findings and Opportunities", heading1_style))
    
    story.append(Paragraph("5.1 Low-Risk Opportunities", heading2_style))
    opportunities = [
        "Northern Cape, Eastern Cape, Limpopo provinces - premium reduction to attract customers",
        "Bus and Light Commercial vehicles - expand market share with competitive pricing",
        "Female driver segment - targeted marketing campaigns",
        "Specific low-risk vehicle makes - competitive pricing strategies"
    ]
    
    for opp in opportunities:
        story.append(Paragraph(f"• {opp}", bullet_style))
    
    story.append(Paragraph("5.2 High-Risk Segments Requiring Action", heading2_style))
    high_risk = [
        "Gauteng, KwaZulu-Natal, Western Cape - premium adjustments needed",
        "Heavy Commercial vehicles - implement risk-based pricing",
        "AUDI and HYUNDAI makes - underwriting review required"
    ]
    
    for risk in high_risk:
        story.append(Paragraph(f"• {risk}", bullet_style))
    
    # 6. Next Steps
    story.append(Paragraph("6. Next Steps and Focus Areas", heading1_style))
    
    next_steps = """
    <b>Immediate Priorities:</b> (1) <b>A/B Hypothesis Testing:</b> Validate risk differences across 
    provinces, zipcodes, and gender using statistical tests. (2) <b>Statistical Modeling:</b> Develop 
    linear regression per zipcode to predict total claims. (3) <b>ML Pipeline:</b> Build predictive 
    model for optimal premium pricing incorporating car features, owner demographics, location, and 
    other risk factors. (4) <b>Feature Engineering:</b> Create derived features and assess importance. 
    (5) <b>Model Evaluation:</b> Assess performance and ensure interpretability for regulatory compliance.
    """
    story.append(Paragraph(next_steps, body_style))
    
    story.append(Paragraph("6.1 Hypothesis Testing Priorities", heading2_style))
    hypotheses = [
        "H₀: No risk differences across provinces → Preliminary evidence suggests rejection",
        "H₀: No risk differences between zipcodes → Requires statistical validation",
        "H₀: No margin differences between zip codes → Critical for pricing strategy",
        "H₀: No risk difference between Women and Men → Evidence suggests difference (82.19% vs 88.39%)"
    ]
    
    for hyp in hypotheses:
        story.append(Paragraph(f"• {hyp}", bullet_style))
    
    # 7. Conclusion
    story.append(Paragraph("7. Conclusion", heading1_style))
    
    conclusion = """
    The EDA has revealed significant risk segmentation opportunities with 4.3x variation across provinces 
    and 12x variation across vehicle types. Data quality analysis identified manageable missing value 
    patterns and outliers requiring attention. The DVC infrastructure ensures reproducible, auditable 
    analytics meeting regulatory requirements. Low-risk segments (Northern Cape, Bus vehicles, Female 
    drivers) present immediate opportunities for premium optimization and market expansion. The project 
    is on track to deliver actionable recommendations for risk-based pricing and targeted marketing 
    strategies that will improve ACIS profitability through data-driven risk assessment.
    """
    story.append(Paragraph(conclusion, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"Enhanced interim report generated successfully: {output_path}")


if __name__ == "__main__":
    create_interim_report()
