"""
Generate Interim Report for Insurance Risk Analytics Project
Covers Task 1 (EDA) and Task 2 (DVC Setup)
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from datetime import datetime
from pathlib import Path
import os


def create_interim_report(output_path="outputs/reports/interim_report.pdf"):
    """Create comprehensive interim report PDF"""
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3949ab'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=8
    )
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Interim Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Insurance Risk Analytics & Predictive Modeling", 
                          ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                                       fontSize=16, alignment=TA_CENTER,
                                       textColor=colors.HexColor('#5c6bc0'))))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("AlphaCare Insurance Solutions (ACIS)", 
                          ParagraphStyle('Company', parent=styles['Normal'], 
                                       fontSize=14, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('Date', parent=styles['Normal'], 
                                       fontSize=12, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Covering Task 1 (EDA) and Task 2 (DVC Setup)", 
                          ParagraphStyle('Coverage', parent=styles['Normal'], 
                                       fontSize=12, alignment=TA_CENTER,
                                       fontStyle='italic')))
    
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading1_style))
    story.append(Spacer(1, 0.2*inch))
    
    toc_items = [
        "1. Executive Summary",
        "2. Understanding and Defining the Business Objective",
        "3. Task 1: Exploratory Data Analysis (EDA)",
        "4. Task 2: Data Version Control (DVC) Setup",
        "5. Key Findings and Insights",
        "6. Next Steps and Key Areas of Focus",
        "7. Conclusion"
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    exec_summary = """
    This interim report presents the progress made on the Insurance Risk Analytics and Predictive 
    Modeling project for AlphaCare Insurance Solutions (ACIS). The project aims to analyze 
    historical insurance claim data to identify low-risk segments and optimize premium pricing 
    strategies. This report covers two completed tasks: (1) comprehensive Exploratory Data 
    Analysis (EDA) of insurance data spanning February 2014 to August 2015, and (2) establishment 
    of a Data Version Control (DVC) system for reproducible and auditable data pipelines.
    
    Key highlights include the identification of significant loss ratio variations across provinces, 
    vehicle types, and demographic segments. The overall portfolio loss ratio of 104.77% indicates 
    that claims exceed premiums, highlighting the critical need for risk-based premium optimization. 
    The DVC infrastructure ensures regulatory compliance and reproducibility, essential for 
    financial services analytics.
    """
    story.append(Paragraph(exec_summary, body_style))
    story.append(PageBreak())
    
    # 2. Understanding and Defining the Business Objective
    story.append(Paragraph("2. Understanding and Defining the Business Objective", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("2.1 Business Context", heading2_style))
    business_context = """
    AlphaCare Insurance Solutions operates in the competitive South African car insurance market. 
    The company's strategic objective is to develop data-driven risk analytics capabilities that 
    enable more precise premium pricing and targeted marketing strategies. In an industry where 
    profitability depends on accurately assessing and pricing risk, the ability to identify 
    low-risk customer segments presents a significant competitive advantage.
    """
    story.append(Paragraph(business_context, body_style))
    
    story.append(Paragraph("2.2 Primary Objectives", heading2_style))
    objectives = [
        "Discover 'low-risk' target segments for premium reduction opportunities",
        "Build predictive models for optimal premium pricing based on risk factors",
        "Optimize marketing strategies by targeting profitable customer segments",
        "Ensure regulatory compliance through auditable and reproducible analytics"
    ]
    
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("2.3 Success Metrics", heading2_style))
    metrics = [
        "Loss Ratio (TotalClaims / TotalPremium) by segment",
        "Risk differentiation across geographic regions (Provinces, Postal Codes)",
        "Risk patterns by vehicle characteristics (Type, Make, Model)",
        "Demographic risk factors (Gender, Marital Status)",
        "Temporal trends in claims frequency and severity"
    ]
    
    for metric in metrics:
        story.append(Paragraph(f"• {metric}", bullet_style))
    
    story.append(PageBreak())
    
    # 3. Task 1: Exploratory Data Analysis
    story.append(Paragraph("3. Task 1: Exploratory Data Analysis (EDA)", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("3.1 Data Overview", heading2_style))
    data_overview = """
    The analysis utilized historical insurance data from February 2014 to August 2015, 
    containing 1,000,098 transaction records across 52 original features. The dataset includes 
    comprehensive information about policies, clients, vehicle characteristics, geographic 
    locations, and financial metrics (premiums and claims).
    """
    story.append(Paragraph(data_overview, body_style))
    
    story.append(Paragraph("3.2 Data Quality Assessment", heading2_style))
    quality_text = """
    Data quality assessment revealed several important characteristics:
    """
    story.append(Paragraph(quality_text, body_style))
    
    quality_findings = [
        "Missing values identified in 20+ columns, with highest missing rates in fleet-related fields (100%)",
        "CustomValueEstimate missing in 77.96% of records, requiring careful handling in modeling",
        "No duplicate rows detected, indicating clean data ingestion",
        "Outliers detected in TotalClaims (0.28%) and CustomValueEstimate (0.81%) using IQR method"
    ]
    
    for finding in quality_findings:
        story.append(Paragraph(f"• {finding}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("3.3 Key Financial Metrics", heading2_style))
    
    # Create a table for key metrics
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Premium', 'ZAR 61,911,562.70'],
        ['Total Claims', 'ZAR 64,867,546.17'],
        ['Overall Loss Ratio', '104.77%'],
        ['Data Period', 'Feb 2014 - Aug 2015'],
        ['Total Records', '1,000,098']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("3.4 Loss Ratio Analysis by Segment", heading2_style))
    
    story.append(Paragraph("3.4.1 By Province", heading2_style))
    province_text = """
    Significant variation in loss ratios across provinces was identified:
    """
    story.append(Paragraph(province_text, body_style))
    
    province_data = [
        ['Province', 'Loss Ratio', 'Risk Level'],
        ['Gauteng', '122.20%', 'High Risk'],
        ['KwaZulu-Natal', '108.27%', 'High Risk'],
        ['Western Cape', '105.95%', 'High Risk'],
        ['North West', '79.04%', 'Moderate Risk'],
        ['Mpumalanga', '72.09%', 'Low Risk'],
        ['Free State', '68.08%', 'Low Risk'],
        ['Limpopo', '66.12%', 'Low Risk'],
        ['Eastern Cape', '63.38%', 'Low Risk'],
        ['Northern Cape', '28.27%', 'Very Low Risk']
    ]
    
    province_table = Table(province_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    province_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(province_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("3.4.2 By Vehicle Type", heading2_style))
    vehicle_text = """
    Vehicle type analysis reveals distinct risk profiles:
    """
    story.append(Paragraph(vehicle_text, body_style))
    
    vehicle_findings = [
        "Heavy Commercial vehicles show highest loss ratio (162.81%) - critical risk segment",
        "Passenger Vehicles and Medium Commercial both exceed 100% loss ratio",
        "Light Commercial (23.21%) and Bus (13.73%) represent low-risk opportunities"
    ]
    
    for finding in vehicle_findings:
        story.append(Paragraph(f"• {finding}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("3.4.3 By Gender", heading2_style))
    gender_text = """
    Gender-based analysis indicates:
    """
    story.append(Paragraph(gender_text, body_style))
    
    gender_findings = [
        "Not specified category shows highest loss ratio (105.93%)",
        "Male drivers: 88.39% loss ratio",
        "Female drivers: 82.19% loss ratio - lowest risk segment"
    ]
    
    for finding in gender_findings:
        story.append(Paragraph(f"• {finding}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("3.5 Temporal Trends", heading2_style))
    temporal_text = """
    Analysis of monthly trends from February 2014 to August 2015 reveals:
    """
    story.append(Paragraph(temporal_text, body_style))
    
    temporal_findings = [
        "Loss ratio peaked in April 2015 at 139.64%",
        "Significant volatility in claims frequency over the 18-month period",
        "Premium collection remained relatively stable",
        "August 2015 shows unusually low loss ratio (14.01%) - requires investigation"
    ]
    
    for finding in temporal_findings:
        story.append(Paragraph(f"• {finding}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("3.6 Vehicle Make Analysis", heading2_style))
    make_text = """
    Top vehicle makes by total claims reveal significant risk concentration:
    """
    story.append(Paragraph(make_text, body_style))
    
    make_findings = [
        "TOYOTA: Highest total claims (ZAR 51.7M) with 103.60% loss ratio",
        "AUDI: 271.35% loss ratio - extremely high risk",
        "HYUNDAI: 398.98% loss ratio - critical risk segment",
        "MERCEDES-BENZ: 106.29% loss ratio with ZAR 2.9M in claims"
    ]
    
    for finding in make_findings:
        story.append(Paragraph(f"• {finding}", bullet_style))
    
    story.append(PageBreak())
    
    # 4. Task 2: DVC Setup
    story.append(Paragraph("4. Task 2: Data Version Control (DVC) Setup", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("4.1 Objective", heading2_style))
    dvc_objective = """
    Established a reproducible and auditable data pipeline using Data Version Control (DVC) to 
    meet regulatory compliance requirements in the financial services industry. DVC ensures that 
    data inputs are as rigorously version-controlled as code, enabling complete reproducibility 
    of analyses and models for auditing, regulatory compliance, and debugging purposes.
    """
    story.append(Paragraph(dvc_objective, body_style))
    
    story.append(Paragraph("4.2 Implementation", heading2_style))
    dvc_implementation = """
    The DVC infrastructure was successfully implemented with the following components:
    """
    story.append(Paragraph(dvc_implementation, body_style))
    
    dvc_components = [
        "DVC repository initialized in project directory",
        "Local remote storage configured at './dvc_storage/'",
        "Primary data file (MachineLearningRating_v3.txt, ~529 MB) added to DVC tracking",
        "Data file metadata (.dvc file) committed to Git repository",
        "Actual data file stored in DVC remote storage, excluded from Git",
        "Configuration files (.dvc/config, .dvcignore) properly version-controlled"
    ]
    
    for component in dvc_components:
        story.append(Paragraph(f"• {component}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("4.3 Benefits", heading2_style))
    dvc_benefits = [
        "Reproducibility: Any analysis can be reproduced using exact data versions",
        "Auditability: Complete history of data changes for regulatory compliance",
        "Storage Efficiency: Large files stored outside Git repository",
        "Version Control: Track different versions of datasets over time",
        "Collaboration: Team members can pull specific data versions as needed"
    ]
    
    for benefit in dvc_benefits:
        story.append(Paragraph(f"• {benefit}", bullet_style))
    
    story.append(PageBreak())
    
    # 5. Key Findings and Insights
    story.append(Paragraph("5. Key Findings and Insights", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("5.1 Critical Business Insights", heading2_style))
    critical_insights = [
        "Overall portfolio is unprofitable with 104.77% loss ratio - immediate action required",
        "Geographic segmentation reveals 3x risk difference between highest (Gauteng: 122.20%) and lowest (Northern Cape: 28.27%) risk provinces",
        "Vehicle type segmentation shows 12x risk difference (Heavy Commercial: 162.81% vs Bus: 13.73%)",
        "Female drivers represent lower risk segment (82.19% vs 88.39% for males)",
        "Specific vehicle makes (AUDI, HYUNDAI) show extreme risk profiles requiring targeted pricing"
    ]
    
    for insight in critical_insights:
        story.append(Paragraph(f"• {insight}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("5.2 Low-Risk Opportunities", heading2_style))
    opportunities = [
        "Northern Cape, Eastern Cape, Limpopo provinces - premium reduction opportunities",
        "Bus and Light Commercial vehicle types - expand market share",
        "Female driver segment - targeted marketing campaigns",
        "Specific vehicle makes with low loss ratios - competitive pricing strategies"
    ]
    
    for opp in opportunities:
        story.append(Paragraph(f"• {opp}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("5.3 High-Risk Segments Requiring Action", heading2_style))
    high_risk = [
        "Gauteng, KwaZulu-Natal, Western Cape provinces - premium adjustments needed",
        "Heavy Commercial vehicles - risk-based pricing implementation",
        "AUDI and HYUNDAI vehicle makes - underwriting review required",
        "Not specified gender category - data quality improvement needed"
    ]
    
    for risk in high_risk:
        story.append(Paragraph(f"• {risk}", bullet_style))
    
    story.append(PageBreak())
    
    # 6. Next Steps and Key Areas of Focus
    story.append(Paragraph("6. Next Steps and Key Areas of Focus", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("6.1 Immediate Next Steps", heading2_style))
    immediate_steps = [
        "A/B Hypothesis Testing: Validate risk differences across provinces, zipcodes, and gender",
        "Statistical Modeling: Develop linear regression models per zipcode to predict total claims",
        "Machine Learning Pipeline: Build predictive model for optimal premium pricing",
        "Feature Engineering: Create derived features from existing variables",
        "Model Evaluation: Assess model performance and feature importance"
    ]
    
    for step in immediate_steps:
        story.append(Paragraph(f"• {step}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("6.2 Hypothesis Testing Priorities", heading2_style))
    hypotheses = [
        "H0: No risk differences across provinces → Reject based on EDA findings",
        "H0: No risk differences between zipcodes → Requires statistical validation",
        "H0: No margin (profit) differences between zip codes → Critical for pricing strategy",
        "H0: No significant risk difference between Women and Men → Preliminary evidence suggests difference exists"
    ]
    
    for hyp in hypotheses:
        story.append(Paragraph(f"• {hyp}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("6.3 Modeling Priorities", heading2_style))
    modeling = [
        "Linear Regression per Zipcode: Predict TotalClaims using local risk factors",
        "Premium Optimization Model: ML model incorporating car features, owner demographics, location, and other relevant factors",
        "Feature Importance Analysis: Identify key drivers of risk and profitability",
        "Model Interpretability: Ensure models are explainable for regulatory compliance"
    ]
    
    for model in modeling:
        story.append(Paragraph(f"• {model}", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("6.4 Data Pipeline Enhancements", heading2_style))
    pipeline = [
        "Implement data validation checks in DVC pipeline",
        "Create automated EDA reports for new data versions",
        "Set up data quality monitoring dashboards",
        "Establish data versioning best practices documentation"
    ]
    
    for item in pipeline:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    story.append(PageBreak())
    
    # 7. Conclusion
    story.append(Paragraph("7. Conclusion", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    conclusion = """
    This interim report demonstrates significant progress in establishing the foundation for 
    data-driven risk analytics at AlphaCare Insurance Solutions. The comprehensive EDA has 
    revealed critical insights into risk segmentation across geographic, vehicle, and demographic 
    dimensions. The identification of low-risk segments (Northern Cape, Bus vehicles, Female 
    drivers) presents immediate opportunities for premium optimization and market expansion.
    
    The establishment of DVC infrastructure ensures that all future analyses will be reproducible 
    and auditable, meeting the stringent requirements of financial services regulation. This 
    foundation enables confident progression to hypothesis testing and predictive modeling phases.
    
    The findings from this analysis provide a clear roadmap for risk-based pricing strategies 
    and targeted marketing initiatives. The next phase of work will focus on statistical 
    validation of these insights and development of predictive models for premium optimization.
    
    The project is on track to deliver actionable recommendations that will enable ACIS to 
    improve profitability through data-driven risk assessment and premium pricing optimization.
    """
    story.append(Paragraph(conclusion, body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Appendices note
    story.append(Paragraph("Appendix", heading2_style))
    appendix_note = """
    Detailed visualizations and analysis outputs are available in the outputs/figures/ directory. 
    Key visualizations include loss ratio heatmaps, risk-return scatter plots, temporal evolution 
    dashboards, and distribution analyses. All code and analysis notebooks are version-controlled 
    in the project repository.
    """
    story.append(Paragraph(appendix_note, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"Interim report generated successfully: {output_path}")


if __name__ == "__main__":
    create_interim_report()

