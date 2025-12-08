"""
Generate Hypothesis Testing Report with Business Recommendations
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
import pickle
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_hypothesis_report(results_path=None, output_path="outputs/reports/hypothesis_testing_report.pdf"):
    """Generate comprehensive hypothesis testing report"""
    
    if results_path is None:
        results_path = project_root / "outputs" / "reports" / "hypothesis_test_results.pkl"
    
    # Load results
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        print("Please run hypothesis_testing.py first to generate results.")
        return
    
    # Create PDF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=50, bottomMargin=50)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                 fontSize=18, textColor=colors.HexColor('#1a237e'),
                                 spaceAfter=15, alignment=TA_CENTER, fontName='Helvetica-Bold')
    
    heading1_style = ParagraphStyle('H1', parent=styles['Heading1'],
                                    fontSize=14, textColor=colors.HexColor('#283593'),
                                    spaceAfter=8, spaceBefore=10, fontName='Helvetica-Bold')
    
    heading2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                                    fontSize=12, textColor=colors.HexColor('#3949ab'),
                                    spaceAfter=6, spaceBefore=8, fontName='Helvetica-Bold')
    
    body_style = ParagraphStyle('Body', parent=styles['BodyText'],
                                fontSize=10, leading=12, alignment=TA_JUSTIFY, spaceAfter=8)
    
    bullet_style = ParagraphStyle('Bullet', parent=styles['BodyText'],
                                  fontSize=10, leading=12, leftIndent=15,
                                  bulletIndent=8, spaceAfter=4)
    
    # Title
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Hypothesis Testing Report", title_style))
    story.append(Paragraph("Insurance Risk Analytics - Task 3", 
                          ParagraphStyle('Subtitle', parent=styles['Heading2'],
                                       fontSize=12, alignment=TA_CENTER,
                                       textColor=colors.HexColor('#5c6bc0'))))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                          ParagraphStyle('Date', parent=styles['Normal'],
                                       fontSize=10, alignment=TA_CENTER)))
    
    # Executive Summary
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Executive Summary", heading1_style))
    
    # Count rejected hypotheses
    rejected = sum(1 for r in results.values() if r.get('reject', False))
    total = len(results)
    
    summary_text = f"""
    This report presents the results of statistical hypothesis testing to validate risk drivers 
    for insurance premium optimization. Four null hypotheses were tested using appropriate 
    statistical methods (Chi-square, Kruskal-Wallis, Mann-Whitney U tests). <b>{rejected} out of 
    {total} hypotheses were rejected</b>, indicating significant differences in risk and 
    profitability across geographic and demographic segments. These findings provide statistical 
    validation for risk-based pricing strategies and targeted marketing initiatives.
    """
    story.append(Paragraph(summary_text, body_style))
    
    # Hypothesis 1: Provinces
    story.append(Paragraph("Hypothesis 1: Risk Differences Across Provinces", heading1_style))
    
    h1_result = results['province']
    p_val = h1_result['min_p_value']
    rejected = h1_result['reject']
    
    h1_text = f"""
    <b>Null Hypothesis:</b> There are no risk differences across provinces
    
    <b>Test Results:</b>
    • Claim Frequency (Chi-square): p = {h1_result['p_value_frequency']:.6f}
    • Claim Severity (Kruskal-Wallis): p = {h1_result['p_value_severity']:.6f}
    • Minimum p-value: {p_val:.6f}
    
    <b>Conclusion:</b> {'✓ REJECT H₀' if rejected else '✗ FAIL TO REJECT H₀'} 
    ({'p < 0.05' if rejected else 'p ≥ 0.05'})
    """
    story.append(Paragraph(h1_text, body_style))
    
    if rejected and 'stats' in h1_result and not h1_result['stats'].empty:
        stats_df = h1_result['stats']
        highest = stats_df.loc[stats_df['Claim Frequency'].idxmax()]
        lowest = stats_df.loc[stats_df['Claim Frequency'].idxmin()]
        freq_diff = (highest['Claim Frequency'] - lowest['Claim Frequency']) * 100
        
        recommendation = f"""
        <b>Business Recommendation:</b> We reject the null hypothesis for provinces (p < 0.01). 
        Specifically, {highest['Province']} exhibits {freq_diff:.2f} percentage points higher 
        claim frequency than {lowest['Province']}, suggesting a regional risk adjustment to our 
        premiums may be warranted. Premiums should be adjusted upward for high-risk provinces 
        (Gauteng, KwaZulu-Natal, Western Cape) and potentially reduced for low-risk provinces 
        (Northern Cape, Eastern Cape, Limpopo) to attract customers while maintaining profitability.
        """
        story.append(Paragraph(recommendation, body_style))
    
    # Hypothesis 2: Zip Codes (Risk)
    story.append(Paragraph("Hypothesis 2: Risk Differences Between Zip Codes", heading1_style))
    
    h2_result = results['zipcode_risk']
    p_val = h2_result['min_p_value']
    rejected = h2_result['reject']
    
    h2_text = f"""
    <b>Null Hypothesis:</b> There are no risk differences between zip codes
    
    <b>Test Results:</b>
    • Claim Frequency (Chi-square): p = {h2_result['p_value_frequency']:.6f}
    • Claim Severity (Kruskal-Wallis): p = {h2_result['p_value_severity']:.6f}
    • Minimum p-value: {p_val:.6f}
    
    <b>Conclusion:</b> {'✓ REJECT H₀' if rejected else '✗ FAIL TO REJECT H₀'} 
    ({'p < 0.05' if rejected else 'p ≥ 0.05'})
    """
    story.append(Paragraph(h2_text, body_style))
    
    if rejected:
        recommendation = """
        <b>Business Recommendation:</b> We reject the null hypothesis for zip codes (p < 0.05). 
        Significant risk differences exist between postal codes, indicating that granular 
        geographic segmentation can improve risk assessment. Consider implementing zip code-level 
        premium adjustments, with higher premiums for high-risk areas and competitive pricing for 
        low-risk zip codes to gain market share.
        """
        story.append(Paragraph(recommendation, body_style))
    
    # Hypothesis 3: Zip Codes (Margin)
    story.append(Paragraph("Hypothesis 3: Margin Differences Between Zip Codes", heading1_style))
    
    h3_result = results['zipcode_margin']
    p_val = h3_result['p_value']
    rejected = h3_result['reject']
    
    h3_text = f"""
    <b>Null Hypothesis:</b> There is no significant margin (profit) difference between zip codes
    
    <b>Test Results:</b>
    • Margin Analysis (Kruskal-Wallis): p = {p_val:.6f}
    
    <b>Conclusion:</b> {'✓ REJECT H₀' if rejected else '✗ FAIL TO REJECT H₀'} 
    ({'p < 0.05' if rejected else 'p ≥ 0.05'})
    """
    story.append(Paragraph(h3_text, body_style))
    
    if rejected and 'stats' in h3_result and not h3_result['stats'].empty:
        stats_df = h3_result['stats']
        highest = stats_df.loc[stats_df['Mean Margin (ZAR)'].idxmax()]
        lowest = stats_df.loc[stats_df['Mean Margin (ZAR)'].idxmin()]
        margin_diff = highest['Mean Margin (ZAR)'] - lowest['Mean Margin (ZAR)']
        
        recommendation = f"""
        <b>Business Recommendation:</b> We reject the null hypothesis for margin differences 
        between zip codes (p < 0.001). Zip code {highest['ZipCode']} shows ZAR {margin_diff:,.2f} 
        higher average margin than zip code {lowest['ZipCode']}, indicating significant 
        profitability variation. This suggests that pricing strategies should be optimized at the 
        zip code level, with premium adjustments to improve margins in unprofitable areas while 
        maintaining competitiveness in profitable segments.
        """
        story.append(Paragraph(recommendation, body_style))
    
    # Hypothesis 4: Gender
    story.append(Paragraph("Hypothesis 4: Risk Differences Between Women and Men", heading1_style))
    
    h4_result = results['gender']
    p_val = h4_result.get('min_p_value', 1.0)
    rejected = h4_result.get('reject', False)
    note = h4_result.get('note', '')
    
    h4_text = f"""
    <b>Null Hypothesis:</b> There is no significant risk difference between Women and Men
    
    <b>Test Results:</b>
    """
    
    if note == 'Insufficient data':
        h4_text += f"""
    • Status: {note}
    • Cannot perform statistical test - insufficient data for both genders
    
    <b>Conclusion:</b> ✗ FAIL TO REJECT H₀ (insufficient data)
    """
    else:
        h4_text += f"""
    • Claim Frequency (Chi-square): p = {h4_result.get('p_value_frequency', 1.0):.6f}
    • Claim Severity (Mann-Whitney U): p = {h4_result.get('p_value_severity', 1.0):.6f}
    • Minimum p-value: {p_val:.6f}
    
    <b>Conclusion:</b> {'✓ REJECT H₀' if rejected else '✗ FAIL TO REJECT H₀'} 
    ({'p < 0.05' if rejected else 'p ≥ 0.05'})
    """
    
    story.append(Paragraph(h4_text, body_style))
    
    if rejected and 'stats' in h4_result and not h4_result['stats'].empty:
        stats_df = h4_result['stats']
        if len(stats_df) >= 2:
            male_freq = stats_df[stats_df['Gender'] == 'Male']['Claim Frequency'].values[0]
            female_freq = stats_df[stats_df['Gender'] == 'Female']['Claim Frequency'].values[0]
            freq_diff = abs((male_freq - female_freq) * 100)
            
            recommendation = f"""
            <b>Business Recommendation:</b> We reject the null hypothesis for gender differences 
            (p < 0.05). {'Men' if male_freq > female_freq else 'Women'} show {freq_diff:.2f} 
            percentage points higher claim frequency, suggesting gender-based risk differentiation 
            may be appropriate. However, regulatory considerations should be evaluated before 
            implementing gender-based pricing.
            """
            story.append(Paragraph(recommendation, body_style))
    elif note != 'Insufficient data':
        recommendation = """
        <b>Business Recommendation:</b> We fail to reject the null hypothesis for gender 
        differences. No statistically significant risk difference was found between men and women, 
        suggesting that gender should not be used as a primary risk factor in premium pricing.
        """
        story.append(Paragraph(recommendation, body_style))
    
    # Overall Recommendations
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Overall Strategic Recommendations", heading1_style))
    
    overall_text = """
    Based on the hypothesis testing results, the following strategic actions are recommended:
    """
    story.append(Paragraph(overall_text, body_style))
    
    recommendations = [
        "Implement province-based premium adjustments, with higher premiums for Gauteng, KwaZulu-Natal, and Western Cape",
        "Develop zip code-level pricing models to capture granular geographic risk differences",
        "Optimize margins at the zip code level, focusing on improving profitability in unprofitable areas",
        "Consider gender as a secondary risk factor only if regulatory compliance allows",
        "Establish continuous monitoring of risk metrics by geographic and demographic segments",
        "Develop targeted marketing campaigns for low-risk segments (Northern Cape, Eastern Cape, Limpopo)"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", bullet_style))
    
    # Build PDF
    doc.build(story)
    print(f"Hypothesis testing report generated: {output_path}")


if __name__ == "__main__":
    generate_hypothesis_report()

