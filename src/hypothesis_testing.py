"""
A/B Hypothesis Testing for Insurance Risk Analytics
Task 3: Statistically validate or reject key hypotheses about risk drivers
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, kruskal
from pathlib import Path
import warnings
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class HypothesisTester:
    """Class for conducting A/B hypothesis tests on insurance data"""
    
    def __init__(self, data_path=None, sample_size=None):
        """
        Initialize hypothesis tester with data loading
        
        Args:
            data_path: Path to data file (optional)
            sample_size: Sample size for faster testing (optional)
        """
        print("=" * 80)
        print("HYPOTHESIS TESTING FOR INSURANCE RISK ANALYTICS")
        print("=" * 80)
        
        # Load data
        if data_path is None:
            data_path = project_root / "data" / "MachineLearningRating_v3.txt"
        
        print(f"\nLoading data from: {data_path}")
        self.df = pd.read_csv(
            data_path,
            sep='|',
            low_memory=False,
            parse_dates=['TransactionMonth'],
            date_parser=lambda x: pd.to_datetime(x, errors='coerce'),
            nrows=sample_size
        )
        
        # Clean and prepare data
        self._prepare_data()
        
        print(f"Data loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        print(f"Date range: {self.df['TransactionMonth'].min()} to {self.df['TransactionMonth'].max()}")
    
    def _prepare_data(self):
        """Prepare data for hypothesis testing"""
        # Convert numeric columns
        numeric_cols = ['TotalPremium', 'TotalClaims', 'PostalCode']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create derived metrics
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        # Filter out invalid data
        self.df = self.df[self.df['TotalPremium'] >= 0]  # Remove negative premiums
        self.df = self.df[self.df['TotalClaims'] >= 0]    # Remove negative claims
        
        # Create claim severity (average claim amount when claim occurred)
        self.df['ClaimSeverity'] = np.where(
            self.df['HasClaim'] == 1,
            self.df['TotalClaims'],
            np.nan
        )
    
    def calculate_claim_frequency(self, group_col, group_value=None):
        """
        Calculate claim frequency for a group
        
        Args:
            group_col: Column to group by
            group_value: Specific value to filter (optional)
            
        Returns:
            Claim frequency (proportion with claims)
        """
        if group_value is not None:
            data = self.df[self.df[group_col] == group_value]
        else:
            data = self.df
        
        if len(data) == 0:
            return 0.0
        
        return data['HasClaim'].mean()
    
    def calculate_claim_severity(self, group_col, group_value=None):
        """
        Calculate claim severity for a group
        
        Args:
            group_col: Column to group by
            group_value: Specific value to filter (optional)
            
        Returns:
            Average claim severity (mean claim amount when claim occurred)
        """
        if group_value is not None:
            data = self.df[self.df[group_col] == group_value]
        else:
            data = self.df
        
        severity_data = data['ClaimSeverity'].dropna()
        if len(severity_data) == 0:
            return 0.0
        
        return severity_data.mean()
    
    def test_province_risk_differences(self):
        """
        Test H₀: There are no risk differences across provinces
        
        Uses:
        - Claim Frequency: Chi-square test
        - Claim Severity: Kruskal-Wallis test (non-parametric ANOVA)
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 1: Risk Differences Across Provinces")
        print("=" * 80)
        
        # Filter valid provinces
        valid_provinces = self.df['Province'].dropna().value_counts()
        valid_provinces = valid_provinces[valid_provinces >= 100]  # At least 100 records
        province_list = valid_provinces.index.tolist()
        
        print(f"\nTesting {len(province_list)} provinces with sufficient data:")
        print(f"Provinces: {', '.join(province_list)}")
        
        # Test 1: Claim Frequency (Chi-square test)
        print("\n--- Test 1: Claim Frequency (Chi-square test) ---")
        
        # Create contingency table
        contingency_data = []
        for province in province_list:
            province_data = self.df[self.df['Province'] == province]
            with_claims = province_data['HasClaim'].sum()
            without_claims = len(province_data) - with_claims
            contingency_data.append([with_claims, without_claims])
        
        contingency_table = pd.DataFrame(
            contingency_data,
            index=province_list,
            columns=['With Claims', 'Without Claims']
        )
        
        print("\nContingency Table:")
        print(contingency_table)
        
        # Chi-square test
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square statistic: {chi2:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"P-value: {p_value_freq:.6f}")
        print(f"Significance level: α = 0.05")
        
        # Test 2: Claim Severity (Kruskal-Wallis test)
        print("\n--- Test 2: Claim Severity (Kruskal-Wallis test) ---")
        
        severity_groups = []
        for province in province_list:
            province_data = self.df[self.df['Province'] == province]
            severity = province_data['ClaimSeverity'].dropna()
            if len(severity) > 0:
                severity_groups.append(severity.values)
        
        if len(severity_groups) >= 2:
            h_stat, p_value_sev = kruskal(*severity_groups)
            print(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}")
            print(f"P-value: {p_value_sev:.6f}")
        else:
            p_value_sev = 1.0
            print("\nInsufficient data for severity test")
        
        # Calculate summary statistics
        print("\n--- Summary Statistics by Province ---")
        province_stats = []
        for province in province_list:
            freq = self.calculate_claim_frequency('Province', province)
            sev = self.calculate_claim_severity('Province', province)
            count = len(self.df[self.df['Province'] == province])
            province_stats.append({
                'Province': province,
                'Claim Frequency': freq,
                'Claim Severity (ZAR)': sev,
                'Sample Size': count
            })
        
        stats_df = pd.DataFrame(province_stats)
        print(stats_df.to_string(index=False))
        
        # Interpretation
        print("\n--- Interpretation ---")
        min_p = min(p_value_freq, p_value_sev)
        
        if min_p < 0.05:
            print(f"✓ REJECT H₀: There ARE significant risk differences across provinces")
            print(f"  (p = {min_p:.6f} < 0.05)")
            
            # Find highest and lowest risk provinces
            highest_freq = stats_df.loc[stats_df['Claim Frequency'].idxmax()]
            lowest_freq = stats_df.loc[stats_df['Claim Frequency'].idxmin()]
            
            freq_diff = (highest_freq['Claim Frequency'] - lowest_freq['Claim Frequency']) * 100
            print(f"\n  Key Finding: {highest_freq['Province']} has {freq_diff:.2f} percentage points")
            print(f"  higher claim frequency than {lowest_freq['Province']}")
        else:
            print(f"✗ FAIL TO REJECT H₀: No significant risk differences across provinces")
            print(f"  (p = {min_p:.6f} >= 0.05)")
        
        return {
            'hypothesis': 'No risk differences across provinces',
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev,
            'min_p_value': min_p,
            'reject': min_p < 0.05,
            'stats': stats_df
        }
    
    def test_zipcode_risk_differences(self):
        """
        Test H₀: There are no risk differences between zip codes
        
        Uses:
        - Claim Frequency: Chi-square test on top zip codes
        - Claim Severity: Kruskal-Wallis test
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 2: Risk Differences Between Zip Codes")
        print("=" * 80)
        
        # Filter valid zip codes (at least 50 records)
        valid_zipcodes = self.df['PostalCode'].dropna().value_counts()
        valid_zipcodes = valid_zipcodes[valid_zipcodes >= 50]
        top_zipcodes = valid_zipcodes.head(10).index.tolist()  # Top 10 by count
        
        print(f"\nTesting top {len(top_zipcodes)} zip codes with sufficient data:")
        print(f"Zip codes: {top_zipcodes}")
        
        # Test 1: Claim Frequency
        print("\n--- Test 1: Claim Frequency (Chi-square test) ---")
        
        contingency_data = []
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df['PostalCode'] == zipcode]
            with_claims = zipcode_data['HasClaim'].sum()
            without_claims = len(zipcode_data) - with_claims
            contingency_data.append([with_claims, without_claims])
        
        contingency_table = pd.DataFrame(
            contingency_data,
            index=top_zipcodes,
            columns=['With Claims', 'Without Claims']
        )
        
        print("\nContingency Table (Top 10 Zip Codes):")
        print(contingency_table)
        
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square statistic: {chi2:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"P-value: {p_value_freq:.6f}")
        
        # Test 2: Claim Severity
        print("\n--- Test 2: Claim Severity (Kruskal-Wallis test) ---")
        
        severity_groups = []
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df['PostalCode'] == zipcode]
            severity = zipcode_data['ClaimSeverity'].dropna()
            if len(severity) > 0:
                severity_groups.append(severity.values)
        
        if len(severity_groups) >= 2:
            h_stat, p_value_sev = kruskal(*severity_groups)
            print(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}")
            print(f"P-value: {p_value_sev:.6f}")
        else:
            p_value_sev = 1.0
        
        # Summary statistics
        print("\n--- Summary Statistics by Zip Code (Top 10) ---")
        zipcode_stats = []
        for zipcode in top_zipcodes:
            freq = self.calculate_claim_frequency('PostalCode', zipcode)
            sev = self.calculate_claim_severity('PostalCode', zipcode)
            count = len(self.df[self.df['PostalCode'] == zipcode])
            zipcode_stats.append({
                'ZipCode': str(zipcode),
                'Claim Frequency': freq,
                'Claim Severity (ZAR)': sev,
                'Sample Size': count
            })
        
        stats_df = pd.DataFrame(zipcode_stats)
        print(stats_df.to_string(index=False))
        
        # Interpretation
        print("\n--- Interpretation ---")
        min_p = min(p_value_freq, p_value_sev)
        
        if min_p < 0.05:
            print(f"✓ REJECT H₀: There ARE significant risk differences between zip codes")
            print(f"  (p = {min_p:.6f} < 0.05)")
        else:
            print(f"✗ FAIL TO REJECT H₀: No significant risk differences between zip codes")
            print(f"  (p = {min_p:.6f} >= 0.05)")
        
        return {
            'hypothesis': 'No risk differences between zip codes',
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev,
            'min_p_value': min_p,
            'reject': min_p < 0.05,
            'stats': stats_df
        }
    
    def test_zipcode_margin_differences(self):
        """
        Test H₀: There is no significant margin (profit) difference between zip codes
        
        Uses:
        - Margin: Kruskal-Wallis test (non-parametric, handles non-normal distributions)
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 3: Margin (Profit) Differences Between Zip Codes")
        print("=" * 80)
        
        # Filter valid zip codes
        valid_zipcodes = self.df['PostalCode'].dropna().value_counts()
        valid_zipcodes = valid_zipcodes[valid_zipcodes >= 50]
        top_zipcodes = valid_zipcodes.head(10).index.tolist()
        
        print(f"\nTesting top {len(top_zipcodes)} zip codes:")
        
        # Kruskal-Wallis test for margin differences
        print("\n--- Test: Margin Differences (Kruskal-Wallis test) ---")
        
        margin_groups = []
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df['PostalCode'] == zipcode]
            margins = zipcode_data['Margin'].dropna()
            if len(margins) > 0:
                margin_groups.append(margins.values)
        
        if len(margin_groups) >= 2:
            h_stat, p_value = kruskal(*margin_groups)
            print(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}")
            print(f"P-value: {p_value:.6f}")
        else:
            p_value = 1.0
            print("\nInsufficient data for test")
        
        # Summary statistics
        print("\n--- Summary Statistics by Zip Code ---")
        zipcode_stats = []
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df['PostalCode'] == zipcode]
            avg_margin = zipcode_data['Margin'].mean()
            median_margin = zipcode_data['Margin'].median()
            count = len(zipcode_data)
            zipcode_stats.append({
                'ZipCode': str(zipcode),
                'Mean Margin (ZAR)': avg_margin,
                'Median Margin (ZAR)': median_margin,
                'Sample Size': count
            })
        
        stats_df = pd.DataFrame(zipcode_stats)
        print(stats_df.to_string(index=False))
        
        # Interpretation
        print("\n--- Interpretation ---")
        if p_value < 0.05:
            print(f"✓ REJECT H₀: There ARE significant margin differences between zip codes")
            print(f"  (p = {p_value:.6f} < 0.05)")
            
            highest = stats_df.loc[stats_df['Mean Margin (ZAR)'].idxmax()]
            lowest = stats_df.loc[stats_df['Mean Margin (ZAR)'].idxmin()]
            margin_diff = highest['Mean Margin (ZAR)'] - lowest['Mean Margin (ZAR)']
            
            print(f"\n  Key Finding: Zip code {highest['ZipCode']} has ZAR {margin_diff:,.2f}")
            print(f"  higher average margin than zip code {lowest['ZipCode']}")
        else:
            print(f"✗ FAIL TO REJECT H₀: No significant margin differences between zip codes")
            print(f"  (p = {p_value:.6f} >= 0.05)")
        
        return {
            'hypothesis': 'No margin difference between zip codes',
            'p_value': p_value,
            'reject': p_value < 0.05,
            'stats': stats_df
        }
    
    def test_gender_risk_differences(self):
        """
        Test H₀: There is no significant risk difference between Women and Men
        
        Uses:
        - Claim Frequency: Chi-square test
        - Claim Severity: Mann-Whitney U test (non-parametric t-test)
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 4: Risk Differences Between Women and Men")
        print("=" * 80)
        
        # Filter valid gender data
        gender_data = self.df[self.df['Gender'].isin(['Male', 'Female'])]
        
        print(f"\nSample sizes:")
        gender_counts = gender_data['Gender'].value_counts()
        print(gender_counts)
        
        # Check if we have both genders with sufficient data
        has_male = 'Male' in gender_counts.index and gender_counts['Male'] >= 10
        has_female = 'Female' in gender_counts.index and gender_counts['Female'] >= 10
        
        if not (has_male and has_female):
            print("\n⚠ WARNING: Insufficient data for both genders")
            print("Cannot perform statistical test - need at least 10 records for each gender")
            return {
                'hypothesis': 'No risk difference between Women and Men',
                'p_value_frequency': 1.0,
                'p_value_severity': 1.0,
                'min_p_value': 1.0,
                'reject': False,
                'stats': pd.DataFrame(),
                'note': 'Insufficient data'
            }
        
        # Test 1: Claim Frequency (Chi-square)
        print("\n--- Test 1: Claim Frequency (Chi-square test) ---")
        
        contingency_data = []
        for gender in ['Male', 'Female']:
            gender_subset = gender_data[gender_data['Gender'] == gender]
            with_claims = gender_subset['HasClaim'].sum()
            without_claims = len(gender_subset) - with_claims
            contingency_data.append([with_claims, without_claims])
        
        contingency_table = pd.DataFrame(
            contingency_data,
            index=['Male', 'Female'],
            columns=['With Claims', 'Without Claims']
        )
        
        print("\nContingency Table:")
        print(contingency_table)
        
        # Check for zero expected frequencies
        try:
            chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        except ValueError as e:
            print(f"\n⚠ Cannot perform chi-square test: {e}")
            print("Using Fisher's exact test or alternative approach")
            # Use alternative: compare proportions directly
            from scipy.stats import fisher_exact
            try:
                oddsratio, p_value_freq = fisher_exact(contingency_table.values)
            except:
                p_value_freq = 1.0
            chi2, dof = 0, 1
        
        print(f"\nChi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value_freq:.6f}")
        
        # Test 2: Claim Severity (Mann-Whitney U test)
        print("\n--- Test 2: Claim Severity (Mann-Whitney U test) ---")
        
        male_severity = gender_data[gender_data['Gender'] == 'Male']['ClaimSeverity'].dropna()
        female_severity = gender_data[gender_data['Gender'] == 'Female']['ClaimSeverity'].dropna()
        
        if len(male_severity) > 0 and len(female_severity) > 0:
            u_stat, p_value_sev = mannwhitneyu(male_severity, female_severity, alternative='two-sided')
            print(f"\nMann-Whitney U statistic: {u_stat:.4f}")
            print(f"P-value: {p_value_sev:.6f}")
        else:
            p_value_sev = 1.0
            print("\nInsufficient data for severity test")
        
        # Summary statistics
        print("\n--- Summary Statistics by Gender ---")
        gender_stats = []
        for gender in ['Male', 'Female']:
            gender_subset = gender_data[gender_data['Gender'] == gender]
            freq = self.calculate_claim_frequency('Gender', gender)
            sev = self.calculate_claim_severity('Gender', gender)
            count = len(gender_subset)
            gender_stats.append({
                'Gender': gender,
                'Claim Frequency': freq,
                'Claim Severity (ZAR)': sev,
                'Sample Size': count
            })
        
        stats_df = pd.DataFrame(gender_stats)
        print(stats_df.to_string(index=False))
        
        # Interpretation
        print("\n--- Interpretation ---")
        min_p = min(p_value_freq, p_value_sev)
        
        if min_p < 0.05:
            print(f"✓ REJECT H₀: There IS a significant risk difference between Women and Men")
            print(f"  (p = {min_p:.6f} < 0.05)")
            
            male_freq = stats_df[stats_df['Gender'] == 'Male']['Claim Frequency'].values[0]
            female_freq = stats_df[stats_df['Gender'] == 'Female']['Claim Frequency'].values[0]
            freq_diff = (male_freq - female_freq) * 100
            
            if freq_diff > 0:
                print(f"\n  Key Finding: Men have {abs(freq_diff):.2f} percentage points")
                print(f"  higher claim frequency than Women")
            else:
                print(f"\n  Key Finding: Women have {abs(freq_diff):.2f} percentage points")
                print(f"  higher claim frequency than Men")
        else:
            print(f"✗ FAIL TO REJECT H₀: No significant risk difference between Women and Men")
            print(f"  (p = {min_p:.6f} >= 0.05)")
        
        return {
            'hypothesis': 'No risk difference between Women and Men',
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev,
            'min_p_value': min_p,
            'reject': min_p < 0.05,
            'stats': stats_df
        }
    
    def run_all_tests(self):
        """Run all hypothesis tests and return results"""
        results = {}
        
        results['province'] = self.test_province_risk_differences()
        results['zipcode_risk'] = self.test_zipcode_risk_differences()
        results['zipcode_margin'] = self.test_zipcode_margin_differences()
        results['gender'] = self.test_gender_risk_differences()
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hypothesis tests')
    parser.add_argument('--sample-size', type=int, default=None, 
                       help='Sample size for faster testing (default: full dataset)')
    args = parser.parse_args()
    
    # Initialize tester
    if args.sample_size:
        print(f"Using sample size: {args.sample_size:,}")
        tester = HypothesisTester(sample_size=args.sample_size)
    else:
        print("Using full dataset")
        tester = HypothesisTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Summary
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "REJECTED" if result.get('reject', False) else "NOT REJECTED"
        p_val = result.get('min_p_value', result.get('p_value', 1.0))
        print(f"\n{result['hypothesis']}: {status} (p = {p_val:.6f})")
    
    # Save results for report generation
    import pickle
    output_dir = project_root / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "hypothesis_test_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {output_dir / 'hypothesis_test_results.pkl'}")

