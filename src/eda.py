"""
Exploratory Data Analysis (EDA) for Insurance Risk Analytics
Task 1: Project Planning - EDA & Statistics
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional

from src.data_loader import load_insurance_data, prepare_data_for_analysis
from src.utils import (
    calculate_loss_ratio,
    detect_outliers_iqr,
    get_data_summary
)

warnings.filterwarnings('ignore')

# Set style for beautiful plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


class InsuranceEDA:
    """Class for performing comprehensive EDA on insurance data"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize EDA with data loading
        
        Args:
            data_path: Path to data file (optional)
        """
        print("=" * 80)
        print("INSURANCE RISK ANALYTICS - EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        self.df = load_insurance_data(file_path=data_path)
        self.df = prepare_data_for_analysis(self.df)
        
        # Create output directory (relative to project root)
        self.output_dir = project_root / "outputs" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nData shape: {self.df.shape}")
        print(f"Date range: {self.df['TransactionMonth'].min()} to {self.df['TransactionMonth'].max()}")
    
    def data_summarization(self):
        """Perform data summarization and descriptive statistics"""
        print("\n" + "=" * 80)
        print("1. DATA SUMMARIZATION")
        print("=" * 80)
        
        # Basic info
        print("\n1.1 Data Structure:")
        print(f"   - Total rows: {len(self.df):,}")
        print(f"   - Total columns: {len(self.df.columns)}")
        
        # Data types summary
        print("\n1.2 Data Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   - {dtype}: {count} columns")
        
        # Descriptive statistics for numerical columns
        print("\n1.3 Descriptive Statistics (Numerical Variables):")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        key_numerical = ['TotalPremium', 'TotalClaims', 'SumInsured', 
                        'CustomValueEstimate', 'CalculatedPremiumPerTerm']
        
        desc_stats = self.df[key_numerical].describe()
        print(desc_stats.round(2))
        
        # Variability measures
        print("\n1.4 Variability Measures (Coefficient of Variation):")
        for col in key_numerical:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan
                print(f"   - {col}: CV = {cv:.2f}%")
        
        return desc_stats
    
    def data_quality_assessment(self):
        """Assess data quality including missing values"""
        print("\n" + "=" * 80)
        print("2. DATA QUALITY ASSESSMENT")
        print("=" * 80)
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'MissingCount': missing.values,
            'MissingPercentage': missing_pct.values
        }).sort_values('MissingCount', ascending=False)
        
        print("\n2.1 Missing Values:")
        print(missing_df[missing_df['MissingCount'] > 0].head(20).to_string(index=False))
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\n2.2 Duplicate Rows: {duplicates:,} ({duplicates/len(self.df)*100:.2f}%)")
        
        return missing_df
    
    def univariate_analysis(self):
        """Perform univariate analysis with distributions"""
        print("\n" + "=" * 80)
        print("3. UNIVARIATE ANALYSIS")
        print("=" * 80)
        
        # Numerical distributions
        numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            if col in self.df.columns:
                data = self.df[col].dropna()
                axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_yscale('log')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'univariate_numerical_distributions.png', dpi=300, bbox_inches='tight')
        print("\n3.1 Saved: univariate_numerical_distributions.png")
        plt.close()
        
        # Categorical distributions
        categorical_cols = ['Province', 'Gender', 'VehicleType', 'CoverType']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            if col in self.df.columns:
                value_counts = self.df[col].value_counts().head(10)
                axes[idx].barh(range(len(value_counts)), value_counts.values, color=sns.color_palette("husl", len(value_counts)))
                axes[idx].set_yticks(range(len(value_counts)))
                axes[idx].set_yticklabels(value_counts.index)
                axes[idx].set_title(f'Top 10 {col} Distribution', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Count')
                axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'univariate_categorical_distributions.png', dpi=300, bbox_inches='tight')
        print("3.2 Saved: univariate_categorical_distributions.png")
        plt.close()
    
    def bivariate_multivariate_analysis(self):
        """Perform bivariate and multivariate analysis"""
        print("\n" + "=" * 80)
        print("4. BIVARIATE & MULTIVARIATE ANALYSIS")
        print("=" * 80)
        
        # Correlation matrix
        numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 
                        'CustomValueEstimate', 'CalculatedPremiumPerTerm']
        corr_data = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: Financial Variables', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("\n4.1 Saved: correlation_matrix.png")
        plt.close()
        
        # Monthly trends: TotalPremium vs TotalClaims
        monthly_data = self.df.groupby('YearMonth').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).reset_index()
        monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()
        
        ax1.plot(monthly_data['YearMonth'], monthly_data['TotalPremium'], 
                marker='o', color='blue', linewidth=2, label='Total Premium')
        ax2.plot(monthly_data['YearMonth'], monthly_data['TotalClaims'], 
                marker='s', color='red', linewidth=2, label='Total Claims')
        
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Total Premium (ZAR)', color='blue', fontsize=12)
        ax2.set_ylabel('Total Claims (ZAR)', color='red', fontsize=12)
        ax1.set_title('Monthly Trends: Premium vs Claims', fontsize=14, fontweight='bold', pad=20)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'monthly_trends.png', dpi=300, bbox_inches='tight')
        print("4.2 Saved: monthly_trends.png")
        plt.close()
        
        # Scatter: Premium vs Claims by Province
        province_data = self.df.groupby('Province').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).reset_index()
        province_data['LossRatio'] = province_data['TotalClaims'] / province_data['TotalPremium']
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(province_data['TotalPremium'], province_data['TotalClaims'],
                            s=province_data['LossRatio']*1000, alpha=0.6, 
                            c=province_data['LossRatio'], cmap='RdYlGn_r', edgecolors='black')
        
        for idx, row in province_data.iterrows():
            plt.annotate(row['Province'], 
                        (row['TotalPremium'], row['TotalClaims']),
                        fontsize=9, alpha=0.7)
        
        plt.xlabel('Total Premium (ZAR)', fontsize=12)
        plt.ylabel('Total Claims (ZAR)', fontsize=12)
        plt.title('Premium vs Claims by Province\n(Bubble size = Loss Ratio)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter, label='Loss Ratio')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'premium_vs_claims_by_province.png', dpi=300, bbox_inches='tight')
        print("4.3 Saved: premium_vs_claims_by_province.png")
        plt.close()
    
    def outlier_detection(self):
        """Detect outliers using box plots"""
        print("\n" + "=" * 80)
        print("5. OUTLIER DETECTION")
        print("=" * 80)
        
        numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            if col in self.df.columns:
                data = self.df[col].dropna()
                bp = axes[idx].boxplot(data, vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                axes[idx].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('Value')
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                # Count outliers
                outliers = detect_outliers_iqr(data)
                outlier_count = outliers.sum()
                axes[idx].text(0.5, 0.95, f'Outliers: {outlier_count:,} ({outlier_count/len(data)*100:.2f}%)',
                             transform=axes[idx].transAxes, ha='center', va='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outlier_detection_boxplots.png', dpi=300, bbox_inches='tight')
        print("\n5.1 Saved: outlier_detection_boxplots.png")
        plt.close()
    
    def answer_guiding_questions(self):
        """Answer the guiding questions from the task"""
        print("\n" + "=" * 80)
        print("6. ANSWERING GUIDING QUESTIONS")
        print("=" * 80)
        
        # Q1: Overall Loss Ratio
        total_premium = self.df['TotalPremium'].sum()
        total_claims = self.df['TotalClaims'].sum()
        overall_loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        print(f"\nQ1. Overall Loss Ratio: {overall_loss_ratio:.4f} ({overall_loss_ratio*100:.2f}%)")
        print(f"   - Total Premium: ZAR {total_premium:,.2f}")
        print(f"   - Total Claims: ZAR {total_claims:,.2f}")
        
        # Loss Ratio by Province
        print("\nQ1a. Loss Ratio by Province:")
        province_lr = self.df.groupby('Province').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        })
        province_lr['LossRatio'] = province_lr['TotalClaims'] / province_lr['TotalPremium']
        province_lr = province_lr.sort_values('LossRatio', ascending=False)
        print(province_lr[['LossRatio']].round(4))
        
        # Loss Ratio by VehicleType
        if 'VehicleType' in self.df.columns:
            print("\nQ1b. Loss Ratio by VehicleType:")
            vehicle_lr = self.df.groupby('VehicleType').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            })
            vehicle_lr['LossRatio'] = vehicle_lr['TotalClaims'] / vehicle_lr['TotalPremium']
            vehicle_lr = vehicle_lr.sort_values('LossRatio', ascending=False)
            print(vehicle_lr[['LossRatio']].head(10).round(4))
        
        # Loss Ratio by Gender
        if 'Gender' in self.df.columns:
            print("\nQ1c. Loss Ratio by Gender:")
            gender_lr = self.df.groupby('Gender').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            })
            gender_lr['LossRatio'] = gender_lr['TotalClaims'] / gender_lr['TotalPremium']
            gender_lr = gender_lr.sort_values('LossRatio', ascending=False)
            print(gender_lr[['LossRatio']].round(4))
        
        # Q2: Distributions and outliers
        print("\nQ2. Key Financial Variables - Outlier Summary:")
        for col in ['TotalClaims', 'CustomValueEstimate']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                outliers = detect_outliers_iqr(data)
                print(f"   - {col}: {outliers.sum():,} outliers ({outliers.sum()/len(data)*100:.2f}%)")
        
        # Q3: Temporal trends
        print("\nQ3. Temporal Trends:")
        monthly_stats = self.df.groupby('YearMonth').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique'
        }).reset_index()
        monthly_stats['LossRatio'] = monthly_stats['TotalClaims'] / monthly_stats['TotalPremium']
        monthly_stats['YearMonth'] = monthly_stats['YearMonth'].astype(str)
        print(monthly_stats[['YearMonth', 'PolicyID', 'TotalPremium', 'TotalClaims', 'LossRatio']].tail(6))
        
        # Q4: Vehicle makes/models
        if 'make' in self.df.columns:
            print("\nQ4. Top Vehicle Makes by Total Claims:")
            make_claims = self.df.groupby('make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).sort_values('TotalClaims', ascending=False)
            make_claims['LossRatio'] = make_claims['TotalClaims'] / make_claims['TotalPremium']
            print(make_claims.head(10))
    
    def create_creative_visualizations(self):
        """Create 3 creative and beautiful visualizations with key insights"""
        print("\n" + "=" * 80)
        print("7. CREATIVE VISUALIZATIONS")
        print("=" * 80)
        
        # Visualization 1: Loss Ratio Heatmap by Province and VehicleType
        if 'Province' in self.df.columns and 'VehicleType' in self.df.columns:
            pivot_data = self.df.groupby(['Province', 'VehicleType']).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).reset_index()
            pivot_data['LossRatio'] = pivot_data['TotalClaims'] / pivot_data['TotalPremium']
            
            # Create pivot table for heatmap
            heatmap_data = pivot_data.pivot(index='Province', columns='VehicleType', values='LossRatio')
            
            plt.figure(figsize=(16, 10))
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                       center=heatmap_data.median().median(), cbar_kws={'label': 'Loss Ratio'})
            plt.title('Loss Ratio Heatmap: Province × Vehicle Type\n(Lower is Better - Green = Low Risk)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Vehicle Type', fontsize=12)
            plt.ylabel('Province', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'creative_viz1_loss_ratio_heatmap.png', dpi=300, bbox_inches='tight')
            print("\n7.1 Saved: creative_viz1_loss_ratio_heatmap.png")
            plt.close()
        
        # Visualization 2: Risk-return scatter with bubble size
        province_summary = self.df.groupby('Province').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique'
        }).reset_index()
        province_summary['LossRatio'] = province_summary['TotalClaims'] / province_summary['TotalPremium']
        province_summary['AvgPremium'] = province_summary['TotalPremium'] / province_summary['PolicyID']
        
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(province_summary['LossRatio'], 
                           province_summary['AvgPremium'],
                           s=province_summary['PolicyID']*2, 
                           alpha=0.6, c=province_summary['TotalPremium'],
                           cmap='viridis', edgecolors='black', linewidth=2)
        
        for idx, row in province_summary.iterrows():
            ax.annotate(row['Province'], 
                       (row['LossRatio'], row['AvgPremium']),
                       fontsize=10, fontweight='bold', alpha=0.8)
        
        ax.set_xlabel('Loss Ratio (Risk)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Premium per Policy (ZAR)', fontsize=14, fontweight='bold')
        ax.set_title('Risk-Return Analysis by Province\n(Bubble size = Number of Policies, Color = Total Premium)', 
                    fontsize=16, fontweight='bold', pad=20)
        plt.colorbar(scatter, label='Total Premium (ZAR)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'creative_viz2_risk_return_province.png', dpi=300, bbox_inches='tight')
        print("7.2 Saved: creative_viz2_risk_return_province.png")
        plt.close()
        
        # Visualization 3: Temporal evolution with multiple metrics
        monthly_evolution = self.df.groupby('YearMonth').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique',
            'SumInsured': 'mean'
        }).reset_index()
        monthly_evolution['LossRatio'] = monthly_evolution['TotalClaims'] / monthly_evolution['TotalPremium']
        monthly_evolution['YearMonth'] = monthly_evolution['YearMonth'].astype(str)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Premium and Claims over time
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        ax1.plot(monthly_evolution['YearMonth'], monthly_evolution['TotalPremium'], 
                'o-', color='blue', linewidth=2, markersize=8, label='Premium')
        ax1_twin.plot(monthly_evolution['YearMonth'], monthly_evolution['TotalClaims'], 
                      's-', color='red', linewidth=2, markersize=8, label='Claims')
        ax1.set_xlabel('Month', fontweight='bold')
        ax1.set_ylabel('Total Premium (ZAR)', color='blue', fontweight='bold')
        ax1_twin.set_ylabel('Total Claims (ZAR)', color='red', fontweight='bold')
        ax1.set_title('Premium vs Claims Over Time', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Loss Ratio trend
        ax2 = axes[0, 1]
        ax2.plot(monthly_evolution['YearMonth'], monthly_evolution['LossRatio'], 
                'o-', color='green', linewidth=2, markersize=8)
        ax2.axhline(y=monthly_evolution['LossRatio'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {monthly_evolution["LossRatio"].mean():.3f}')
        ax2.set_xlabel('Month', fontweight='bold')
        ax2.set_ylabel('Loss Ratio', fontweight='bold')
        ax2.set_title('Loss Ratio Trend', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Number of policies
        ax3 = axes[1, 0]
        ax3.bar(monthly_evolution['YearMonth'], monthly_evolution['PolicyID'], 
               color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Month', fontweight='bold')
        ax3.set_ylabel('Number of Policies', fontweight='bold')
        ax3.set_title('Policy Count Over Time', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: Average Sum Insured
        ax4 = axes[1, 1]
        ax4.plot(monthly_evolution['YearMonth'], monthly_evolution['SumInsured'], 
                'o-', color='orange', linewidth=2, markersize=8)
        ax4.set_xlabel('Month', fontweight='bold')
        ax4.set_ylabel('Average Sum Insured (ZAR)', fontweight='bold')
        ax4.set_title('Average Coverage Over Time', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Evolution: Key Insurance Metrics (Feb 2014 - Aug 2015)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'creative_viz3_temporal_evolution.png', dpi=300, bbox_inches='tight')
        print("7.3 Saved: creative_viz3_temporal_evolution.png")
        plt.close()
    
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE EDA PIPELINE")
        print("=" * 80)
        
        self.data_summarization()
        self.data_quality_assessment()
        self.univariate_analysis()
        self.bivariate_multivariate_analysis()
        self.outlier_detection()
        self.answer_guiding_questions()
        self.create_creative_visualizations()
        
        print("\n" + "=" * 80)
        print("EDA COMPLETE!")
        print(f"All visualizations saved to: {self.output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    # Initialize and run EDA
    eda = InsuranceEDA()
    eda.run_full_eda()

