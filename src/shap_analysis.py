"""
SHAP Analysis for Model Interpretability
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import pickle
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """Class for performing SHAP analysis on trained models"""
    
    def __init__(self, models_path=None, results_path=None):
        """
        Initialize SHAP analyzer
        
        Args:
            models_path: Path to saved models directory
            results_path: Path to results pickle file
        """
        if models_path is None:
            models_path = project_root / "outputs" / "models"
        if results_path is None:
            results_path = project_root / "outputs" / "models" / "modeling_results.pkl"
        
        self.models_path = Path(models_path)
        self.results_path = Path(results_path)
        
        # Load results
        try:
            with open(self.results_path, 'rb') as f:
                self.results = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Results file not found at {results_path}")
            self.results = {}
    
    def analyze_model(self, model_type='premium', model_name='XGBoost', 
                     X_test=None, model=None, feature_names=None, sample_size=100):
        """
        Perform SHAP analysis on a model
        
        Args:
            model_type: 'claim_severity' or 'premium'
            model_name: Name of the model
            X_test: Test features (optional, will load from results if not provided)
            model: Trained model (optional, will load if not provided)
            feature_names: Feature names (optional)
            sample_size: Number of samples to use for SHAP (for speed)
        """
        print(f"\n" + "=" * 80)
        print(f"SHAP ANALYSIS: {model_type.upper()} - {model_name}")
        print("=" * 80)
        
        # Load model if not provided
        if model is None:
            model_file = self.models_path / f"{model_type}_{model_name.lower().replace(' ', '_')}.pkl"
            if not model_file.exists():
                print(f"⚠ Model file not found: {model_file}")
                return None
            model = joblib.load(model_file)
            print(f"Loaded model from: {model_file}")
        
        # Load test data if not provided
        if X_test is None or feature_names is None:
            if model_type in self.results.get('results', {}):
                X_test = self.results['results'][model_type].get('X_test')
                feature_names = self.results['results'][model_type].get('feature_names')
            else:
                print("⚠ Test data not available")
                return None
        
        # Sample data for faster SHAP computation
        if len(X_test) > sample_size:
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices] if isinstance(X_test, pd.DataFrame) else X_test[sample_indices]
            print(f"Using {sample_size} samples for SHAP analysis (from {len(X_test)} total)")
        else:
            X_sample = X_test
            print(f"Using all {len(X_sample)} samples for SHAP analysis")
        
        # Convert to numpy if needed
        if isinstance(X_sample, pd.DataFrame):
            X_sample_np = X_sample.values
        else:
            X_sample_np = X_sample
        
        # Create SHAP explainer
        print("\nCreating SHAP explainer...")
        try:
            if model_name == 'XGBoost' or hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
            elif model_name == 'Random Forest':
                explainer = shap.TreeExplainer(model)
            else:
                # For linear models, use KernelExplainer (slower but works)
                explainer = shap.KernelExplainer(model.predict, X_sample_np[:50])  # Use small sample for background
                print("Using KernelExplainer (this may take a while)...")
            
            # Calculate SHAP values
            print("Calculating SHAP values...")
            shap_values = explainer.shap_values(X_sample_np)
            
            # Get feature importance from SHAP
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output models
            
            # Calculate mean absolute SHAP values for feature importance
            mean_shap = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            print(f"\nTop 10 Features by Mean Absolute SHAP Value:")
            print(feature_importance.head(10).to_string(index=False))
            
            # Calculate feature impact (average SHAP value)
            feature_impact = pd.DataFrame({
                'feature': feature_names,
                'mean_shap': shap_values.mean(0),
                'std_shap': shap_values.std(0)
            }).sort_values('mean_shap', ascending=False)
            
            return {
                'shap_values': shap_values,
                'explainer': explainer,
                'X_sample': X_sample_np,
                'feature_names': feature_names,
                'feature_importance': feature_importance,
                'feature_impact': feature_impact
            }
            
        except Exception as e:
            print(f"⚠ Error in SHAP analysis: {e}")
            print("Falling back to model feature importance...")
            return None
    
    def get_top_features(self, shap_results, top_n=10):
        """
        Get top N features with business interpretation
        
        Args:
            shap_results: Results from analyze_model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature interpretations
        """
        if shap_results is None:
            return None
        
        importance_df = shap_results['feature_importance'].head(top_n).copy()
        impact_df = shap_results['feature_impact']
        
        # Merge with impact information
        importance_df = importance_df.merge(
            impact_df[['feature', 'mean_shap']],
            on='feature',
            how='left'
        )
        
        # Add interpretation
        interpretations = []
        for _, row in importance_df.iterrows():
            feature = row['feature']
            mean_shap = row['mean_shap']
            
            # Generate interpretation based on feature name
            if 'VehicleAge' in feature or 'RegistrationYear' in feature:
                interp = f"For every year older a vehicle is, the predicted {'claim amount' if 'claim' in feature else 'premium'} {'increases' if mean_shap > 0 else 'decreases'} by approximately ZAR {abs(mean_shap):,.2f}, holding other factors constant."
            elif 'Province' in feature:
                interp = f"Policies in {feature.split('_')[-1]} show {'higher' if mean_shap > 0 else 'lower'} risk/premium by approximately ZAR {abs(mean_shap):,.2f} compared to baseline, holding other factors constant."
            elif 'SumInsured' in feature or 'cubiccapacity' in feature or 'kilowatts' in feature:
                interp = f"For each unit increase in {feature}, the predicted value {'increases' if mean_shap > 0 else 'decreases'} by approximately ZAR {abs(mean_shap):,.2f}, holding other factors constant."
            elif 'CoverType' in feature or 'CoverCategory' in feature:
                interp = f"Coverage type {feature.split('_')[-1]} is associated with {'higher' if mean_shap > 0 else 'lower'} {'claim amounts' if 'claim' in feature else 'premiums'} by approximately ZAR {abs(mean_shap):,.2f}, holding other factors constant."
            else:
                interp = f"Feature {feature} has a {'positive' if mean_shap > 0 else 'negative'} impact of approximately ZAR {abs(mean_shap):,.2f} on predictions, holding other factors constant."
            
            interpretations.append(interp)
        
        importance_df['business_interpretation'] = interpretations
        
        return importance_df


if __name__ == "__main__":
    analyzer = SHAPAnalyzer()
    
    # Analyze premium model
    print("\nAnalyzing Premium Model...")
    premium_shap = analyzer.analyze_model('premium', 'XGBoost', sample_size=100)
    
    if premium_shap:
        top_features = analyzer.get_top_features(premium_shap, top_n=10)
        print("\n" + "=" * 80)
        print("TOP 10 FEATURES WITH BUSINESS INTERPRETATION")
        print("=" * 80)
        print(top_features[['feature', 'mean_abs_shap', 'business_interpretation']].to_string(index=False))
    
    # Analyze claim severity model (if enough data)
    print("\n\nAnalyzing Claim Severity Model...")
    claim_shap = analyzer.analyze_model('claim_severity', 'XGBoost', sample_size=50)
    
    if claim_shap:
        top_features = analyzer.get_top_features(claim_shap, top_n=10)
        print("\n" + "=" * 80)
        print("TOP 10 FEATURES WITH BUSINESS INTERPRETATION")
        print("=" * 80)
        print(top_features[['feature', 'mean_abs_shap', 'business_interpretation']].to_string(index=False))

