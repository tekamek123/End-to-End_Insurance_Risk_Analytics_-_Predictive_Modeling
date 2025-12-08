"""
Predictive Modeling for Insurance Risk Analytics
Task 4: Build and evaluate predictive models for claim severity and premium optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from pathlib import Path
import sys
import warnings
import pickle
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class InsuranceModeling:
    """Class for building and evaluating predictive models"""
    
    def __init__(self, data_path=None, sample_size=None):
        """
        Initialize modeling pipeline
        
        Args:
            data_path: Path to data file
            sample_size: Sample size for faster testing
        """
        print("=" * 80)
        print("INSURANCE RISK ANALYTICS - PREDICTIVE MODELING")
        print("=" * 80)
        
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
        
        print(f"Data loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        # Prepare data
        self._prepare_data()
        
        # Models storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
    
    def _prepare_data(self):
        """Prepare data for modeling"""
        print("\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        
        # Convert numeric columns
        numeric_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'PostalCode',
                       'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
                       'NumberOfDoors', 'CustomValueEstimate', 'CalculatedPremiumPerTerm']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create derived features
        print("\n1. Creating derived features...")
        
        # Time-based features
        if 'TransactionMonth' in self.df.columns:
            self.df['Year'] = self.df['TransactionMonth'].dt.year
            self.df['Month'] = self.df['TransactionMonth'].dt.month
            self.df['YearMonth'] = self.df['TransactionMonth'].dt.to_period('M')
        
        # Vehicle age
        if 'RegistrationYear' in self.df.columns and 'Year' in self.df.columns:
            self.df['VehicleAge'] = self.df['Year'] - self.df['RegistrationYear']
            self.df['VehicleAge'] = self.df['VehicleAge'].clip(lower=0, upper=50)  # Cap at 50 years
        
        # Risk indicators
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        self.df['LossRatio'] = np.where(
            self.df['TotalPremium'] > 0,
            self.df['TotalClaims'] / self.df['TotalPremium'],
            np.nan
        )
        
        # Premium per unit insured
        self.df['PremiumPerInsured'] = np.where(
            self.df['SumInsured'] > 0,
            self.df['TotalPremium'] / self.df['SumInsured'],
            np.nan
        )
        
        # Claim severity (for policies with claims)
        self.df['ClaimSeverity'] = np.where(
            self.df['HasClaim'] == 1,
            self.df['TotalClaims'],
            np.nan
        )
        
        print(f"   Created {len([c for c in self.df.columns if c not in numeric_cols])} derived features")
        
        # Handle missing values
        print("\n2. Handling missing values...")
        self._handle_missing_values()
        
        # Filter invalid data
        print("\n3. Filtering invalid data...")
        initial_count = len(self.df)
        self.df = self.df[self.df['TotalPremium'] >= 0]
        self.df = self.df[self.df['TotalClaims'] >= 0]
        filtered_count = len(self.df)
        print(f"   Removed {initial_count - filtered_count:,} rows with negative values")
        
        print(f"\nFinal dataset: {len(self.df):,} rows, {len(self.df.columns)} columns")
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        missing_before = self.df.isnull().sum().sum()
        
        # For high missing rate columns (>50%), drop them
        missing_rates = self.df.isnull().sum() / len(self.df)
        high_missing_cols = missing_rates[missing_rates > 0.5].index.tolist()
        if high_missing_cols:
            print(f"   Dropping {len(high_missing_cols)} columns with >50% missing: {high_missing_cols[:5]}...")
            self.df = self.df.drop(columns=high_missing_cols)
        
        # For numeric columns, impute with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0 and self.df[col].isnull().sum() / len(self.df) < 0.5:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        # For categorical columns, impute with mode or 'Unknown'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"   Reduced missing values from {missing_before:,} to {missing_after:,}")
    
    def prepare_features(self, target='TotalClaims', include_categorical=True):
        """
        Prepare features for modeling
        
        Args:
            target: Target variable name
            include_categorical: Whether to include categorical features
            
        Returns:
            X, y: Features and target
        """
        print(f"\nPreparing features for target: {target}")
        
        # Select feature columns
        exclude_cols = [
            'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 'YearMonth',
            'TotalPremium', 'TotalClaims', 'HasClaim', 'Margin', 'LossRatio',
            'ClaimSeverity', 'PremiumPerInsured', target
        ]
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Separate numeric and categorical
        numeric_features = []
        categorical_features = []
        
        for col in feature_cols:
            if self.df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            elif include_categorical and self.df[col].dtype == 'object':
                # Only include categoricals with reasonable cardinality
                if self.df[col].nunique() < 50:
                    categorical_features.append(col)
        
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
        # Create feature matrix
        X = self.df[numeric_features + categorical_features].copy()
        y = self.df[target].copy()
        
        # Encode categorical variables
        if categorical_features:
            print(f"   Encoding {len(categorical_features)} categorical features...")
            X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True, dummy_na=True)
            X = X_encoded
        
        # Remove any remaining NaN
        X = X.fillna(X.median() if len(X.select_dtypes(include=[np.number]).columns) > 0 else 0)
        
        print(f"   Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
        
        return X, y
    
    def build_claim_severity_model(self):
        """Build model to predict claim severity (for policies with claims)"""
        print("\n" + "=" * 80)
        print("CLAIM SEVERITY PREDICTION MODEL")
        print("=" * 80)
        
        # Filter to policies with claims
        claims_data = self.df[self.df['TotalClaims'] > 0].copy()
        print(f"\nPolicies with claims: {len(claims_data):,}")
        
        if len(claims_data) < 100:
            print("⚠ Warning: Insufficient data for claim severity modeling")
            return None
        
        # Prepare features
        X, y = self.prepare_features(target='TotalClaims', include_categorical=True)
        X = X.loc[claims_data.index]
        y = y.loc[claims_data.index]
        
        # Remove any infinite or very large values
        # Convert X to numeric and handle any non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        if len(X_numeric.columns) < len(X.columns):
            # Convert boolean columns to int
            for col in X.columns:
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)
            X_numeric = X.select_dtypes(include=[np.number])
        
        mask = np.isfinite(X_numeric.values).all(axis=1) & np.isfinite(y.values) & (y.values < 1e6)
        X = X[mask]
        y = y[mask]
        
        print(f"\nAfter filtering: {len(X):,} samples")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['claim_severity'] = scaler
        
        # Models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n--- Training {model_name} ---")
            
            # Use scaled data for linear regression, original for tree-based
            if model_name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[model_name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'y_test': y_test,
                'y_pred': y_pred_test,
                'feature_names': X.columns.tolist()
            }
            
            print(f"  Train RMSE: {train_rmse:,.2f}")
            print(f"  Test RMSE: {test_rmse:,.2f}")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Train MAE: {train_mae:,.2f}")
            print(f"  Test MAE: {test_mae:,.2f}")
        
        self.models['claim_severity'] = results
        self.results['claim_severity'] = {
            'X_test': X_test,
            'feature_names': X.columns.tolist()
        }
        
        return results
    
    def build_premium_model(self):
        """Build model to predict premium"""
        print("\n" + "=" * 80)
        print("PREMIUM OPTIMIZATION MODEL")
        print("=" * 80)
        
        # Prepare features
        X, y = self.prepare_features(target='CalculatedPremiumPerTerm', include_categorical=True)
        
        # Remove invalid values
        # Convert X to numeric and handle any non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        if len(X_numeric.columns) < len(X.columns):
            # Convert boolean columns to int
            for col in X.columns:
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)
            X_numeric = X.select_dtypes(include=[np.number])
        
        mask = np.isfinite(X_numeric.values).all(axis=1) & np.isfinite(y.values) & (y.values > 0) & (y.values < 1e5)
        X = X[mask]
        y = y[mask]
        
        print(f"\nValid samples: {len(X):,}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['premium'] = scaler
        
        # Models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n--- Training {model_name} ---")
            
            if model_name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[model_name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'y_test': y_test,
                'y_pred': y_pred_test,
                'feature_names': X.columns.tolist()
            }
            
            print(f"  Train RMSE: {train_rmse:,.2f}")
            print(f"  Test RMSE: {test_rmse:,.2f}")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
        
        self.models['premium'] = results
        self.results['premium'] = {
            'X_test': X_test,
            'feature_names': X.columns.tolist()
        }
        
        return results
    
    def analyze_feature_importance(self, model_type='claim_severity', model_name='XGBoost'):
        """
        Analyze feature importance using model's built-in feature importance
        
        Args:
            model_type: 'claim_severity' or 'premium'
            model_name: Name of the model to analyze
        """
        print(f"\n" + "=" * 80)
        print(f"FEATURE IMPORTANCE ANALYSIS: {model_type.upper()} - {model_name}")
        print("=" * 80)
        
        if model_type not in self.models:
            print(f"⚠ No models found for {model_type}")
            return None
        
        if model_name not in self.models[model_type]:
            print(f"⚠ Model {model_name} not found")
            return None
        
        model = self.models[model_type][model_name]['model']
        feature_names = self.models[model_type][model_name]['feature_names']
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("⚠ Model does not support feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def save_results(self, output_dir=None):
        """Save models and results"""
        if output_dir is None:
            output_dir = project_root / "outputs" / "models"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_type, models in self.models.items():
            for model_name, result in models.items():
                filename = f"{model_type}_{model_name.lower().replace(' ', '_')}.pkl"
                joblib.dump(result['model'], output_dir / filename)
        
        # Save results
        results_file = output_dir / "modeling_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'models': {k: {m: {'metrics': {mk: mv for mk, mv in v.items() 
                                               if mk not in ['model', 'y_test', 'y_pred']}} 
                              for m, v in models.items()} 
                          for k, models in self.models.items()},
                'results': self.results
            }, f)
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build predictive models')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for faster testing')
    args = parser.parse_args()
    
    # Initialize modeling
    if args.sample_size:
        print(f"Using sample size: {args.sample_size:,}")
        modeler = InsuranceModeling(sample_size=args.sample_size)
    else:
        print("Using full dataset")
        modeler = InsuranceModeling()
    
    # Build models
    print("\n" + "=" * 80)
    print("BUILDING MODELS")
    print("=" * 80)
    
    # Claim severity model
    claim_results = modeler.build_claim_severity_model()
    
    # Premium model
    premium_results = modeler.build_premium_model()
    
    # Feature importance
    if claim_results:
        modeler.analyze_feature_importance('claim_severity', 'XGBoost')
    
    modeler.analyze_feature_importance('premium', 'XGBoost')
    
    # Save results
    modeler.save_results()
    
    # Save full results for SHAP analysis
    import pickle
    full_results_path = project_root / "outputs" / "models" / "full_modeling_results.pkl"
    with open(full_results_path, 'wb') as f:
        pickle.dump({
            'models': modeler.models,
            'results': modeler.results,
            'scalers': modeler.scalers
        }, f)
    
    print(f"\nFull results saved for SHAP analysis: {full_results_path}")
    print("\n" + "=" * 80)
    print("MODELING COMPLETE")
    print("=" * 80)
    print("\nNext step: Run SHAP analysis with:")
    print("  python src/shap_analysis.py")

