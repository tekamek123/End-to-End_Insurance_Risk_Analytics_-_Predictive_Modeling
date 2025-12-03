"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_insurance_data, clean_numeric_columns, prepare_data_for_analysis


def test_load_insurance_data():
    """Test data loading functionality"""
    # This test will only run if data file exists
    data_path = Path("data/MachineLearningRating_v3.txt")
    if data_path.exists():
        df = load_insurance_data(sample_size=1000)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'TotalPremium' in df.columns
        assert 'TotalClaims' in df.columns


def test_clean_numeric_columns():
    """Test numeric column cleaning"""
    # Create sample data
    df = pd.DataFrame({
        'TotalPremium': ['100', '200', 'invalid'],
        'TotalClaims': ['50', '75', '25']
    })
    
    cleaned = clean_numeric_columns(df)
    assert pd.api.types.is_numeric_dtype(cleaned['TotalPremium'])
    assert pd.api.types.is_numeric_dtype(cleaned['TotalClaims'])


def test_prepare_data_for_analysis():
    """Test data preparation"""
    # Create sample data
    df = pd.DataFrame({
        'TotalPremium': [100, 200, 300],
        'TotalClaims': [50, 75, 100],
        'TransactionMonth': pd.date_range('2014-02-01', periods=3, freq='M')
    })
    
    prepared = prepare_data_for_analysis(df)
    assert 'LossRatio' in prepared.columns
    assert 'Year' in prepared.columns
    assert 'Month' in prepared.columns

