"""
Utility functions for the insurance analytics project
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_loss_ratio(claims: pd.Series, premium: pd.Series) -> pd.Series:
    """
    Calculate Loss Ratio: TotalClaims / TotalPremium
    
    Args:
        claims: Series of total claims
        premium: Series of total premium
        
    Returns:
        Series of loss ratios
    """
    return claims / premium.replace(0, np.nan)


def format_currency(value: float, currency: str = "ZAR") -> str:
    """
    Format numeric value as currency string
    
    Args:
        value: Numeric value to format
        currency: Currency code (default: ZAR)
        
    Returns:
        Formatted currency string
    """
    return f"{currency} {value:,.2f}"


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        data: Series of numeric data
        multiplier: IQR multiplier (default: 1.5)
        
    Returns:
        Boolean series indicating outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data < lower_bound) | (data > upper_bound)


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive data summary including data types and missing values
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        DataFrame with summary statistics
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'DataType': df.dtypes,
        'NonNullCount': df.count(),
        'NullCount': df.isnull().sum(),
        'NullPercentage': (df.isnull().sum() / len(df) * 100).round(2),
        'UniqueCount': [df[col].nunique() for col in df.columns]
    })
    return summary

