"""
Data loading utilities for insurance analytics project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


def load_insurance_data(
    file_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load insurance data from pipe-delimited text file
    
    Args:
        file_path: Path to data file. If None, uses default path.
        sample_size: If provided, load only a sample of the data
        random_state: Random seed for sampling
        
    Returns:
        DataFrame with insurance data
    """
    if file_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "MachineLearningRating_v3.txt"
    
    print(f"Loading data from: {file_path}")
    
    # Read pipe-delimited file
    df = pd.read_csv(
        file_path,
        sep='|',
        low_memory=False,
        parse_dates=['TransactionMonth'],
        date_parser=lambda x: pd.to_datetime(x, errors='coerce')
    )
    
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"Sampled to {len(df):,} rows")
    
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns by converting to appropriate types
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # List of numeric columns that should be converted
    numeric_columns = [
        'TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm',
        'CustomValueEstimate', 'CapitalOutstanding', 'Cylinders', 'cubiccapacity',
        'kilowatts', 'NumberOfDoors', 'RegistrationYear', 'PostalCode'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Replace empty strings and invalid values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def prepare_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for analysis by cleaning and creating derived features
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Prepared DataFrame
    """
    df = clean_numeric_columns(df)
    
    # Create derived features
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    
    if 'TransactionMonth' in df.columns:
        df['Year'] = df['TransactionMonth'].dt.year
        df['Month'] = df['TransactionMonth'].dt.month
        df['YearMonth'] = df['TransactionMonth'].dt.to_period('M')
    
    return df

