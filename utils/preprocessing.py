"""
Data preprocessing utilities for Decision Trees Dashboard.
Handles CSV loading, cleaning, encoding, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, Union
import csv
import io


def load_data(file_path: Union[str, any]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load data from CSV file with intelligent parsing.
    
    Args:
        file_path: Path to CSV file or UploadedFile object from Streamlit
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        # Handle Streamlit UploadedFile
        if hasattr(file_path, 'read'):
            # Read the file content
            content = file_path.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Detect delimiter
            sample = content[:8192] if len(content) > 8192 else content
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            # Load with detected delimiter
            df = pd.read_csv(io.StringIO(content), sep=delimiter)
        else:
            # Handle regular file path
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(8192)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            
            # Load with detected delimiter
            df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8')
        
        # Check if it has header
        if df.columns[0].isdigit() or len(df.columns) == 1:
            # Reload without header
            if hasattr(file_path, 'read'):
                file_path.seek(0)  # Reset file pointer
                df = pd.read_csv(file_path, sep=delimiter, header=None)
            else:
                df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8', header=None)
            df.columns = [f'Feature_{i}' for i in range(len(df.columns))]
        
        # Remove index/ID columns
        for col in df.columns:
            if col.lower() in ['index', 'id', 'unnamed: 0']:
                df = df.drop(columns=[col])
                
        return df, None
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def preprocess_data(df: pd.DataFrame, 
                    target_column: str,
                    handle_missing: str = 'mean',
                    encode_categorical: bool = True) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Preprocess data for decision tree modeling.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        handle_missing: Strategy for missing values ('mean', 'median', 'mode', 'drop')
        encode_categorical: Whether to encode categorical variables
        
    Returns:
        Tuple of (X, y, preprocessing_info)
    """
    df_processed = df.copy()
    preprocessing_info = {
        'original_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'categorical_columns': [],
        'label_encoders': {}
    }
    
    # Separate features and target
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    y = df_processed[target_column]
    X = df_processed.drop(columns=[target_column])
    
    # Handle missing values
    if handle_missing == 'drop':
        X = X.dropna()
        y = y[X.index]
    else:
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['int64', 'float64']:
                    if handle_missing == 'mean':
                        X[col].fillna(X[col].mean(), inplace=True)
                    elif handle_missing == 'median':
                        X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
    
    # Encode categorical variables
    if encode_categorical:
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                preprocessing_info['categorical_columns'].append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                preprocessing_info['label_encoders'][col] = le
    
    # Encode target if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index, name=target_column)
        preprocessing_info['target_encoder'] = le_target
    
    preprocessing_info['processed_shape'] = (X.shape[0], X.shape[1])
    
    return X, y, preprocessing_info


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'column_types': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return summary
