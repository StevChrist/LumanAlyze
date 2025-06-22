import numpy as np
import pandas as pd
import math

def safe_json_serializer(obj):
    """Convert numpy/pandas objects to JSON-serializable format"""
    if isinstance(obj, (np.integer, np.floating)):
        # Handle infinity and NaN values
        if np.isnan(obj) or np.isinf(obj) or math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # Clean array of infinity/NaN values
        cleaned_array = np.where(np.isfinite(obj), obj, None)
        return cleaned_array.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, float):
        # Handle regular Python floats
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def clean_data_for_json(data):
    """Clean entire data structure for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    else:
        return safe_json_serializer(data)

def serialize_dataframe_preview(df, max_rows=5):
    """Serialize DataFrame preview for JSON response"""
    preview_df = df.head(max_rows)
    return clean_data_for_json(preview_df.to_dict('records'))

def serialize_missing_values(df):
    """Serialize missing values count for JSON response"""
    missing_values = df.isnull().sum().to_dict()
    return clean_data_for_json(missing_values)
