import numpy as np
import pandas as pd

def safe_json_serializer(obj):
    """Convert numpy/pandas objects to JSON-serializable format"""
    if isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def serialize_dataframe_preview(df, num_rows=5):
    """Safely serialize DataFrame preview for JSON response"""
    preview_data = df.head(num_rows).fillna("").to_dict('records')
    safe_preview = []
    for row in preview_data:
        safe_row = {}
        for key, value in row.items():
            safe_row[key] = safe_json_serializer(value)
        safe_preview.append(safe_row)
    return safe_preview

def serialize_missing_values(df):
    """Safely serialize missing values statistics"""
    missing_values = df.isnull().sum().to_dict()
    return {k: safe_json_serializer(v) for k, v in missing_values.items()}
