import pandas as pd
import numpy as np
import io
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self):
        self.uploaded_data = {}
    
    async def process_csv_upload(self, file):
        """Process uploaded CSV file and validate data"""
        logger.info(f"Processing file: {file.filename}")
        
        try:
            if not file.filename or not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="File harus berformat CSV")
            
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            df = df.replace([np.inf, -np.inf], np.nan)
            
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_columns) == 0:
                raise HTTPException(status_code=400, detail="File CSV harus memiliki minimal 1 kolom numerik")
            
            self.uploaded_data['dataframe'] = df
            self.uploaded_data['filename'] = file.filename
            
            return df, numeric_columns
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    def get_dataframe(self, preprocessed=False):
        """Get stored dataframe"""
        key = 'preprocessed_dataframe' if preprocessed else 'dataframe'
        if key not in self.uploaded_data:
            raise HTTPException(status_code=404, detail="No data found")
        return self.uploaded_data[key]
    
    def store_preprocessed_data(self, df):
        """Store preprocessed dataframe"""
        self.uploaded_data['preprocessed_dataframe'] = df
        self.uploaded_data['preprocessing_applied'] = True
