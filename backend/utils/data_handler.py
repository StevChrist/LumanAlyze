import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
import io
import logging
import csv
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataHandler:
    """Class untuk menangani upload dan validasi data CSV dengan advanced parsing"""
    
    def __init__(self):
        self.uploaded_data = {}
        self.supported_encodings = ['utf-8', 'latin1', 'windows-1252', 'iso-8859-1', 'cp1252']
        self.common_delimiters = [',', ';', '\t', '|', ' ']
    
    def detect_delimiter(self, sample_text: str) -> str:
        """Detect delimiter using csv.Sniffer and fallback methods"""
        try:
            # Try csv.Sniffer first
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text[:1024])
            return dialect.delimiter
        except:
            # Fallback: count occurrences of common delimiters
            delimiter_counts = {}
            lines = sample_text.split('\n')[:5]  # Check first 5 lines
            
            for delimiter in self.common_delimiters:
                counts = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        counts.append(line.count(delimiter))
                
                # Check if delimiter appears consistently
                if counts and len(set(counts)) <= 2:  # Allow some variation
                    delimiter_counts[delimiter] = max(counts)
            
            if delimiter_counts:
                return max(delimiter_counts.items(), key=lambda x: x[1])[0]
            
            return ','  # Default fallback
    
    async def process_csv_upload(self, file: UploadFile) -> Tuple[pd.DataFrame, List[str]]:
        """Process uploaded CSV file dengan advanced parsing dan error handling"""
        logger.info(f"Processing file: {file.filename}")
        
        try:
            # Read file content
            content = await file.read()
            
            # Try different encodings
            df = None
            encoding_used = None
            delimiter_used = None
            
            for encoding in self.supported_encodings:
                try:
                    # Decode content with specific encoding
                    content_str = content.decode(encoding)
                    
                    # Detect delimiter
                    delimiter = self.detect_delimiter(content_str)
                    
                    # Try reading with detected delimiter and advanced options
                    df = pd.read_csv(
                        io.StringIO(content_str),
                        sep=delimiter,
                        engine='python',  # More robust parser
                        on_bad_lines='skip',  # Skip problematic lines
                        skipinitialspace=True,  # Handle extra spaces
                        quotechar='"',  # Handle quoted fields
                        doublequote=True,  # Handle double quotes
                        encoding_errors='replace'  # Replace bad characters
                    )
                    
                    encoding_used = encoding
                    delimiter_used = delimiter
                    logger.info(f"Successfully read file with encoding: {encoding}, delimiter: '{delimiter}'")
                    break
                    
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed to read with encoding {encoding}: {str(e)}")
                    continue
                except Exception as e:
                    logger.warning(f"Error reading with encoding {encoding}: {str(e)}")
                    continue
            
            if df is None:
                # Last resort: try with different approach
                try:
                    # Try with most permissive settings
                    content_str = content.decode('utf-8', errors='replace')
                    
                    # Try each delimiter explicitly
                    for delimiter in self.common_delimiters:
                        try:
                            df = pd.read_csv(
                                io.StringIO(content_str),
                                sep=delimiter,
                                engine='python',
                                on_bad_lines='skip',
                                header=None,  # No header assumption
                                skipinitialspace=True,
                                quotechar='"',
                                encoding_errors='replace'
                            )
                            
                            # Check if we got reasonable data
                            if len(df) > 0 and len(df.columns) > 1:
                                encoding_used = 'utf-8 (with errors replaced)'
                                delimiter_used = delimiter
                                logger.info(f"Successfully read with fallback method, delimiter: '{delimiter}'")
                                
                                # Add generic column names
                                df.columns = [f'column_{i+1}' for i in range(len(df.columns))]
                                break
                                
                        except Exception:
                            continue
                            
                except Exception as e:
                    logger.error(f"All parsing attempts failed: {str(e)}")
            
            if df is None or df.empty:
                raise HTTPException(
                    status_code=400, 
                    detail="Unable to read file. The file may be corrupted, have an unsupported format, or contain inconsistent data structure."
                )
            
            # Validate DataFrame
            if len(df.columns) < 1:
                raise HTTPException(status_code=400, detail="File must have at least 1 column")
            
            # Clean column names
            df.columns = df.columns.astype(str)
            df.columns = [col.strip() for col in df.columns]
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # If no numeric columns, try to convert some columns
            if len(numeric_columns) == 0:
                for col in df.columns:
                    try:
                        # Try to convert to numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if not df[col].isna().all():  # If conversion was successful for some values
                            numeric_columns.append(col)
                    except:
                        continue
            
            if len(numeric_columns) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="File must contain at least one numeric column for machine learning analysis"
                )
            
            # Store data
            self.uploaded_data = {
                'filename': file.filename,
                'dataframe': df,
                'numeric_columns': numeric_columns,
                'encoding_used': encoding_used,
                'delimiter_used': delimiter_used,
                'preprocessing_applied': False
            }
            
            logger.info(f"File processed successfully: {len(df)} rows, {len(df.columns)} columns, encoding: {encoding_used}, delimiter: '{delimiter_used}'")
            return df, numeric_columns
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    def get_dataframe(self, preprocessed: bool = False) -> pd.DataFrame:
        """Get stored dataframe"""
        if preprocessed and 'preprocessed_dataframe' in self.uploaded_data:
            return self.uploaded_data['preprocessed_dataframe']
        elif 'dataframe' in self.uploaded_data:
            return self.uploaded_data['dataframe']
        else:
            raise HTTPException(status_code=404, detail="No data found. Please upload a file first.")
    
    def store_preprocessed_data(self, df: pd.DataFrame):
        """Store preprocessed dataframe"""
        self.uploaded_data['preprocessed_dataframe'] = df
        self.uploaded_data['preprocessing_applied'] = True
        logger.info("Preprocessed data stored successfully")
    
    def get_file_info(self) -> dict:
        """Get information about uploaded file"""
        if 'dataframe' not in self.uploaded_data:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = self.uploaded_data['dataframe']
        return {
            'filename': self.uploaded_data['filename'],
            'encoding_used': self.uploaded_data.get('encoding_used', 'unknown'),
            'delimiter_used': self.uploaded_data.get('delimiter_used', 'unknown'),
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'numeric_columns': self.uploaded_data['numeric_columns'],
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
