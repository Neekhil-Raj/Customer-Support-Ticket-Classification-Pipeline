import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime
from config.database import DatabaseManager

class ETLPipeline:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
    def extract_from_csv(self, file_path: str) -> pd.DataFrame:
        """Extract data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Extracted {len(df)} records from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            return pd.DataFrame()
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform and clean the data"""
        # Handle missing values
        df['ticket_description'] = df['ticket_description'].fillna('')
        df['ticket_subject'] = df['ticket_subject'].fillna('')
        df['resolution'] = df['resolution'].fillna('')
        
        # Convert date columns
        df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'], errors='coerce')
        df['first_response_time'] = pd.to_datetime(df['first_response_time'], errors='coerce')
        
        # Create combined text for NLP processing
        df['combined_text'] = df['ticket_subject'] + ' ' + df['ticket_description']
        
        # Handle categorical variables
        df['ticket_priority'] = df['ticket_priority'].fillna('Medium')
        df['ticket_status'] = df['ticket_status'].fillna('Open')
        
        # Data validation
        df = df[df['ticket_id'].notna()]
        
        self.logger.info(f"Transformed data shape: {df.shape}")
        return df
    
    def load_to_mysql(self, df: pd.DataFrame) -> bool:
        """Load data to MySQL database"""
        try:
            conn = self.db_manager.connect_mysql()
            if conn:
                df.to_sql('support_tickets', conn, if_exists='append', index=False, method='multi')
                self.logger.info(f"Loaded {len(df)} records to MySQL")
                return True
        except Exception as e:
            self.logger.error(f"Error loading to MySQL: {e}")
            return False
    
    def load_to_mongodb(self, df: pd.DataFrame) -> bool:
        """Load semi-structured data to MongoDB"""
        try:
            db = self.db_manager.connect_mongodb()
            if db:
                collection = db['tickets']
                records = df.to_dict('records')
                collection.insert_many(records)
                self.logger.info(f"Loaded {len(records)} records to MongoDB")
                return True
        except Exception as e:
            self.logger.error(f"Error loading to MongoDB: {e}")
            return False
    
    def run_pipeline(self, file_path: str) -> bool:
        """Run the complete ETL pipeline"""
        # Extract
        df = self.extract_from_csv(file_path)
        if df.empty:
            return False
        
        # Transform
        df_transformed = self.transform_data(df)
        
        # Load
        mysql_success = self.load_to_mysql(df_transformed)
        mongodb_success = self.load_to_mongodb(df_transformed)
        
        return mysql_success and mongodb_success
