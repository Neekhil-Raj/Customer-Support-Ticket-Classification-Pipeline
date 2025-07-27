import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import re

class CustomerSupportDataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_and_validate_csv(self, file_path: str) -> pd.DataFrame:
        """Load and validate the customer support tickets CSV"""
        try:
            # Load the CSV
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            
            # Validate required columns
            required_columns = [
                'Ticket ID', 'Customer Name', 'Customer Email', 'Customer Age',
                'Customer Gender', 'Product Purchased', 'Date of Purchase',
                'Ticket Type', 'Ticket Subject', 'Ticket Description',
                'Ticket Status', 'Resolution', 'Ticket Priority', 'Ticket Channel',
                'First Response Time', 'Time to Resolution', 'Customer Satisfaction Rating'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Log basic statistics
            self.logger.info(f"Dataset shape: {df.shape}")
            self.logger.info(f"Ticket Types: {df['Ticket Type'].nunique()} unique values")
            self.logger.info(f"Date range: {df['Date of Purchase'].min()} to {df['Date of Purchase'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
    
    def clean_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        df_cleaned = df.copy()
        
        # Standardize column names (remove spaces, lowercase)
        column_mapping = {
            'Ticket ID': 'ticket_id',
            'Customer Name': 'customer_name',
            'Customer Email': 'customer_email',
            'Customer Age': 'customer_age',
            'Customer Gender': 'customer_gender',
            'Product Purchased': 'product_purchased',
            'Date of Purchase': 'date_of_purchase',
            'Ticket Type': 'ticket_type',
            'Ticket Subject': 'ticket_subject',
            'Ticket Description': 'ticket_description',
            'Ticket Status': 'ticket_status',
            'Resolution': 'resolution',
            'Ticket Priority': 'ticket_priority',
            'Ticket Channel': 'ticket_channel',
            'First Response Time': 'first_response_time',
            'Time to Resolution': 'time_to_resolution',
            'Customer Satisfaction Rating': 'customer_satisfaction_rating'
        }
        
        df_cleaned = df_cleaned.rename(columns=column_mapping)
        
        # Handle missing values
        df_cleaned['ticket_description'] = df_cleaned['ticket_description'].fillna('')
        df_cleaned['ticket_subject'] = df_cleaned['ticket_subject'].fillna('')
        df_cleaned['resolution'] = df_cleaned['resolution'].fillna('')
        df_cleaned['customer_age'] = pd.to_numeric(df_cleaned['customer_age'], errors='coerce')
        
        # Clean and standardize text fields
        df_cleaned['ticket_subject'] = df_cleaned['ticket_subject'].astype(str).str.strip()
        df_cleaned['ticket_description'] = df_cleaned['ticket_description'].astype(str).str.strip()
        
        # Create combined text for NLP processing
        df_cleaned['combined_text'] = (
            df_cleaned['ticket_subject'] + ' ' + df_cleaned['ticket_description']
        ).str.strip()
        
        # Convert date columns
        df_cleaned['date_of_purchase'] = pd.to_datetime(
            df_cleaned['date_of_purchase'], errors='coerce'
        )
        
        # Standardize categorical variables
        df_cleaned['ticket_priority'] = df_cleaned['ticket_priority'].str.title()
        df_cleaned['ticket_status'] = df_cleaned['ticket_status'].str.title()
        df_cleaned['ticket_channel'] = df_cleaned['ticket_channel'].str.title()
        df_cleaned['customer_gender'] = df_cleaned['customer_gender'].str.title()
        
        # Handle numeric fields
        df_cleaned['first_response_time'] = pd.to_numeric(
            df_cleaned['first_response_time'], errors='coerce'
        )
        df_cleaned['time_to_resolution'] = pd.to_numeric(
            df_cleaned['time_to_resolution'], errors='coerce'
        )
        df_cleaned['customer_satisfaction_rating'] = pd.to_numeric(
            df_cleaned['customer_satisfaction_rating'], errors='coerce'
        )
        
        # Remove rows with critical missing data
        df_cleaned = df_cleaned.dropna(subset=['ticket_id', 'ticket_type'])
        
        self.logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        
        return df_cleaned
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features for ML"""
        df_features = df.copy()
        
        # Text-based features
        df_features['subject_length'] = df_features['ticket_subject'].str.len()
        df_features['description_length'] = df_features['ticket_description'].str.len()
        df_features['combined_text_length'] = df_features['combined_text'].str.len()
        df_features['word_count'] = df_features['combined_text'].str.split().str.len()
        
        # Priority encoding
        priority_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df_features['priority_numeric'] = df_features['ticket_priority'].map(priority_mapping)
        
        # Time-based features
        if 'date_of_purchase' in df_features.columns:
            df_features['days_since_purchase'] = (
                datetime.now() - df_features['date_of_purchase']
            ).dt.days
            
            df_features['purchase_year'] = df_features['date_of_purchase'].dt.year
            df_features['purchase_month'] = df_features['date_of_purchase'].dt.month
            df_features['purchase_quarter'] = df_features['date_of_purchase'].dt.quarter
        
        # Customer features
        df_features['has_resolution'] = (
            df_features['resolution'].notna() & (df_features['resolution'] != '')
        ).astype(int)
        
        # Email domain extraction
        df_features['email_domain'] = df_features['customer_email'].str.split('@').str[1]
        df_features['is_business_email'] = df_features['email_domain'].str.contains(
            'gmail|yahoo|hotmail|outlook', case=False, na=False
        ) == False
        
        # Urgency indicators in text
        urgency_words = ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'help']
        df_features['has_urgency_keywords'] = df_features['combined_text'].str.lower().str.contains(
            '|'.join(urgency_words), na=False
        ).astype(int)
        
        self.logger.info("Feature generation completed")
        
        return df_features
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data statistics"""
        stats = {
            'basic_info': {
                'total_tickets': len(df),
                'date_range': {
                    'start': df['date_of_purchase'].min(),
                    'end': df['date_of_purchase'].max()
                },
                'missing_values': df.isnull().sum().to_dict()
            },
            'ticket_distribution': {
                'by_type': df['ticket_type'].value_counts().to_dict(),
                'by_priority': df['ticket_priority'].value_counts().to_dict(),
                'by_status': df['ticket_status'].value_counts().to_dict(),
                'by_channel': df['ticket_channel'].value_counts().to_dict()
            },
            'customer_demographics': {
                'age_distribution': {
                    'mean': df['customer_age'].mean(),
                    'median': df['customer_age'].median(),
                    'std': df['customer_age'].std()
                },
                'gender_distribution': df['customer_gender'].value_counts().to_dict()
            },
            'performance_metrics': {
                'avg_response_time': df['first_response_time'].mean(),
                'avg_resolution_time': df['time_to_resolution'].mean(),
                'avg_satisfaction': df['customer_satisfaction_rating'].mean()
            },
            'text_statistics': {
                'avg_subject_length': df['ticket_subject'].str.len().mean(),
                'avg_description_length': df['ticket_description'].str.len().mean(),
                'avg_word_count': df['combined_text'].str.split().str.len().mean()
            }
        }
        
        return stats
