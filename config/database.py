import mysql.connector
from pymongo import MongoClient
import pandas as pd
from typing import Optional, Dict, Any
from config.config import Config

class DatabaseManager:
    def __init__(self):
        self.config = Config()
        self.mysql_conn = None
        self.mongo_client = None
        
    def connect_mysql(self):
        """Establish MySQL connection"""
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.config.database.mysql_host,
                port=self.config.database.mysql_port,
                user=self.config.database.mysql_user,
                password=self.config.database.mysql_password,
                database=self.config.database.mysql_database
            )
            return self.mysql_conn
        except Exception as e:
            print(f"MySQL connection failed: {e}")
            return None
    
    def connect_mongodb(self):
        """Establish MongoDB connection"""
        try:
            self.mongo_client = MongoClient(self.config.database.mongodb_uri)
            return self.mongo_client[self.config.database.mongodb_database]
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return None
    
    def create_mysql_tables(self):
        """Create MySQL tables for structured data"""
        cursor = self.mysql_conn.cursor()
        
        create_tickets_table = """
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id VARCHAR(50) PRIMARY KEY,
            customer_name VARCHAR(255),
            customer_email VARCHAR(255),
            customer_age INT,
            customer_gender VARCHAR(10),
            product_purchased VARCHAR(255),
            date_of_purchase DATE,
            ticket_type VARCHAR(100),
            ticket_subject TEXT,
            ticket_description LONGTEXT,
            ticket_status VARCHAR(50),
            resolution TEXT,
            ticket_priority VARCHAR(20),
            ticket_channel VARCHAR(50),
            first_response_time TIMESTAMP,
            time_to_resolution INT,
            customer_satisfaction_rating FLOAT,
            predicted_category VARCHAR(100),
            confidence_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_tickets_table)
        self.mysql_conn.commit()
        cursor.close()
