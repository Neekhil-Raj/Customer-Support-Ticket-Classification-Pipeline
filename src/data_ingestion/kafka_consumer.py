from kafka import KafkaConsumer
import json
import logging
from typing import Dict, Any
from config.config import Config
from src.data_ingestion.etl_pipeline import ETLPipeline

class KafkaTicketConsumer:
    def __init__(self):
        self.config = Config()
        self.consumer = None
        self.etl_pipeline = ETLPipeline()
        self.logger = logging.getLogger(__name__)
        
    def create_consumer(self):
        """Create Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.config.kafka.topic_name,
                bootstrap_servers=self.config.kafka.bootstrap_servers,
                group_id=self.config.kafka.group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            self.logger.info("Kafka consumer created successfully")
        except Exception as e:
            self.logger.error(f"Error creating Kafka consumer: {e}")
    
    def process_message(self, message: Dict[str, Any]):
        """Process individual ticket message"""
        try:
            # Extract ticket data
            ticket_data = message
            
            # Transform and validate
            # Add your transformation logic here
            
            # Store in database
            # Add database insertion logic here
            
            self.logger.info(f"Processed ticket: {ticket_data.get('ticket_id')}")
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def start_consuming(self):
        """Start consuming messages from Kafka"""
        self.create_consumer()
        
        if self.consumer:
            try:
                for message in self.consumer:
                    self.process_message(message.value)
            except KeyboardInterrupt:
                self.logger.info("Consumer stopped by user")
            finally:
                self.consumer.close()
