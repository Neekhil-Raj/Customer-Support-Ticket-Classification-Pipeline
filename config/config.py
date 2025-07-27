import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "password")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "support_tickets")
    
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "tickets_db")

@dataclass
class ModelConfig:
    model_name: str = "ticket_classifier"
    max_features: int = 10000
    embedding_dim: int = 300
    lstm_units: int = 128
    dropout_rate: float = 0.3
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_name: str = "support_tickets"
    group_id: str = "ticket_processor"

class Config:
    database = DatabaseConfig()
    model = ModelConfig()
    api = APIConfig()
    kafka = KafkaConfig()
    
    # MLflow settings
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name: str = "ticket_classification"
