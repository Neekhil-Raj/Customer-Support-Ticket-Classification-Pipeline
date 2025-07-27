from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import pandas as pd
import logging

# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'ticket_classification_pipeline',
    default_args=default_args,
    description='Customer Support Ticket Classification Pipeline',
    schedule_interval=timedelta(hours=6),
    catchup=False,
    tags=['ml', 'tickets', 'classification']
)

def extract_and_preprocess_data(**context):
    """Extract and preprocess ticket data"""
    from src.data_ingestion.etl_pipeline import ETLPipeline
    from src.preprocessing.text_processor import TextPreprocessor
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        etl = ETLPipeline()
        text_processor = TextPreprocessor()
        
        # Extract data from database
        # This would typically query your data source
        logger.info("Extracting ticket data...")
        
        # For demo, using a placeholder
        # df = etl.extract_from_database()
        # processed_df = text_processor.preprocess_for_ml(df, 'combined_text', 'ticket_type')
        
        logger.info("Data extraction and preprocessing completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in data extraction: {e}")
        raise

def train_classification_models(**context):
    """Train and evaluate classification models"""
    from src.models.ml_models import MLModelTrainer
    from src.pipeline.mlflow_tracking import MLflowTracker
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model training...")
        
        # Initialize trainer
        trainer = MLModelTrainer()
        mlflow_tracker = MLflowTracker()
        
        # Load preprocessed data
        # X_train, X_test, y_train, y_test = load_preprocessed_data()
        
        # Train models
        # results = trainer.train_and_evaluate_all_models(X_train, y_train, X_test, y_test)
        
        # Log results to MLflow
        # mlflow_tracker.log_model_results(results)
        
        logger.info("Model training completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def deploy_best_model(**context):
    """Deploy the best performing model"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Deploying best model...")
        
        # Load best model from MLflow
        # Deploy to production endpoint
        
        logger.info("Model deployment completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in model deployment: {e}")
        raise

def validate_model_performance(**context):
    """Validate deployed model performance"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Validating model performance...")
        
        # Run validation tests
        # Check performance metrics
        
        logger.info("Model validation completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in model validation: {e}")
        raise

# Define tasks
data_sensor = FileSensor(
    task_id='wait_for_data',
    filepath='/data/raw/tickets_*.csv',
    fs_conn_id='fs_default',
    poke_interval=300,
    timeout=3600,
    dag=dag
)

extract_task = PythonOperator(
    task_id='extract_and_preprocess',
    python_callable=extract_and_preprocess_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_classification_models,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_best_model,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model_performance,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='rm -rf /tmp/model_training_*',
    dag=dag
)

# Define task dependencies
data_sensor >> extract_task >> train_task >> deploy_task >> validate_task >> cleanup_task
