from fastapi import APIRouter, HTTPException, File, UploadFile
import pandas as pd
from io import StringIO
from src.data_ingestion.csv_data_loader import CustomerSupportDataLoader
from src.pipeline.training_pipeline_actual import ActualDataTrainingPipeline

router = APIRouter()

@router.post("/data/upload")
async def upload_csv_data(file: UploadFile = File(...)):
    """Upload and validate CSV data"""
    try:
        # Read the uploaded file
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        
        # Create DataFrame
        df = pd.read_csv(StringIO(csv_string))
        
        # Validate and process
        loader = CustomerSupportDataLoader()
        df_processed = loader.clean_and_preprocess(df)
        stats = loader.get_data_statistics(df_processed)
        
        return {
            "message": "CSV uploaded and processed successfully",
            "records_count": len(df_processed),
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@router.post("/train/pipeline")
async def trigger_training_pipeline():
    """Trigger the complete training pipeline"""
    try:
        pipeline = ActualDataTrainingPipeline('customer_support_tickets.csv')
        results = pipeline.run_complete_pipeline()
        
        return {
            "message": "Training pipeline completed successfully",
            "ml_model_count": len(results['ml_results']),
            "best_ml_f1": max([r['metrics']['f1_score'] for r in results['ml_results'].values()]),
            "dl_accuracy": results['dl_results']['test_accuracy']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/data/statistics")
async def get_data_statistics():
    """Get comprehensive data statistics"""
    try:
        loader = CustomerSupportDataLoader()
        df = loader.load_and_validate_csv('customer_support_tickets.csv')
        df_processed = loader.clean_and_preprocess(df)
        stats = loader.get_data_statistics(df_processed)
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")
