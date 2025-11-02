"""
FastAPI server for ML Model Monitoring Dashboard API.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pickle
import pandas as pd
import numpy as np
import json
import uuid
import tempfile
import os
from datetime import datetime
from typing import Optional, Dict, Any
import base64
import io

# Import sklearn metrics directly
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Import cloud database
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from cloud_database import db_manager

# Helper functions for the API
def load_model_from_file(file_path: str):
    """Load model from file."""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # Fallback to joblib
        import joblib
        return joblib.load(file_path)

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate classification metrics."""
    try:
        results = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted')),
            'recall': float(recall_score(y_true, y_pred, average='weighted')),
            'f1': float(f1_score(y_true, y_pred, average='weighted')),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        return results
    except Exception as e:
        return {"error": str(e)}

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate regression metrics."""
    try:
        results = {
            'r2': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else float('inf')
        }
        return results
    except Exception as e:
        return {"error": str(e)}

app = FastAPI(title="ML Model Monitoring API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove in-memory storage - now using shared storage
# projects_storage = {}
# models_storage = {}
# evaluations_storage = {}

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "ML Model Monitoring API",
        "version": "1.0.0",
        "endpoints": {
            "upload_model": "/api/v1/models/upload",
            "evaluate_model": "/api/v1/models/{model_id}/evaluate",
            "get_results": "/api/v1/evaluations/{evaluation_id}/results",
            "health": "/api/v1/health"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/models/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    model_type: str = Form(...),
    project_code: str = Form(...),  # Use project_code instead of project_name
    description: Optional[str] = Form("")
):
    """Upload a machine learning model."""
    
    try:
        # Validate file type
        if not model_file.filename.endswith('.pkl'):
            raise HTTPException(status_code=400, detail="Only .pkl files are supported")
        
        # Check if project exists
        project_data = db_manager.get_project_by_code(project_code)
        if not project_data:
            raise HTTPException(status_code=404, detail=f"Project code '{project_code}' not found")
        
        # Generate unique model ID
        model_id = f"mdl_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            content = await model_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load and validate model
        try:
            model = load_model_from_file(tmp_file_path)
            
            if not hasattr(model, 'predict'):
                raise HTTPException(status_code=400, detail="Invalid model: no predict method")
                
        except Exception as e:
            os.unlink(tmp_file_path)
            raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")
        
        # Store model metadata
        model_metadata = {
            "model_id": model_id,
            "project_code": project_code,
            "model_type": model_type,
            "description": description,
            "filename": model_file.filename,
            "upload_time": datetime.now().isoformat(),
            "file_path": tmp_file_path,
            "status": "uploaded"
        }
        
        # Save model to MongoDB
        db_manager.save_model(model_metadata)
        
        return {
            "model_id": model_id,
            "project_code": project_code,
            "status": "uploaded",
            "upload_time": model_metadata["upload_time"],
            "dashboard_url": f"http://localhost:8501?project_code={project_code}&model={model_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/models/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    dataset_file: UploadFile = File(...),
    target_column: str = Form(...),
    test_name: Optional[str] = Form("api_evaluation")
):
    """Evaluate a model with provided dataset."""
    
    try:
        # Check if model exists
        model_metadata = db_manager.get_model_by_id(model_id)
        if not model_metadata:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load model
        model = load_model_from_file(model_metadata["file_path"])
        
        # Read dataset
        if not dataset_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported for datasets")
        
        content = await dataset_file.read()
        dataset = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate target column
        if target_column not in dataset.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")
        
        # Prepare features and target
        X = dataset.drop(columns=[target_column])
        y_true = dataset[target_column]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics based on model type
        if model_metadata["model_type"] == "classification":
            metrics = calculate_classification_metrics(y_true, y_pred)
        else:
            metrics = calculate_regression_metrics(y_true, y_pred)
        
        # Generate evaluation ID
        evaluation_id = f"eval_{uuid.uuid4().hex[:8]}"
        
        # Store evaluation results
        evaluation_data = {
            "evaluation_id": evaluation_id,
            "model_id": model_id,
            "project_code": model_metadata["project_code"],
            "test_name": test_name,
            "dataset_info": {
                "shape": list(dataset.shape),
                "columns": list(dataset.columns),
                "target_column": target_column
            },
            "metrics": metrics,
            "predictions": y_pred.tolist() if len(y_pred) < 1000 else y_pred[:1000].tolist(),
            "actual_values": y_true.tolist() if len(y_true) < 1000 else y_true[:1000].tolist(),
            "evaluation_time": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Save to MongoDB
        db_manager.save_evaluation(evaluation_data)
        
        project_code = model_metadata["project_code"]
        
        return {
            "evaluation_id": evaluation_id,
            "model_id": model_id,
            "project_code": project_code,
            "status": "completed",
            "metrics": metrics,
            "dashboard_url": f"http://localhost:8501?project_code={project_code}&evaluation={evaluation_id}",
            "evaluation_time": evaluation_data["evaluation_time"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/v1/evaluations/{evaluation_id}/results")
async def get_evaluation_results(evaluation_id: str):
    """Get detailed evaluation results."""
    
    evaluation_data = db_manager.get_evaluation_by_id(evaluation_id)
    if not evaluation_data:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return evaluation_data

@app.get("/api/v1/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get model information."""
    
    model_data = db_manager.get_model_by_id(model_id)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Remove file path for security
    safe_model_data = model_data.copy()
    safe_model_data.pop("file_path", None)
    
    return safe_model_data

@app.get("/api/v1/projects/{project_code}")
async def get_project_info(project_code: str):
    """Get project information with all models and evaluations."""
    
    project_data = db_manager.get_project_by_code(project_code)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get detailed model info
    detailed_models = db_manager.get_models_by_project(project_code)
    for model in detailed_models:
        model.pop("file_path", None)  # Remove file path for security
    
    # Get detailed evaluation info
    detailed_evaluations = db_manager.get_evaluations_by_project(project_code)
    for evaluation in detailed_evaluations:
        # Limit prediction data for API response
        if evaluation.get("predictions") and len(evaluation["predictions"]) > 10:
            evaluation["predictions"] = evaluation["predictions"][:10]
        if evaluation.get("actual_values") and len(evaluation["actual_values"]) > 10:
            evaluation["actual_values"] = evaluation["actual_values"][:10]
    
    response_data = project_data.copy()
    response_data["models"] = detailed_models
    response_data["evaluations"] = detailed_evaluations
    
    return response_data

@app.get("/api/v1/projects")
async def list_projects():
    """List all projects."""
    
    all_projects = db_manager.get_all_projects()
    projects_list = []
    
    for project_data in all_projects:
        project_code = project_data["project_code"]
        models = db_manager.get_models_by_project(project_code)
        evaluations = db_manager.get_evaluations_by_project(project_code)
        
        summary = {
            "project_code": project_code,
            "project_name": project_data["project_name"],
            "created_at": project_data["created_at"],
            "model_count": len(models),
            "evaluation_count": len(evaluations),
            "dashboard_url": f"http://localhost:8501?project_code={project_code}"
        }
        projects_list.append(summary)
    
    return {"projects": projects_list}

@app.post("/api/v1/projects")
async def create_project(project_name: str = Form(...), description: str = Form(""), github_repo: str = Form("")):
    """Create a new project and get a unique project code."""
    
    try:
        project_data = db_manager.create_project(project_name, description, github_repo)
        return {
            "project_code": project_data["project_code"],
            "project_name": project_data["project_name"],
            "created_at": project_data["created_at"],
            "dashboard_url": f"http://localhost:8501?project_code={project_data['project_code']}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

if __name__ == "__main__":
    print("Starting ML Model Monitoring API server...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)