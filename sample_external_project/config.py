"""
Configuration settings for the sample external project.
"""

# API Configuration
API_BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"

# Default Project Configuration
DEFAULT_PROJECT_NAME = "Sample_ML_Project"
MODEL_TYPE = "classification"  # or "regression"

# Model Training Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# API Endpoints
ENDPOINTS = {
    "create_project": f"{API_BASE_URL}/api/v1/projects",
    "upload_model": f"{API_BASE_URL}/api/v1/models/upload",
    "evaluate_model": f"{API_BASE_URL}/api/v1/models/{{model_id}}/evaluate",
    "get_results": f"{API_BASE_URL}/api/v1/evaluations/{{evaluation_id}}/results",
    "get_model_info": f"{API_BASE_URL}/api/v1/models/{{model_id}}/info",
    "get_project": f"{API_BASE_URL}/api/v1/projects/{{project_code}}",
    "list_projects": f"{API_BASE_URL}/api/v1/projects"
}