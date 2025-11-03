"""
API client utilities for communicating with the ML Model Monitoring Dashboard.
"""

import requests
import json
import time
from typing import Dict, Optional, Any
from config import ENDPOINTS

class DashboardAPIClient:
    """Client for ML Model Monitoring Dashboard API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_project(self, project_name: str, description: str = "", github_repo: str = "") -> Dict[str, Any]:
        """
        Create a new project and return the project code.
        
        Args:
            project_name (str): Name of the project
            description (str): Optional project description
            github_repo (str): Optional GitHub repository URL
            
        Returns:
            Dict containing project_code and project details
        """
        try:
            # Send as form data (multipart/form-data) as expected by the API
            data = {
                'project_name': project_name,
                'description': description,
                'github_repo': github_repo
            }
            
            response = self.session.post(ENDPOINTS["create_project"], data=data)
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Project created successfully!")
            print(f"   Project Code: {result['project_code']}")
            print(f"   Project Name: {result['project_name']}")
            print(f"   Dashboard URL: http://localhost:8501/?project_code={result['project_code']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create project: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            raise
    
    def upload_model(self, 
                    model_path: str, 
                    model_type: str,
                    project_code: str,
                    model_name: str,
                    model_version: str = "1.0",
                    description: str = "") -> Dict[str, Any]:
        """
        Upload a trained model to the dashboard.
        
        Args:
            model_path (str): Path to the .pkl model file
            model_type (str): Type of model ('classification' or 'regression')
            project_code (str): Project code to upload to
            model_name (str): Name for the model
            model_version (str): Version of the model
            description (str): Optional description
            
        Returns:
            Dict containing model_id and upload status
        """
        try:
            with open(model_path, 'rb') as model_file:
                files = {'model_file': model_file}
                data = {
                    'project_code': project_code,
                    'model_name': model_name,
                    'model_version': model_version,
                    'model_type': model_type,
                    'description': description
                }
                
                response = self.session.post(
                    ENDPOINTS["upload_model"],
                    files=files,
                    data=data
                )
                
                response.raise_for_status()
                result = response.json()
                
                print(f"✅ Model uploaded successfully!")
                print(f"   Model ID: {result['model_id']}")
                print(f"   Dashboard URL: http://localhost:8501/?project_code={project_code}")
                
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to upload model: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            raise
    
    def evaluate_model(self, 
                      model_id: str, 
                      dataset_path: str, 
                      target_column: str,
                      test_name: str = "api_evaluation") -> Dict[str, Any]:
        """
        Evaluate a model with a test dataset.
        
        Args:
            model_id (str): ID of the uploaded model
            dataset_path (str): Path to the test dataset CSV
            target_column (str): Name of the target column
            test_name (str): Name for this evaluation
            
        Returns:
            Dict containing evaluation results
        """
        try:
            with open(dataset_path, 'rb') as dataset_file:
                files = {'dataset_file': dataset_file}
                data = {
                    'target_column': target_column,
                    'test_name': test_name
                }
                
                url = ENDPOINTS["evaluate_model"].format(model_id=model_id)
                response = self.session.post(url, files=files, data=data)
                
                response.raise_for_status()
                result = response.json()
                
                print(f"✅ Model evaluation completed!")
                print(f"   Evaluation ID: {result['evaluation_id']}")
                print(f"   Metrics: {json.dumps(result['metrics'], indent=2)}")
                print(f"   Dashboard URL: {result['dashboard_url']}")
                
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to evaluate model: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            raise
    
    def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get detailed evaluation results.
        
        Args:
            evaluation_id (str): ID of the evaluation
            
        Returns:
            Dict containing detailed results
        """
        try:
            url = ENDPOINTS["get_results"].format(evaluation_id=evaluation_id)
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get results: {str(e)}")
            raise
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            Dict containing model information
        """
        try:
            url = ENDPOINTS["get_model_info"].format(model_id=model_id)
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get model info: {str(e)}")
            raise
    
    def get_project_info(self, project_code: str) -> Dict[str, Any]:
        """
        Get project information with all models and evaluations.
        
        Args:
            project_code (str): Code of the project
            
        Returns:
            Dict containing project information
        """
        try:
            url = ENDPOINTS["get_project"].format(project_code=project_code)
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get project info: {str(e)}")
            raise
    
    def list_projects(self) -> Dict[str, Any]:
        """
        List all projects in the dashboard.
        
        Returns:
            Dict containing list of projects
        """
        try:
            response = self.session.get(ENDPOINTS["list_projects"])
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to list projects: {str(e)}")
            raise
    
    def wait_for_api(self, timeout: int = 30) -> bool:
        """
        Wait for the API server to be available.
        
        Args:
            timeout (int): Maximum time to wait in seconds
            
        Returns:
            bool: True if API is available, False otherwise
        """
        print("🔄 Waiting for API server to be available...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/api/v1/health")
                if response.status_code == 200:
                    print("✅ API server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        print("❌ API server is not responding!")
        return False

# Example usage functions
def demo_api_usage():
    """Demonstrate API usage."""
    client = DashboardAPIClient()
    
    # Wait for API to be ready
    if not client.wait_for_api():
        print("Please start the API server first: python api_server.py")
        return
    
    try:
        # Create a new project
        print("\n🆕 Creating a new demo project:")
        project = client.create_project(
            project_name="Demo Project",
            description="A demonstration project created via API"
        )
        project_code = project['project_code']
        
        # List existing projects
        print("\n📋 Listing existing projects:")
        projects = client.list_projects()
        for project in projects.get('projects', []):
            print(f"   - {project['project_name']} (Code: {project['project_code']}, {project['model_count']} models)")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")

if __name__ == "__main__":
    demo_api_usage()