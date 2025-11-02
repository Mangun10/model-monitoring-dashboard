"""
Shared data storage for ML Model Monitoring Dashboard.
This module handles persistent storage that can be accessed by both
the Streamlit dashboard and the API server.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

class SharedDataStorage:
    """Shared storage system for projects, models, and evaluations."""
    
    def __init__(self, data_dir: str = "shared_data"):
        self.data_dir = data_dir
        self.projects_file = os.path.join(data_dir, "projects.json")
        self.models_file = os.path.join(data_dir, "models.json")
        self.evaluations_file = os.path.join(data_dir, "evaluations.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize storage files
        self._init_storage_files()
    
    def _init_storage_files(self):
        """Initialize storage files if they don't exist."""
        for file_path in [self.projects_file, self.models_file, self.evaluations_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump({}, f)
    
    def _load_json(self, file_path: str) -> Dict:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_json(self, file_path: str, data: Dict):
        """Save data to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # Project methods
    def save_project(self, project_data: Dict[str, Any]):
        """Save project data."""
        projects = self._load_json(self.projects_file)
        projects[project_data["project_name"]] = project_data
        self._save_json(self.projects_file, projects)
    
    def get_project(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get project data."""
        projects = self._load_json(self.projects_file)
        return projects.get(project_name)
    
    def get_all_projects(self) -> Dict[str, Any]:
        """Get all projects."""
        return self._load_json(self.projects_file)
    
    # Model methods
    def save_model(self, model_data: Dict[str, Any]):
        """Save model data."""
        models = self._load_json(self.models_file)
        models[model_data["model_id"]] = model_data
        self._save_json(self.models_file, models)
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model data."""
        models = self._load_json(self.models_file)
        return models.get(model_id)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all models."""
        return self._load_json(self.models_file)
    
    # Evaluation methods
    def save_evaluation(self, evaluation_data: Dict[str, Any]):
        """Save evaluation data."""
        evaluations = self._load_json(self.evaluations_file)
        evaluations[evaluation_data["evaluation_id"]] = evaluation_data
        self._save_json(self.evaluations_file, evaluations)
    
    def get_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation data."""
        evaluations = self._load_json(self.evaluations_file)
        return evaluations.get(evaluation_id)
    
    def get_all_evaluations(self) -> Dict[str, Any]:
        """Get all evaluations."""
        return self._load_json(self.evaluations_file)
    
    def get_evaluations_for_project(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a project."""
        evaluations = self._load_json(self.evaluations_file)
        project_evaluations = []
        
        for eval_data in evaluations.values():
            # Check if evaluation belongs to this project
            model_id = eval_data.get("model_id")
            if model_id:
                model_data = self.get_model(model_id)
                if model_data and model_data.get("project_name") == project_name:
                    project_evaluations.append(eval_data)
        
        return project_evaluations
    
    def get_models_for_project(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all models for a project."""
        models = self._load_json(self.models_file)
        project_models = []
        
        for model_data in models.values():
            if model_data.get("project_name") == project_name:
                project_models.append(model_data)
        
        return project_models

# Global instance
shared_storage = SharedDataStorage()