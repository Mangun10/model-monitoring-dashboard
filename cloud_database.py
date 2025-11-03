"""
Simplified Database Configuration for Cloud Deployment
Shared MongoDB Atlas Database - All users connect to the same cloud database
No authentication required - only project code-based access
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MongoDB Configuration - SHARED CLOUD DATABASE
from pymongo import MongoClient
from bson import ObjectId

# Use shared cloud MongoDB Atlas database
MONGODB_URL = os.getenv("MONGODB_URL")

if not MONGODB_URL:
    raise ValueError("❌ MONGODB_URL environment variable is required! Please check your .env file.")

try:
    client = MongoClient(MONGODB_URL)
    # Test the connection
    client.admin.command('ping')
    db = client['ml_monitoring']  # Shared database for all users
    
    # Collections - shared across all users
    projects_collection = db['projects']
    models_collection = db['models']
    evaluations_collection = db['evaluations']
    
    # Create indexes for faster queries (safe to run multiple times)
    try:
        projects_collection.create_index("project_code", unique=True)
        models_collection.create_index("model_id", unique=True)
        evaluations_collection.create_index("evaluation_id", unique=True)
        models_collection.create_index("project_code")
        evaluations_collection.create_index("project_code")
    except Exception:
        pass  # Indexes may already exist
    
    print("✅ Connected to shared MongoDB Atlas database")
    print(f"📊 Database: {db.name}")
    
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {str(e)}")
    print(f"🔍 Connection string: {MONGODB_URL[:50]}...")
    raise


def generate_project_code() -> str:
    """Generate a unique 8-character project code."""
    import secrets
    import string
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(8))


# ========================================
# Database Operations - SHARED CLOUD DATABASE
# ========================================

class DatabaseManager:
    """Unified database manager for shared MongoDB Atlas cloud database."""
    
    def __init__(self):
        self.db_type = "mongodb"  # Always use MongoDB cloud database
    
    # PROJECT OPERATIONS
    def create_project(self, project_name: str, description: str = "", github_repo: str = "") -> Dict[str, Any]:
        """Create a new project with unique code in shared cloud database."""
        project_code = generate_project_code()
        
        project_data = {
            "project_code": project_code,
            "project_name": project_name,
            "description": description,
            "github_repo": github_repo,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "is_public": True,
            "models": [],
            "evaluations": []
        }
        
        try:
            result = projects_collection.insert_one(project_data)
            project_data['_id'] = str(result.inserted_id)
            print(f"✅ Project created in cloud database: {project_code}")
            return project_data
        except Exception as e:
            print(f"❌ Failed to create project: {str(e)}")
            raise
    
    def get_project_by_code(self, project_code: str) -> Optional[Dict[str, Any]]:
        """Get project by code from shared cloud database."""
        try:
            project = projects_collection.find_one({"project_code": project_code.upper()})
            if project:
                project['_id'] = str(project['_id'])
                print(f"✅ Project found in cloud database: {project_code}")
                return project
            else:
                print(f"⚠️ Project not found in cloud database: {project_code}")
                return None
        except Exception as e:
            print(f"❌ Error fetching project: {str(e)}")
            return None
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects from shared cloud database."""
        try:
            projects = list(projects_collection.find())
            for p in projects:
                p['_id'] = str(p['_id'])
            print(f"✅ Found {len(projects)} projects in cloud database")
            return projects
        except Exception as e:
            print(f"❌ Error fetching projects: {str(e)}")
            return []
    
    # MODEL OPERATIONS
    def save_model(self, model_data: Dict[str, Any]) -> bool:
        """Save model metadata to shared cloud database."""
        try:
            models_collection.insert_one(model_data)
            # Update project
            projects_collection.update_one(
                {"project_code": model_data["project_code"]},
                {"$addToSet": {"models": model_data["model_id"]}}
            )
            print(f"✅ Model saved to cloud database: {model_data['model_id']}")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            return False
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by its ID from shared cloud database."""
        try:
            model = models_collection.find_one({"model_id": model_id})
            if model:
                model['_id'] = str(model['_id'])
            return model
        except Exception as e:
            print(f"❌ Error fetching model: {str(e)}")
            return None
    
    def get_models_by_project(self, project_code: str) -> List[Dict[str, Any]]:
        """Get all models for a project from shared cloud database."""
        try:
            models = list(models_collection.find({"project_code": project_code.upper()}))
            for m in models:
                m['_id'] = str(m['_id'])
            print(f"✅ Found {len(models)} models for project {project_code}")
            return models
        except Exception as e:
            print(f"❌ Error fetching models: {str(e)}")
            return []
    
    # EVALUATION OPERATIONS
    def save_evaluation(self, eval_data: Dict[str, Any]) -> bool:
        """Save evaluation results to shared cloud database."""
        try:
            evaluations_collection.insert_one(eval_data)
            # Update project
            projects_collection.update_one(
                {"project_code": eval_data["project_code"]},
                {"$addToSet": {"evaluations": eval_data["evaluation_id"]}}
            )
            print(f"✅ Evaluation saved to cloud database: {eval_data['evaluation_id']}")
            return True
        except Exception as e:
            print(f"❌ Failed to save evaluation: {str(e)}")
            return False
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get an evaluation by its ID from shared cloud database."""
        try:
            evaluation = evaluations_collection.find_one({"evaluation_id": evaluation_id})
            if evaluation:
                evaluation['_id'] = str(evaluation['_id'])
            return evaluation
        except Exception as e:
            print(f"❌ Error fetching evaluation: {str(e)}")
            return None
    
    def get_evaluations_by_project(self, project_code: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a project from shared cloud database."""
        try:
            evals = list(evaluations_collection.find({"project_code": project_code.upper()}))
            for e in evals:
                e['_id'] = str(e['_id'])
            print(f"✅ Found {len(evals)} evaluations for project {project_code}")
            return evals
        except Exception as e:
            print(f"❌ Error fetching evaluations: {str(e)}")
            return []


# Global database manager instance - connects to shared cloud database
db_manager = DatabaseManager()


if __name__ == "__main__":
    print(f"✅ Database configured: Shared MongoDB Atlas Cloud Database")
    print("✅ Testing database connection...")
    
    try:
        # Test project creation
        test_project = db_manager.create_project(
            project_name="Test Project",
            description="Testing shared cloud database connection"
        )
        print(f"✅ Test project created: {test_project['project_code']}")
        
        # Test retrieval
        retrieved = db_manager.get_project_by_code(test_project['project_code'])
        print(f"✅ Project retrieved: {retrieved['project_name']}")
        
        # Test list all projects
        all_projects = db_manager.get_all_projects()
        print(f"✅ Total projects in shared database: {len(all_projects)}")
        
        print("\n🎉 Shared cloud database is ready to use!")
        print("🌐 All users will connect to the same database and share project codes")
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        print("🔍 Please check your MongoDB connection string in .env file")
