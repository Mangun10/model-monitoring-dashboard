"""
Simplified Database Configuration for Cloud Deployment
No authentication required - only project code-based access
Supports both MongoDB and PostgreSQL
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# Detect database type from environment
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "mongodb")  # mongodb or postgresql

if DATABASE_TYPE == "mongodb":
    # MongoDB Configuration
    from pymongo import MongoClient
    from bson import ObjectId
    
    MONGODB_URL = os.getenv(
        "MONGODB_URL",
        "mongodb://localhost:27017/"
    )
    
    client = MongoClient(MONGODB_URL)
    db = client['ml_monitoring']
    
    # Collections
    projects_collection = db['projects']
    models_collection = db['models']
    evaluations_collection = db['evaluations']
    
    # Create indexes for faster queries
    projects_collection.create_index("project_code", unique=True)
    models_collection.create_index("model_id", unique=True)
    evaluations_collection.create_index("evaluation_id", unique=True)
    
    print("✅ Connected to MongoDB")

else:
    # PostgreSQL Configuration
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Text, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://localhost:5432/ml_monitoring"
    )
    
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    class Project(Base):
        """Project model."""
        __tablename__ = "projects"
        
        id = Column(Integer, primary_key=True, index=True)
        project_code = Column(String(20), unique=True, index=True, nullable=False)
        project_name = Column(String(200), nullable=False)
        description = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        github_repo = Column(String(300))
        is_public = Column(Boolean, default=True)
    
    class Model(Base):
        """Model metadata."""
        __tablename__ = "models"
        
        id = Column(Integer, primary_key=True, index=True)
        model_id = Column(String(50), unique=True, index=True, nullable=False)
        project_code = Column(String(20), index=True, nullable=False)
        model_type = Column(String(50), nullable=False)
        filename = Column(String(200), nullable=False)
        file_path = Column(String(500))
        description = Column(Text)
        upload_time = Column(DateTime, default=datetime.utcnow)
        status = Column(String(50), default="uploaded")
        framework = Column(String(50))
        version = Column(String(20))
    
    class Evaluation(Base):
        """Evaluation results."""
        __tablename__ = "evaluations"
        
        id = Column(Integer, primary_key=True, index=True)
        evaluation_id = Column(String(50), unique=True, index=True, nullable=False)
        model_id = Column(String(50), index=True, nullable=False)
        project_code = Column(String(20), index=True, nullable=False)
        test_name = Column(String(100), nullable=False)
        evaluation_time = Column(DateTime, default=datetime.utcnow)
        status = Column(String(50), default="completed")
        
        # Dataset info and metrics stored as JSON
        dataset_info = Column(JSON)
        metrics = Column(JSON, nullable=False)
        predictions = Column(JSON)
        actual_values = Column(JSON)
        
        # GitHub commit tracking
        commit_hash = Column(String(40))
        commit_message = Column(Text)
        branch_name = Column(String(100))
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("✅ Connected to PostgreSQL")


def generate_project_code() -> str:
    """Generate a unique 8-character project code."""
    import secrets
    import string
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(8))


# ========================================
# Database Operations (works for both DBs)
# ========================================

class DatabaseManager:
    """Unified database manager for MongoDB and PostgreSQL."""
    
    def __init__(self):
        self.db_type = DATABASE_TYPE
    
    # PROJECT OPERATIONS
    def create_project(self, project_name: str, description: str = "", github_repo: str = "") -> Dict[str, Any]:
        """Create a new project with unique code."""
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
        
        if self.db_type == "mongodb":
            result = projects_collection.insert_one(project_data)
            project_data['_id'] = str(result.inserted_id)
        else:
            db = SessionLocal()
            try:
                db_project = Project(**project_data)
                db.add(db_project)
                db.commit()
                db.refresh(db_project)
            finally:
                db.close()
        
        return project_data
    
    def get_project_by_code(self, project_code: str) -> Optional[Dict[str, Any]]:
        """Get project by code."""
        if self.db_type == "mongodb":
            project = projects_collection.find_one({"project_code": project_code})
            if project:
                project['_id'] = str(project['_id'])
            return project
        else:
            db = SessionLocal()
            try:
                project = db.query(Project).filter(Project.project_code == project_code).first()
                if project:
                    return {
                        "project_code": project.project_code,
                        "project_name": project.project_name,
                        "description": project.description,
                        "created_at": project.created_at.isoformat(),
                        "github_repo": project.github_repo
                    }
            finally:
                db.close()
            return None
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects."""
        if self.db_type == "mongodb":
            projects = list(projects_collection.find())
            for p in projects:
                p['_id'] = str(p['_id'])
            return projects
        else:
            db = SessionLocal()
            try:
                projects = db.query(Project).all()
                return [
                    {
                        "project_code": p.project_code,
                        "project_name": p.project_name,
                        "description": p.description,
                        "created_at": p.created_at.isoformat()
                    }
                    for p in projects
                ]
            finally:
                db.close()
    
    # MODEL OPERATIONS
    def save_model(self, model_data: Dict[str, Any]) -> bool:
        """Save model metadata."""
        if self.db_type == "mongodb":
            models_collection.insert_one(model_data)
            # Update project
            projects_collection.update_one(
                {"project_code": model_data["project_code"]},
                {"$addToSet": {"models": model_data["model_id"]}}
            )
        else:
            db = SessionLocal()
            try:
                db_model = Model(**model_data)
                db.add(db_model)
                db.commit()
            finally:
                db.close()
        return True
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by its ID."""
        if self.db_type == "mongodb":
            model = models_collection.find_one({"model_id": model_id})
            if model:
                model['_id'] = str(model['_id'])
            return model
        else:
            db = SessionLocal()
            try:
                model = db.query(Model).filter(Model.model_id == model_id).first()
                if model:
                    return {
                        "model_id": model.model_id,
                        "project_code": model.project_code,
                        "model_type": model.model_type,
                        "filename": model.filename,
                        "file_path": model.file_path,
                        "upload_time": model.upload_time.isoformat(),
                        "status": model.status,
                        "description": model.description
                    }
            finally:
                db.close()
            return None
    
    def get_models_by_project(self, project_code: str) -> List[Dict[str, Any]]:
        """Get all models for a project."""
        if self.db_type == "mongodb":
            models = list(models_collection.find({"project_code": project_code}))
            for m in models:
                m['_id'] = str(m['_id'])
            return models
        else:
            db = SessionLocal()
            try:
                models = db.query(Model).filter(Model.project_code == project_code).all()
                return [
                    {
                        "model_id": m.model_id,
                        "project_code": m.project_code,
                        "model_type": m.model_type,
                        "filename": m.filename,
                        "upload_time": m.upload_time.isoformat(),
                        "status": m.status
                    }
                    for m in models
                ]
            finally:
                db.close()
    
    # EVALUATION OPERATIONS
    def save_evaluation(self, eval_data: Dict[str, Any]) -> bool:
        """Save evaluation results."""
        if self.db_type == "mongodb":
            evaluations_collection.insert_one(eval_data)
            # Update project
            projects_collection.update_one(
                {"project_code": eval_data["project_code"]},
                {"$addToSet": {"evaluations": eval_data["evaluation_id"]}}
            )
        else:
            db = SessionLocal()
            try:
                db_eval = Evaluation(**eval_data)
                db.add(db_eval)
                db.commit()
            finally:
                db.close()
        return True
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get an evaluation by its ID."""
        if self.db_type == "mongodb":
            evaluation = evaluations_collection.find_one({"evaluation_id": evaluation_id})
            if evaluation:
                evaluation['_id'] = str(evaluation['_id'])
            return evaluation
        else:
            db = SessionLocal()
            try:
                evaluation = db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first()
                if evaluation:
                    return {
                        "evaluation_id": evaluation.evaluation_id,
                        "model_id": evaluation.model_id,
                        "project_code": evaluation.project_code,
                        "test_name": evaluation.test_name,
                        "metrics": evaluation.metrics,
                        "dataset_info": evaluation.dataset_info,
                        "predictions": evaluation.predictions,
                        "actual_values": evaluation.actual_values,
                        "evaluation_time": evaluation.evaluation_time.isoformat(),
                        "status": evaluation.status,
                        "commit_hash": evaluation.commit_hash
                    }
            finally:
                db.close()
            return None
    
    def get_evaluations_by_project(self, project_code: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a project."""
        if self.db_type == "mongodb":
            evals = list(evaluations_collection.find({"project_code": project_code}))
            for e in evals:
                e['_id'] = str(e['_id'])
            return evals
        else:
            db = SessionLocal()
            try:
                evals = db.query(Evaluation).filter(Evaluation.project_code == project_code).all()
                return [
                    {
                        "evaluation_id": e.evaluation_id,
                        "model_id": e.model_id,
                        "test_name": e.test_name,
                        "metrics": e.metrics,
                        "evaluation_time": e.evaluation_time.isoformat(),
                        "status": e.status,
                        "commit_hash": e.commit_hash
                    }
                    for e in evals
                ]
            finally:
                db.close()


# Global database manager instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    print(f"✅ Database configured: {DATABASE_TYPE}")
    print("✅ Testing database connection...")
    
    # Test project creation
    test_project = db_manager.create_project(
        project_name="Test Project",
        description="Testing cloud database"
    )
    print(f"✅ Test project created: {test_project['project_code']}")
    
    # Test retrieval
    retrieved = db_manager.get_project_by_code(test_project['project_code'])
    print(f"✅ Project retrieved: {retrieved['project_name']}")
    
    print("\n🎉 Database is ready to use!")
