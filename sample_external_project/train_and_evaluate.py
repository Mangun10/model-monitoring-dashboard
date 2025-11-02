"""
Main script that demonstrates training a model and evaluating it using the ML Model Monitoring Dashboard API.
"""

import pickle
import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
import time

from api_client import DashboardAPIClient
from create_sample_data import create_sample_datasets
from config import DEFAULT_PROJECT_NAME, MODEL_TYPE

def train_classification_model():
    """Train a classification model and save it."""
    print("🤖 Training classification model...")
    
    # Load training data
    if not os.path.exists('data/classification_train.csv'):
        print("Creating sample data first...")
        create_sample_datasets()
    
    train_data = pd.read_csv('data/classification_train.csv')
    X_train = train_data.drop('is_positive', axis=1)
    y_train = train_data['is_positive']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Quick validation
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"   Training accuracy: {accuracy:.3f}")
    
    # Save model
    model_path = 'models/classification_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

def train_regression_model():
    """Train a regression model and save it."""
    print("🤖 Training regression model...")
    
    # Load training data
    if not os.path.exists('data/regression_train.csv'):
        print("Creating sample data first...")
        create_sample_datasets()
    
    train_data = pd.read_csv('data/regression_train.csv')
    X_train = train_data.drop('target_value', axis=1)
    y_train = train_data['target_value']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Quick validation
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    print(f"   Training R²: {r2:.3f}")
    
    # Save model
    model_path = 'models/regression_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

def main():
    """Main function that demonstrates the complete workflow."""
    
    print("=" * 60)
    print("🚀 ML Model Monitoring Dashboard - Sample External Project")
    print("=" * 60)
    
    # Initialize API client
    client = DashboardAPIClient()
    
    # Wait for API server
    if not client.wait_for_api():
        print("\n❌ API server is not running!")
        print("Please start the API server first:")
        print("   cd ../")
        print("   python api_server.py")
        return
    
    try:
        # Step 1: Create sample data if needed
        if not os.path.exists('data'):
            print("\n📊 Creating sample datasets...")
            create_sample_datasets()
        
        # Step 2: Create a new project
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"{DEFAULT_PROJECT_NAME}_{timestamp}"
        
        print(f"\n🆕 Creating new project: {project_name}")
        project_result = client.create_project(
            project_name=project_name,
            description=f"Sample {MODEL_TYPE} project created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        project_code = project_result['project_code']
        
        # Step 3: Train model based on configuration
        print(f"\n🎯 Training {MODEL_TYPE} model...")
        if MODEL_TYPE == "classification":
            model_path = train_classification_model()
            test_dataset = 'data/classification_test.csv'
            target_column = 'is_positive'
        else:
            model_path = train_regression_model()
            test_dataset = 'data/regression_test.csv'
            target_column = 'target_value'
        
        # Step 4: Upload model to dashboard
        print(f"\n📤 Uploading model to dashboard...")
        upload_result = client.upload_model(
            model_path=model_path,
            model_type=MODEL_TYPE,
            project_code=project_code,
            model_name=f"{MODEL_TYPE}_model_{timestamp}",
            model_version="1.0",
            description=f"Sample {MODEL_TYPE} model trained with RandomForest"
        )
        
        model_id = upload_result['model_id']
        
        # Step 5: Evaluate model
        print(f"\n🧪 Evaluating model...")
        evaluation_result = client.evaluate_model(
            model_id=model_id,
            dataset_path=test_dataset,
            target_column=target_column,
            test_name=f"{MODEL_TYPE}_evaluation_{timestamp}"
        )
        
        evaluation_id = evaluation_result['evaluation_id']
        
        # Step 6: Get detailed results
        print(f"\n📊 Getting detailed results...")
        detailed_results = client.get_evaluation_results(evaluation_id)
        
        print("\n" + "=" * 50)
        print("📈 EVALUATION SUMMARY")
        print("=" * 50)
        
        metrics = detailed_results['metrics']
        if MODEL_TYPE == "classification":
            print(f"Accuracy:  {metrics.get('accuracy', 'N/A'):.3f}")
            print(f"Precision: {metrics.get('precision', 'N/A'):.3f}")
            print(f"Recall:    {metrics.get('recall', 'N/A'):.3f}")
            print(f"F1 Score:  {metrics.get('f1', 'N/A'):.3f}")
        else:
            print(f"R² Score:  {metrics.get('r2', 'N/A'):.3f}")
            print(f"RMSE:      {metrics.get('rmse', 'N/A'):.3f}")
            print(f"MAE:       {metrics.get('mae', 'N/A'):.3f}")
            print(f"MAPE:      {metrics.get('mape', 'N/A'):.1f}%")
        
        # Step 7: Show project information
        print(f"\n📋 Project Information:")
        project_info = client.get_project_info(project_code)
        print(f"   Project Name: {project_info['project_name']}")
        print(f"   Project Code: {project_code}")
        print(f"   Models: {len(project_info['models'])}")
        print(f"   Evaluations: {len(project_info['evaluations'])}")
        
        # Step 8: Dashboard links
        print(f"\n🌐 Dashboard Links:")
        print(f"   Project Dashboard: http://localhost:8501/?project_code={project_code}")
        print(f"   Direct Link: http://localhost:8501/?project_code={project_code}")
        
        print(f"\n✅ Successfully completed model training and evaluation!")
        print(f"   🔑 Your project code is: {project_code}")
        print(f"   💡 Use this code to access your project in the dashboard.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()