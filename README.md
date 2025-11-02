# 📊 ML Model Monitoring Dashboard

A comprehensive machine learning model monitoring and evaluation dashboard that helps teams track, compare, and analyze their ML models in one centralized location.

## 🎯 What This Dashboard Does

This is a **cloud-based ML monitoring platform** that automatically tracks your models' performance, stores evaluation results, and provides interactive visualizations - all accessible through a simple web interface.

### ✨ Key Features

- **🎯 Automatic Model Tracking**: Upload models and get instant performance metrics
- **📊 Interactive Dashboard**: Real-time charts, metrics, and model comparisons
- **🔍 Model Explainability**: SHAP analysis to understand how your models make decisions  
- **🌐 Cloud Database**: MongoDB Atlas for persistent, team-accessible storage
- **🔗 REST API**: Complete API for programmatic integration with your ML pipelines
- **👥 Multi-Project Organization**: Organize models by projects with shareable project codes
- **🚀 CI/CD Integration**: Automatic model evaluation in GitHub Actions workflows
- **📱 Mobile-Friendly**: Access your dashboard from any device

## 🎯 Who This Is For

- **Data Scientists** who want to monitor model performance over time
- **ML Engineers** implementing model monitoring in production pipelines  
- **Research Teams** comparing different model approaches and experiments
- **Organizations** needing centralized model tracking across multiple projects

## 🚀 How It Works

### 1. **Project-Based Organization**
Each project gets a unique 8-character code (e.g., `M74V8Y09`) that you can share with your team. No accounts or authentication needed - just share the code for instant access.

### 2. **Multiple Integration Options**
- **Web Interface**: Upload models directly through the dashboard
- **Python API**: Integrate with existing ML pipelines
- **GitHub Actions**: Automatic evaluation on every commit
- **REST API**: Use from any programming language

### 3. **Comprehensive Analytics**
- Model performance metrics (accuracy, precision, recall, F1)
- Interactive visualizations and trend analysis  
- Model explainability with SHAP values
- Data drift detection and monitoring

---

## 📖 User Guide

### 🔧 Quick Setup (5 minutes)

**Prerequisites**: Python 3.8+

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mangun10/model-monitoring-dashboard.git
   cd model-monitoring-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB Atlas** (free tier)
   - Create account at [mongodb.com](https://mongodb.com)
   - Create a cluster and get connection string
   - Create `.env` file:
     ```
     MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/database
     ```

4. **Start the system**
   ```bash
   # Terminal 1: Start API server
   python api_server.py
   
   # Terminal 2: Start dashboard
   streamlit run app.py
   ```

5. **Access the dashboard**
   - Open browser to `http://localhost:8501`
   - Start monitoring your models!

### 🎯 Using the Dashboard

#### **Option 1: Web Interface (Easiest)**

1. **Create a Project**
   - Click "Create New Project" on the homepage
   - Enter project name and description
   - Get your unique project code (e.g., `M74V8Y09`)

2. **Upload a Model**
   - Go to your project dashboard
   - Upload your `.pkl` model file
   - Add model metadata (name, type, version)

3. **Run Evaluation**
   - Upload test dataset (CSV format)
   - Select target column
   - Get comprehensive performance metrics

4. **Share with Team**
   - Share your project code: `M74V8Y09`
   - Team members access via: `http://localhost:8501/?project_code=M74V8Y09`

#### **Option 2: Python API Integration**

```python
from sample_external_project.api_client import DashboardAPIClient

# Initialize client
client = DashboardAPIClient("http://localhost:8000")

# Create project
project = client.create_project(
    project_name="My ML Project",
    description="Customer churn prediction model"
)
project_code = project['project_code']  # e.g., "M74V8Y09"

# Upload model
model_result = client.upload_model(
    model_path="my_model.pkl",
    model_type="classification", 
    project_code=project_code,
    model_name="Random Forest v1.0"
)

# Evaluate model
evaluation = client.evaluate_model(
    model_id=model_result['model_id'],
    dataset_path="test_data.csv",
    target_column="target"
)

print(f"Model accuracy: {evaluation['metrics']['accuracy']}")
print(f"Dashboard: http://localhost:8501/?project_code={project_code}")
```

#### **Option 3: Automated CI/CD (GitHub Actions)**

Set up automatic model training and evaluation on every commit:

1. **Copy the template workflow**
   ```bash
   # Copy from sample to your repository
   cp sample_external_project/.github/workflows/ci.yml .github/workflows/
   ```

2. **Add repository secrets**
   - Go to GitHub repo → Settings → Secrets → Actions
   - Add `MONGODB_CONNECTION_STRING` with your Atlas connection string
   - Add `DASHBOARD_API_URL` (optional): your deployed dashboard URL

3. **Commit and push**
   - Workflow runs automatically
   - Creates project code and posts results in PR comments
   - Models appear in dashboard instantly

### 📊 Dashboard Features

#### **Home Page**
- Overview of all projects
- Recent model uploads and evaluations
- Quick access to create new projects

#### **Project Dashboard**
- Model performance over time
- Comparison between different models
- Detailed metrics and visualizations

#### **Model Details**
- Comprehensive evaluation metrics
- SHAP explainability plots
- Model metadata and version history

#### **API Documentation**
- Interactive API docs at `http://localhost:8000/docs`
- Complete endpoint reference
- Code examples in multiple languages

### 🔧 Configuration Options

#### **Environment Variables** (`.env` file)
```bash
# Required
MONGODB_CONNECTION_STRING=mongodb+srv://...

# Optional
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
DEBUG=false
```

#### **MongoDB Collections**
The system automatically creates these collections:
- `projects` - Project metadata and codes
- `models` - Stored model binaries and metadata  
- `evaluations` - Test results and metrics

#### **Supported Model Types**
- **Classification**: Accuracy, precision, recall, F1-score, confusion matrix
- **Regression**: R², RMSE, MAE, MAPE
- **Custom metrics**: Add your own evaluation functions

### 🚀 Production Deployment

#### **Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access dashboard at http://localhost:8501
```

#### **Cloud Deployment Options**
- **Streamlit Cloud**: Deploy dashboard with one click
- **Heroku**: Easy deployment with git integration
- **AWS/GCP/Azure**: Full cloud infrastructure deployment

#### **Security Considerations**
- MongoDB Atlas provides enterprise-grade security
- Project codes act as access tokens (share carefully)
- Consider adding authentication for production use
- Use environment variables for sensitive configuration

### 🔍 API Reference

#### **Key Endpoints**

**Projects**
- `POST /api/projects` - Create new project
- `GET /api/projects/{project_code}` - Get project details
- `GET /api/projects` - List all projects

**Models**  
- `POST /api/models/upload` - Upload model binary
- `GET /api/models/{model_id}` - Get model details
- `DELETE /api/models/{model_id}` - Delete model

**Evaluations**
- `POST /api/evaluations` - Run model evaluation  
- `GET /api/evaluations/{evaluation_id}` - Get results
- `GET /api/evaluations/model/{model_id}` - List model evaluations

#### **Example Responses**

**Create Project Response**
```json
{
  "project_code": "M74V8Y09",
  "project_name": "Customer Churn Prediction",
  "description": "Production ML model for customer retention",
  "created_at": "2025-11-02T14:30:22Z"
}
```

**Model Evaluation Response**
```json
{
  "evaluation_id": "eval_abc123",
  "model_id": "model_xyz789", 
  "metrics": {
    "accuracy": 0.8542,
    "precision": 0.8234,
    "recall": 0.8891,
    "f1": 0.8553
  },
  "status": "completed"
}
```

### 🛠 Troubleshooting

#### **Common Issues**

**❌ "Connection to MongoDB failed"**
- Check your connection string in `.env`
- Verify MongoDB Atlas cluster is running
- Ensure IP address is whitelisted in Atlas

**❌ "API server not responding"**
- Confirm `python api_server.py` is running
- Check if port 8000 is available
- Try accessing `http://localhost:8000/docs`

**❌ "Dashboard shows no data"**
- Verify project code is correct
- Check if models were uploaded successfully
- Ensure API server and dashboard are connected

**❌ "Model upload fails"**
- Verify file is a valid pickle (.pkl) format
- Check file size (large models may timeout)
- Ensure scikit-learn version compatibility

#### **Getting Help**

- **API Documentation**: `http://localhost:8000/docs`
- **Sample Integration**: Check `sample_external_project/` folder
- **GitHub Issues**: Report bugs and feature requests
- **MongoDB Atlas Support**: For database-related issues

---

## 🧪 Example: Complete Workflow

Here's a complete example showing how to use the dashboard:

```python
# example_workflow.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sample_external_project.api_client import DashboardAPIClient

# 1. Create sample data and train model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('example_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save test data  
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)

# 2. Initialize dashboard client
client = DashboardAPIClient("http://localhost:8000")

# 3. Create project
project = client.create_project(
    project_name="Example Classification Project",
    description="Demo workflow for the ML monitoring dashboard"
)
project_code = project['project_code']

# 4. Upload model
model_result = client.upload_model(
    model_path="example_model.pkl",
    model_type="classification",
    project_code=project_code,
    model_name="Random Forest Demo",
    model_version="1.0",
    description="Example model for demonstration"
)

# 5. Evaluate model
evaluation = client.evaluate_model(
    model_id=model_result['model_id'],
    dataset_path="test_data.csv", 
    target_column="target",
    test_name="Initial Evaluation"
)

# 6. Print results
print(f"✅ Project created: {project_code}")
print(f"📊 Model accuracy: {evaluation['metrics']['accuracy']:.3f}")
print(f"🔗 Dashboard: http://localhost:8501/?project_code={project_code}")
```

Run this example:
```bash
python example_workflow.py
```

---

## 📁 Project Structure

```
model-monitoring-dashboard/
├── 🔧 Core Components
│   ├── app.py                    # Streamlit dashboard
│   ├── api_server.py             # FastAPI REST server  
│   ├── cloud_database.py         # MongoDB integration
│   └── requirements.txt          # Python dependencies
│
├── 📈 ML Components
│   ├── metrics.py                # Evaluation metrics
│   ├── model_utils.py            # Model utilities
│   ├── explainability.py         # SHAP analysis
│   └── synthetic_data.py         # Test data generation
│
├── 🔌 Integration Template
│   └── sample_external_project/  # Copy this to your repo
│       ├── .github/workflows/ci.yml    # GitHub Actions template
│       ├── api_client.py               # Python API client
│       ├── train_and_evaluate.py       # Sample training script  
│       └── requirements.txt            # Dependencies
│
└── 🧪 Testing
    └── tests/                    # Unit tests
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and open an issue or pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Get started in 5 minutes** - Clone, install, configure MongoDB, and start monitoring your ML models! 🚀