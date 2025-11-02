# ML Model Monitoring Dashboard

A comprehensive machine learning model monitoring platform with cloud database integration, API endpoints, and project-based collaboration features.

## 🎯 Project Overview

This dashboard provides a complete solution for monitoring, analyzing, and sharing ML model performance with team collaboration features. It supports both individual use and production deployment with cloud database storage.

## 🚀 Key Features Developed

### ✨ **Core Features**

1. **🌐 Cloud Database Integration**
   - MongoDB Atlas cloud database with free tier
   - Persistent storage across deployments
   - Multi-project support with unique project codes

2. **📡 REST API Server**
   - FastAPI-based API with automatic documentation
   - Project creation and management endpoints
   - Model upload and evaluation endpoints
   - Team collaboration through project codes

3. **👁️ Project-Based Access System**
   - Unique 8-character project codes for secure access
   - No authentication required - share codes with team
   - Direct URL access: `?project_code=M74V8Y09`

4. **📊 Enhanced Dashboard Interface**
   - Streamlit-based web interface
   - Real-time model monitoring and visualization
   - SHAP explainability integration
   - Performance testing and synthetic data generation

5. **🤝 Team Collaboration**
   - Share project codes for instant team access
   - Centralized model and evaluation storage
   - Cross-team project visibility

## 📁 Project Structure

```
model-monitoring-dashboard/
├── 🔧 Core Components
│   ├── app.py                 # Main Streamlit dashboard
│   ├── api_server.py          # FastAPI REST server
│   ├── cloud_database.py      # MongoDB Atlas integration
│   └── requirements.txt       # Dependencies
│
├── 📈 ML Components  
│   ├── metrics.py             # Model evaluation metrics
│   ├── model_utils.py         # Model loading utilities
│   ├── explainability.py      # SHAP analysis
│   └── synthetic_data.py      # Data generation
│
├── 🔌 Integration
│   ├── sample_external_project/  # API integration example
│   │   ├── api_client.py         # Python API client
│   │   ├── train_and_evaluate.py # End-to-end workflow
│   │   └── test_integration.py   # Integration testing
│   └── example_client/          # Additional examples
│
├── ⚙️ Configuration
│   ├── .env                   # MongoDB connection string
│   ├── config.py              # Application config
│   └── schema_utils.py        # Data validation
│
└── 🧪 Testing & CI/CD
    ├── tests/                 # Unit tests
    └── .github/workflows/     # GitHub Actions
```

## 🛠️ Features Implementation Guide

### 1. **Cloud Database System** (`cloud_database.py`)

**What it does:**
- Provides unified database operations for MongoDB Atlas
- Handles project, model, and evaluation storage
- Supports both local and cloud deployments

**Key Components:**
```python
class DatabaseManager:
    def create_project(project_name, description) -> project_code
    def get_project_by_code(project_code) -> project_data
    def store_model(project_code, model_data) -> model_id
    def store_evaluation(model_id, results) -> evaluation_id
```

**Integration Steps:**
1. Set up MongoDB Atlas account (free tier)
2. Add connection string to `.env` file:
   ```
   MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/db
   ```
3. Import and use: `from cloud_database import db_manager`

### 2. **API Server** (`api_server.py`)

**What it does:**
- FastAPI server with automatic OpenAPI documentation
- Handles model uploads, project creation, evaluations
- Provides endpoints for team collaboration

**Key Endpoints:**
```
POST /api/v1/projects              # Create project
GET  /api/v1/projects              # List projects  
GET  /api/v1/projects/{code}       # Get project details
POST /api/v1/models/upload         # Upload model
POST /api/v1/models/{id}/evaluate  # Run evaluation
GET  /docs                         # API documentation
```

**Integration Steps:**
1. Start server: `python api_server.py`
2. Access docs: `http://localhost:8000/docs`
3. Use endpoints programmatically or via curl

### 3. **Project Code System**

**What it does:**
- Generates unique 8-character codes for each project
- Enables secure sharing without user accounts
- Supports direct dashboard access via URL parameters

**Implementation:**
```python
# Generate project code
import secrets, string
project_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))

# Access project
dashboard_url = f"http://localhost:8501/?project_code={project_code}"
```

### 4. **Enhanced Dashboard** (`app.py`)

**What it does:**
- Multi-page Streamlit interface
- Project code input and URL parameter support
- Real-time visualization and team collaboration

**Key Features:**
- **Home Page**: Setup instructions and workflow guides
- **View Project**: Project code input with comprehensive help
- **Model Upload**: Direct file upload with evaluation
- **Performance Testing**: Latency and throughput analysis
- **SHAP Explainability**: Model interpretation tools

### 5. **API Integration Example** (`sample_external_project/`)

**What it does:**
- Complete Python example of API integration
- Demonstrates end-to-end workflow from training to dashboard
- Includes error handling and best practices

**Workflow:**
1. Create project → Get project code
2. Train model → Save as .pkl file
3. Upload model → Get model ID
4. Run evaluation → Get results
5. View in dashboard → Share project code

## 🔧 Installation & Setup

### Prerequisites
```bash
# Python 3.8+
pip install -r requirements.txt
```

### Environment Setup
1. **Create `.env` file:**
   ```
   MONGODB_CONNECTION_STRING=your_mongodb_atlas_connection_string
   ```

2. **MongoDB Atlas Setup:**
   - Create free account at mongodb.com
   - Create cluster and database
   - Get connection string and add to `.env`

### Running the System
```bash
# Terminal 1: Start API Server
python api_server.py

# Terminal 2: Start Dashboard  
streamlit run app.py

# Terminal 3: Test Integration (optional)
cd sample_external_project
python test_integration.py
```

## 🤝 Team Integration Guide

### For Your Teammate to Integrate Your Features:

#### Step 1: Copy Core Files
Copy these files to the main codebase:
- `cloud_database.py` - Database integration
- `api_server.py` - API endpoints  
- Updated `app.py` - Enhanced dashboard
- `sample_external_project/` - API examples

#### Step 2: Update Dependencies
Merge `requirements.txt` with existing dependencies:
```bash
pip install fastapi uvicorn pymongo python-dotenv streamlit plotly
```

#### Step 3: Database Configuration
1. Create `.env` file with MongoDB connection
2. Test connection: `python -c "from cloud_database import db_manager; print('Connected!')"`

#### Step 4: API Integration
1. Update existing code to use project codes instead of project names
2. Replace file-based storage with `db_manager` calls
3. Test endpoints: `curl http://localhost:8000/api/v1/health`

#### Step 5: Dashboard Updates
1. Merge navigation updates in `app.py`
2. Add project code input functionality
3. Update URL parameter handling

### Code Migration Examples

**Before (File-based):**
```python
# Old approach
projects = load_projects_from_json()
save_model_to_file(model_data)
```

**After (Cloud Database):**
```python
# New approach  
from cloud_database import db_manager
project_code = db_manager.create_project(name, desc)
model_id = db_manager.store_model(project_code, model_data)
```

## 📋 API Usage Examples

### Create Project
```bash
curl -X POST "http://localhost:8000/api/v1/projects" \
  -F "project_name=Production Model v1" \
  -F "description=Customer churn prediction"
```

### Upload Model
```bash
curl -X POST "http://localhost:8000/api/v1/models/upload" \
  -F "model_file=@model.pkl" \
  -F "project_code=M74V8Y09" \
  -F "model_name=Churn Predictor" \
  -F "model_type=classification"
```

### Python Integration
```python
from sample_external_project.api_client import DashboardAPIClient

client = DashboardAPIClient()
project = client.create_project("My Project")
model_id = client.upload_model("model.pkl", "classification", project['project_code'])
```

## 🧪 Testing

### Quick Tests
```bash
# API Server Test
curl http://localhost:8000/api/v1/health

# Dashboard Test  
# Open http://localhost:8501 in browser

# Integration Test
cd sample_external_project
python test_integration.py
```

### End-to-End Test
```bash
cd sample_external_project  
python train_and_evaluate.py
# Follow the output instructions to view results
```

## 📊 Monitoring & Troubleshooting

### Common Issues

**MongoDB Connection:**
- Check `.env` file exists and has correct connection string
- Verify network access to MongoDB Atlas
- Test: `python -c "from cloud_database import db_manager; print('OK')"`

**API Server:**
- Ensure port 8000 is free: `netstat -an | findstr 8000`
- Check logs for detailed error messages
- Verify all dependencies installed

**Dashboard:**
- Ensure port 8501 is free
- Check Streamlit logs for errors
- Verify emoji display (UTF-8 encoding)

### Performance Tips
- MongoDB Atlas free tier: 512MB storage limit
- API server: Can handle ~100 concurrent requests
- Dashboard: Best with <1000 projects for optimal performance

## 🔒 Security Considerations

- **Project codes** provide access control without authentication
- **No sensitive data** stored in project codes
- **MongoDB Atlas** includes built-in security features
- **API rate limiting** recommended for production

## 📞 Support & Contributing

### For Your Teammate:
1. **Questions**: Check the inline code documentation
2. **Issues**: Test with `sample_external_project/test_integration.py`
3. **Features**: Follow the existing patterns in `api_server.py` and `cloud_database.py`

### Development Workflow:
1. Test locally with sample project
2. Verify API endpoints with `/docs`
3. Check dashboard functionality with test project codes
4. Review MongoDB data structure in Atlas dashboard

---

## 🎉 Summary

This ML Model Monitoring Dashboard provides:
- ✅ **Cloud-based storage** with MongoDB Atlas
- ✅ **REST API** for programmatic access  
- ✅ **Team collaboration** via project codes
- ✅ **Production-ready** architecture
- ✅ **Complete documentation** and examples

Your teammate can integrate these features by following the step-by-step guide above. The modular design allows for gradual adoption and testing of each component.