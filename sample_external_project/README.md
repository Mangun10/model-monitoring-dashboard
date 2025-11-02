# Sample External Project Template

This folder contains a **ready-to-copy template** showing how to integrate your ML projects with the Model Monitoring Dashboard. Copy these files to your own repository to get started quickly.

## 📁 What's Included

```
sample_external_project/
├── .github/workflows/ci.yml      # GitHub Actions workflow template
├── api_client.py                 # Python client for dashboard API
├── train_and_evaluate.py         # Sample training script
├── create_sample_data.py         # Test data generation
├── test_integration.py           # Integration tests
├── config.py                     # Configuration
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start (Copy to Your Repo)

### Step 1: Copy Files
Copy these files to your ML project repository:

```bash
# Copy the workflow template
cp sample_external_project/.github/workflows/ci.yml YOUR_REPO/.github/workflows/

# Copy the training script (adapt as needed)
cp sample_external_project/train_and_evaluate.py YOUR_REPO/
cp sample_external_project/api_client.py YOUR_REPO/
```

### Step 2: Configure Secrets
In your repository, go to **Settings → Secrets → Actions** and add:

| Secret Name | Value | Required |
|------------|-------|----------|
| `MONGODB_CONNECTION_STRING` | Your MongoDB Atlas connection string | ✅ Yes |
| `DASHBOARD_API_URL` | Your dashboard URL (e.g., `https://your-dashboard.com`) | ⚪ Optional |

### Step 3: Adapt the Training Script
Edit `train_and_evaluate.py` to use your actual model training code. The script should:

1. **Train your model**
2. **Print project code**: `"Your project code is: {code}"`  
3. **Print metrics**: `"Model Accuracy: {accuracy}"`
4. **Save model file** (optional)

### Step 4: Test Integration
```bash
# Test locally first
python train_and_evaluate.py

# Push to trigger CI
git add .
git commit -m "Add model monitoring"
git push
```

## 📝 Template Files Explained

### `.github/workflows/ci.yml`
**GitHub Actions workflow** that:
- Triggers on pushes and pull requests
- Runs your training script
- Extracts project codes from output
- Posts results as PR comments
- Uploads model artifacts

**Key environment variables it uses:**
- `MONGODB_CONNECTION_STRING` - To store results directly in database
- `DASHBOARD_API_URL` - Alternative: POST to dashboard API
- `GITHUB_*` variables - Automatically provided by GitHub

### `train_and_evaluate.py`
**Sample training script** that demonstrates:
- Creating synthetic data for testing
- Training a RandomForest model  
- Integration with dashboard API or MongoDB
- Structured output for CI parsing

**Output format expected by CI:**
```
Your project code is: A1B2C3D4
Model ID: model_xyz789
Model Accuracy: 0.8542
```

### `api_client.py`
**Python client** for the dashboard API with methods:
- `create_project()` - Create new projects
- `upload_model()` - Upload trained models
- `evaluate_model()` - Run evaluations
- `get_project_info()` - Retrieve project data

## 🔧 Customization Guide

### Modify Training Script
Replace the sample model training with your actual code:

```python
# Replace this section in train_and_evaluate.py
def train_your_model():
    # Your training code here
    model = YourModelClass()
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    
    # Save model
    joblib.dump(model, 'your_model.pkl')
    
    return model, accuracy
```

### Add Custom Metrics
Extend the evaluation to include your metrics:

```python
# Add to the evaluation section
metrics = {
    'accuracy': accuracy,
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'custom_metric': your_custom_metric(y_test, y_pred)
}
```

### Modify CI Workflow
Adapt the workflow for your needs:

```yaml
# Add custom steps before training
- name: Download data
  run: python download_training_data.py

# Add custom steps after training  
- name: Deploy model
  run: python deploy_to_production.py
```

## 🧪 Testing Integration

### Test 1: Local API Integration
```bash
# Start dashboard API server (in dashboard repo)
python api_server.py

# Test integration (in your project repo)
python test_integration.py
```

### Test 2: Direct MongoDB Integration
```bash
# Set environment variable
export MONGODB_CONNECTION_STRING="mongodb+srv://..."

# Run training script
python train_and_evaluate.py
```

### Test 3: CI Workflow
```bash
# Make a small change and push
echo "# Test CI" >> README.md
git add README.md
git commit -m "Test CI integration"
git push origin main
```

Check GitHub Actions tab for workflow execution.

## 📊 Expected CI Output

### Successful Run:
```
🚀 CI Train and Evaluate (sample_external_project)
🆕 Project: CI_yourorg_yourrepo_main_20251102_143022
🔑 Your project code is: A1B2C3D4
🆔 Model ID: model_xyz789
🎯 Model Accuracy: 0.8542
💾 Model file: CI_RandomForest_20251102_143022.pkl
============================================================
```

### PR Comment:
```markdown
## 🤖 Sample ML CI Results

🔑 Project Code: `A1B2C3D4`
🎯 Accuracy: 0.8542

[View in Dashboard](https://your-dashboard.com/?project_code=A1B2C3D4)
```

## 🔗 Integration Options

### Option 1: API Integration (Recommended)
- Dashboard API server must be running
- POST data to `/api/projects/` endpoints
- Good for development and testing

### Option 2: Direct MongoDB (Production)
- Write directly to MongoDB Atlas
- No API server dependency
- Better for CI/CD environments

### Option 3: Hybrid Approach
```python
# Try API first, fallback to MongoDB
try:
    # Use API if available
    result = api_client.create_project(...)
except:
    # Fallback to direct MongoDB
    result = db_manager.create_project(...)
```

## 🛠 Troubleshooting

### Common Issues:

**❌ "No project code found in CI output"**
- Check if training script prints the expected format
- Verify MongoDB connection string is set
- Look for errors in GitHub Actions logs

**❌ "Workflow fails on dependencies"**
- Update `requirements.txt` with your project's dependencies
- Check Python version compatibility

**❌ "API connection timeout"**
- Use direct MongoDB integration for CI
- Check if dashboard API URL is accessible from CI

### Debug Commands:
```bash
# Test MongoDB connection
python -c "from api_client import *; print('Connection OK')"

# Check environment variables (in CI)
echo "MongoDB string length: ${#MONGODB_CONNECTION_STRING}"

# Test project code extraction
grep "Your project code is:" ci_output.log
```

---

## 📋 Checklist for Your Repository

- [ ] Copied workflow file to `.github/workflows/ci.yml`
- [ ] Added `MONGODB_CONNECTION_STRING` secret
- [ ] Adapted `train_and_evaluate.py` for your model
- [ ] Updated `requirements.txt` with dependencies
- [ ] Tested integration locally
- [ ] Triggered CI with a test commit
- [ ] Verified project code appears in dashboard

## 🎯 Next Steps

1. **Customize the training script** with your actual model
2. **Add more sophisticated evaluation** metrics
3. **Integrate with your existing** ML pipeline
4. **Set up production deployment** with the dashboard
5. **Share project codes** with your team

Happy model monitoring! 🚀

---

Need help? Check the main repository README or open an issue!