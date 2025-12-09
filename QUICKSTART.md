# Quick Start Guide

This guide will help you set up and run the Credit Card Fraud Detection system in minutes.

## 📋 Prerequisites Checklist

- [ ] Python 3.12 or higher installed
- [ ] At least 4GB RAM available
- [ ] Internet connection for downloading dataset and packages
- [ ] Git installed (for cloning repository)

## 🚀 5-Minute Setup

### Step 1: Clone & Navigate

```bash
git clone https://github.com/yourusername/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Visit [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in `data/raw/creditcard.csv`

### Step 5: Train Model

```bash
python run_full_pipeline.py
```

Expected output:
```
✅ Data ingestion completed
✅ Data cleaning completed
✅ Feature engineering completed
✅ Model training completed
✅ Model saved: models/fraud_detection_model.pkl
Performance: ROC-AUC = 97.14%
```

### Step 6: Start Services

**Terminal 1 - API Backend:**
```bash
python -m uvicorn api.main:app --reload
```

**Terminal 2 - Streamlit Dashboard:**
```bash
streamlit run app.py
```

### Step 7: Access Dashboard

Open browser and navigate to: **http://localhost:8501**

## ✅ Verify Installation

### Test API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-24T..."
}
```

### Test Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12345,
    "Amount": 150.0,
    "V1": -1.359, "V2": -0.073, "V3": 2.536,
    "V4": 1.378, "V5": -0.338, "V6": 0.462,
    "V7": 0.240, "V8": 0.098, "V9": 0.364,
    "V10": 0.091, "V11": -0.551, "V12": -0.618,
    "V13": -0.991, "V14": -0.311, "V15": 1.468,
    "V16": -0.470, "V17": 0.208, "V18": 0.026,
    "V19": 0.404, "V20": 0.251, "V21": -0.019,
    "V22": 0.278, "V23": -0.110, "V24": 0.067,
    "V25": 0.129, "V26": -0.189, "V27": 0.134,
    "V28": -0.021
  }'
```

## 🎯 What's Included

After setup, you'll have:

- ✅ FastAPI backend running on port 8000
- ✅ Streamlit dashboard on port 8501
- ✅ Trained XGBoost model (97.14% ROC-AUC)
- ✅ Prometheus metrics endpoint
- ✅ Interactive fraud detection UI
- ✅ Batch prediction capability
- ✅ Real-time analytics dashboard

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Change API port
uvicorn api.main:app --port 8001

# Change Streamlit port
streamlit run app.py --server.port 8502
```

### Module Not Found

```bash
pip install -r requirements.txt --force-reinstall
```

### Model Not Loading

```bash
# Retrain the model
python run_full_pipeline.py
```

### Memory Issues

For systems with limited RAM, edit the pipeline to process in chunks:
```python
# In run_full_pipeline.py
chunk_size = 10000  # Reduce from default
```

## 📖 Next Steps

1. **Explore Dashboard**: Try single and batch predictions
2. **Check Analytics**: View real-time monitoring metrics
3. **API Documentation**: Visit http://localhost:8000/docs
4. **Customize Model**: Edit `configs/model_config.yaml`
5. **Add Features**: Modify `src/features/engineering.py`

## 🆘 Need Help?

- 📚 Check [README.md](README.md) for detailed documentation
- 🐛 Report issues on [GitHub Issues](../../issues)
- 💬 Join discussions on [GitHub Discussions](../../discussions)
- 📧 Email: your.email@example.com

## 🎉 You're All Set!

Your fraud detection system is ready for:
- Real-time predictions
- Batch processing
- Model monitoring
- Production deployment

Happy detecting! 🛡️
