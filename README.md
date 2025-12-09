# 🛡️ Credit Card Fraud Detection - Production MLOps Pipeline

A complete production-ready machine learning system for real-time credit card fraud detection with MLOps best practices, featuring FastAPI backend and interactive Streamlit dashboard.

## 🌟 Live Demo

![Fraud Detection Dashboard](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-97.14%25-success)

## 🏗️ System Architecture

```
Data Ingestion → Data Cleaning & Validation → Feature Engineering → 
Model Training (XGBoost) → Model Evaluation → Model Deployment → 
FastAPI Backend → Streamlit Dashboard → Prometheus Monitoring
```

## ✨ Key Features

### Machine Learning Pipeline
- **Automated Data Processing**: Ingestion, cleaning, and validation with Pandera schemas
- **Advanced Feature Engineering**: Time-based and amount-based feature extraction
- **Model Training**: XGBoost classifier with class imbalance handling
- **Performance**: 97.14% ROC-AUC, 99.92% Accuracy, 78.47% F1-Score
- **MLflow Integration**: Experiment tracking and model versioning

### Production API
- **FastAPI Backend**: High-performance REST API on port 8000
- **Endpoints**: 
  - `/predict` - Single transaction prediction
  - `/predict/batch` - Batch predictions with CSV upload
  - `/health` - Health check
  - `/metrics` - Prometheus metrics
  - `/model/info` - Model metadata
- **Validation**: Pydantic models for request/response validation
- **Monitoring**: Prometheus metrics collection

### Interactive Dashboard
- **Streamlit UI**: Beautiful web interface on port 8501
- **Single Prediction**: Interactive fraud probability gauge with confidence scores
- **Batch Analysis**: CSV upload with fraud distribution visualization
- **Model Info**: Real-time model status and performance metrics
- **Advanced Analytics**: 
  - Real-time monitoring dashboard
  - Prediction distribution charts
  - Model performance gauges
  - Feature importance visualization
  - System health monitoring


## 📁 Project Structure

```
fraud-detection-mlops/
├── api/
│   └── main.py                    # FastAPI backend application
├── app.py                          # Streamlit dashboard
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Data loading and ingestion
│   │   ├── cleaning.py            # Data cleaning pipeline
│   │   └── validation.py          # Pandera schema validation
│   ├── features/
│   │   └── engineering.py         # Feature creation & scaling
│   ├── models/
│   │   ├── train.py               # Model training (XGBoost)
│   │   ├── evaluate.py            # Model evaluation metrics
│   │   └── predict.py             # Prediction service
│   ├── monitoring/
│   │   └── metrics.py             # Custom metrics tracking
│   └── utils/
│       └── logger.py              # Logging configuration
├── data/
│   ├── raw/                       # Raw datasets (gitignored)
│   ├── processed/                 # Cleaned data
│   └── features/                  # Engineered features
├── models/                        # Trained models (gitignored)
│   ├── fraud_detection_model.pkl  # XGBoost model
│   └── scaler.pkl                 # StandardScaler
├── configs/
│   ├── config.yaml                # Main configuration
│   └── model_config.yaml          # Model parameters
├── run_full_pipeline.py           # Complete pipeline execution
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ recommended
- 4GB+ RAM
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-mlops.git
cd fraud-detection-mlops
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Place it in `data/raw/creditcard.csv`

### Running the Full Pipeline

```bash
# Run complete data pipeline and model training
python run_full_pipeline.py
```

This will execute:
1. ✅ Data ingestion (284,807 transactions)
2. ✅ Data cleaning and validation
3. ✅ Feature engineering (12 new features)
4. ✅ Model training with XGBoost
5. ✅ Model evaluation and saving

### Starting the Services

**Terminal 1 - FastAPI Backend:**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Streamlit Dashboard:**
```bash
streamlit run app.py
```

### Access the Application

- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 💻 Usage Examples

### 1. Single Prediction via Dashboard

1. Navigate to http://localhost:8501
2. Select "🔍 Single Prediction"
3. Enter transaction details (Time, Amount, V1-V28 features)
4. Click "Predict Fraud"
5. View fraud probability gauge and confidence score

### 2. Batch Predictions

1. Go to "📊 Batch Prediction" page
2. Upload CSV file with transactions
3. View fraud distribution charts
4. Download predictions

### 3. API Usage

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "Time": 12345,
    "Amount": 150.50,
    "V1": -1.359,
    "V2": -0.073,
    # ... V3 to V28
})

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
with open('transactions.csv', 'rb') as f:
    response = requests.post("http://localhost:8000/predict/batch",
                           files={'file': f})
```

### 4. Model Information

```python
# Get model metadata
response = requests.get("http://localhost:8000/model/info")
print(response.json())
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 97.14% |
| **Accuracy** | 99.92% |
| **Precision** | 73.87% |
| **Recall** | 83.67% |
| **F1-Score** | 78.47% |
| **PR-AUC** | 85.77% |

### Feature Engineering

**Time-based Features (6):**
- Hour of day
- Day of week
- Time since last transaction
- Transaction frequency
- Time-based statistics

**Amount-based Features (6):**
- Amount bins
- Amount z-score
- Amount percentile
- Rolling statistics
- Amount categories

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  type: xgboost
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    scale_pos_weight: 577  # For class imbalance
```

### API Configuration

Edit `api/main.py` for:
- CORS settings
- Model paths
- Logging levels
- Prometheus metrics


## 🔍 Monitoring & Observability

### Prometheus Metrics

The API exposes Prometheus metrics at `/metrics`:

- `fraud_detection_requests_total` - Total API requests
- `fraud_predictions_total` - Predictions by class (Fraud/Normal)
- `fraud_detection_request_duration_seconds` - Request latency histogram

### Analytics Dashboard

Navigate to the "📊 Analytics" page in Streamlit to view:

- **System Status**: API health, dataset size, version info
- **Real-time Monitoring**: Request count, fraud rate, prediction distribution
- **Model Performance**: ROC-AUC gauge, performance metrics
- **Feature Importance**: Top features contributing to predictions
- **ML Monitoring**: Data drift status, model drift tracking

## 🛠️ Development

### Project Dependencies

```txt
# Core ML
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
xgboost==2.1.3

# API & UI
fastapi==0.115.6
uvicorn==0.34.0
streamlit==1.40.2
pydantic==2.10.4

# Monitoring
prometheus-client==0.21.1
mlflow==2.19.0

# Visualization
plotly==5.24.1
seaborn==0.13.2
matplotlib==3.9.3

# Validation
pandera==0.20.4
```

### Adding New Features

1. **Create feature function** in `src/features/engineering.py`
2. **Update pipeline** in `run_full_pipeline.py`
3. **Retrain model** with new features
4. **Update API schema** in `api/main.py`

### Custom Model Training

```python
from src.models.train import ModelTrainer
from src.features.engineering import FeatureEngineer

# Load and prepare data
fe = FeatureEngineer()
X_train, y_train = fe.create_features(train_df)

# Train custom model
trainer = ModelTrainer()
model = trainer.train_custom_model(X_train, y_train, model_params)
trainer.save_model(model, 'custom_model.pkl')
```

## 🐛 Troubleshooting

### Common Issues

**1. Model not found error**
```bash
# Ensure models are trained
python run_full_pipeline.py
```

**2. Port already in use**
```bash
# Change port in commands
uvicorn api.main:app --port 8001
streamlit run app.py --server.port 8502
```

**3. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. Low memory**
```bash
# Process data in chunks (edit configs/config.yaml)
data:
  chunk_size: 10000
```

## 📚 Dataset Information

**Credit Card Fraud Detection Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1-V28 from PCA)
- **Target**: Class (0: Normal, 1: Fraud)
- **Imbalance**: 99.83% Normal, 0.17% Fraud

### Data Privacy
- Features V1-V28 are PCA-transformed for confidentiality
- No direct customer information included
- Safe for public research and development

## 🚀 Future Enhancements

### Planned Features

- [ ] **Real-time Streaming**: Kafka integration for live fraud detection
- [ ] **Model Explainability**: SHAP values and LIME explanations
- [ ] **A/B Testing**: Framework for comparing model versions
- [ ] **Data Drift Detection**: Evidently AI integration
- [ ] **Advanced Models**: Deep learning with PyTorch/TensorFlow
- [ ] **Cloud Deployment**: AWS/GCP/Azure deployment guides
- [ ] **Load Testing**: Performance benchmarking
- [ ] **GraphQL API**: Alternative API interface
- [ ] **Mobile App**: Flutter/React Native frontend
- [ ] **Notification System**: Email/SMS alerts for fraud detection

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **XGBoost**: Tianqi Chen and Carlos Guestrin
- **FastAPI**: Sebastián Ramírez
- **Streamlit**: Streamlit Team

## 📞 Contact

**Project Maintainer**: Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for production-grade ML systems**
