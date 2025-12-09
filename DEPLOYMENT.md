# Deployment Guide

Complete guide for deploying the Credit Card Fraud Detection system to production.

## 🚀 Deployment Options

### 1. Docker Deployment (Recommended)

#### Quick Start
```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### With Monitoring
```bash
# Start with Prometheus and Grafana
docker-compose --profile monitoring up -d

# Access services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

#### Production Build
```bash
# Build optimized image
docker build -t fraud-detection:latest .

# Tag for registry
docker tag fraud-detection:latest yourusername/fraud-detection:v1.0.0

# Push to Docker Hub
docker push yourusername/fraud-detection:v1.0.0
```

### 2. Cloud Deployment

#### AWS Deployment

**Option A: AWS ECS (Elastic Container Service)**

1. **Push to ECR**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name fraud-detection

# Tag and push
docker tag fraud-detection:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
```

2. **Create ECS Task Definition**
```json
{
  "family": "fraud-detection",
  "containerDefinitions": [
    {
      "name": "fraud-detection-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "memory": 2048,
      "cpu": 1024
    }
  ]
}
```

**Option B: AWS EC2**

```bash
# SSH into EC2 instance
ssh -i key.pem ec2-user@<instance-ip>

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Clone and run
git clone <repo-url>
cd fraud-detection-mlops
docker-compose up -d
```

**Option C: AWS Lambda (Serverless)**

Create a Lambda function with API Gateway:
```python
# lambda_function.py
import json
from src.models.predict import PredictionService

predictor = PredictionService()
predictor.load_model('fraud_detection_model.pkl')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    prediction = predictor.predict(body)
    return {
        'statusCode': 200,
        'body': json.dumps(prediction)
    }
```

#### Google Cloud Platform (GCP)

**Cloud Run Deployment**

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/fraud-detection

# Deploy to Cloud Run
gcloud run deploy fraud-detection \
  --image gcr.io/<project-id>/fraud-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

**GKE (Google Kubernetes Engine)**

```bash
# Create cluster
gcloud container clusters create fraud-detection-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

#### Microsoft Azure

**Azure Container Instances**

```bash
# Login
az login

# Create resource group
az group create --name fraud-detection-rg --location eastus

# Create container
az container create \
  --resource-group fraud-detection-rg \
  --name fraud-detection \
  --image yourusername/fraud-detection:latest \
  --dns-name-label fraud-detection-app \
  --ports 8000
```

### 3. Kubernetes Deployment

Create `kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
    spec:
      containers:
      - name: api
        image: yourusername/fraud-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /app/models/fraud_detection_model.pkl
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: fraud-detection
```

Deploy:
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl get services
```

### 4. Heroku Deployment

```bash
# Login to Heroku
heroku login

# Create app
heroku create fraud-detection-app

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

Create `Procfile`:
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

## 🔒 Security Best Practices

### 1. Environment Variables

Never commit sensitive data. Use environment variables:

```bash
# .env (DO NOT COMMIT)
API_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@host:5432/db
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

### 2. API Authentication

Add authentication to FastAPI:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(data: Transaction, api_key: str = Depends(get_api_key)):
    # Your code here
```

### 3. HTTPS/TLS

Use reverse proxy (Nginx) with SSL:

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Rate Limiting

Install and configure:
```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: Transaction):
    # Your code here
```

## 📊 Monitoring & Logging

### Production Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10000000,
    backupCount=5
)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Application Performance Monitoring (APM)

**New Relic:**
```bash
pip install newrelic
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program uvicorn api.main:app
```

**DataDog:**
```bash
pip install ddtrace
ddtrace-run uvicorn api.main:app
```

## 🔄 CI/CD Pipeline

GitHub Actions workflow is included (`.github/workflows/ci-cd.yml`)

### Secrets to Add in GitHub:

1. Go to Settings → Secrets and variables → Actions
2. Add:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

## 📈 Scaling Strategies

### Horizontal Scaling
- Use Kubernetes HPA (Horizontal Pod Autoscaler)
- AWS Auto Scaling Groups
- Load balancers

### Vertical Scaling
- Increase container resources
- Upgrade instance types

### Database Scaling
- Use connection pooling
- Implement caching (Redis)
- Read replicas

## ✅ Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Models trained and saved
- [ ] Docker images built
- [ ] Health checks working
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Security measures implemented
- [ ] Documentation updated
- [ ] Backup strategy in place

## 🆘 Troubleshooting

### Common Issues

**Out of Memory:**
```yaml
# Increase Docker memory limit
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

**Slow Predictions:**
```python
# Use batch prediction
# Cache frequent results
# Optimize model size
```

**Connection Timeouts:**
```python
# Increase timeout
uvicorn.run(app, timeout_keep_alive=120)
```

## 📞 Support

For deployment issues:
- Check logs: `docker-compose logs -f`
- Review health endpoint: `/health`
- Contact: your.email@example.com

---

**Ready for Production! 🚀**
