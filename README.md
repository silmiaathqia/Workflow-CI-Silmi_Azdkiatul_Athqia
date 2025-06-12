# Workflow-CI-Silmi_Azdkiatul_Athqia

# Worker Productivity Classification - MLflow CI/CD

Repositori ini mengimplementasikan **Kriteria 3 Level Advance** untuk submission Machine Learning Operations dengan MLflow Project dan GitHub Actions CI/CD.

## 🗂️ Struktur Repository

```
Workflow-CI-Silmi_Azdkiatul_Athqia/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLProject              # MLflow project config
│   ├── conda.yaml             # Conda environment
│   ├── modelling.py           # Enhanced training script
│   ├── requirements.txt       # Python dependencies
│   └── processed_data/        # Dataset (from Kriteria 1)
├── README.md
└── .gitignore
```

## 🚀 Fitur CI/CD Pipeline

### Automated Workflows

1. **Training Pipeline**: Otomatis training model ketika push ke `main` atau `develop`
2. **Docker Build**: Build dan push Docker image ke Docker Hub
3. **Artifact Management**: Simpan model artifacts ke GitHub Actions
4. **Model Registry**: Register model ke DagsHub MLflow
5. **Multi-Environment**: Support manual trigger dengan custom parameters

### Enhanced MLflow Tracking

- **Manual Logging**: Implementasi manual logging (bukan autolog)
- **Additional Metrics**: 10+ metrics termasuk Matthews Correlation, Cohen's Kappa
- **Visualizations**: Confusion matrix, metrics comparison, class-wise performance
- **Model Signature**: Automatic model signature inference
- **DagsHub Integration**: Online MLflow tracking

## 🐳 Docker Deployment

### Quick Start

```bash
# Pull dan jalankan model
docker pull silmiathqia/worker-productivity-mlp:latest
docker run -p 8080:8080 silmiathqia/worker-productivity-mlp:latest

# Test endpoint
curl http://localhost:8080/health
```

### Docker Hub Repository

- **Repository**: [silmiathqia/worker-productivity-mlp](https://hub.docker.com/r/silmiathqia/worker-productivity-mlp)
- **Tags**: `latest`, `<commit-sha>`
- **Built with**: MLflow `models build-docker`

## 📊 Model
