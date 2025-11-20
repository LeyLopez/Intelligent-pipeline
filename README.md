# 🤖 MLOps Final Project: Pipeline Inteligente

[![CI/CD](https://github.com/usuario/mlops-final-project/workflows/MLOps%20CI/CD%20Pipeline/badge.svg)](https://github.com/usuario/mlops-final-project/actions)

## 📋 Descripción

Sistema modular que integra tres tipos de modelos de IA:
- 🗣️ **LLM (Large Language Model)**: Conversación y procesamiento de lenguaje natural
- 📊 **ML Clásico**: Clasificación con scikit-learn (Random Forest)
- 🖼️ **CNN**: Clasificación de imágenes con redes convolucionales

## 🏗️ Arquitectura
```
┌─────────────────────────────────────────────┐
│         Gradio Frontend (Port 7860)         │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┼───────┐
       │       │       │
       ▼       ▼       ▼
   ┌─────┐ ┌─────┐ ┌─────┐
   │ LLM │ │ SKL │ │ CNN │
   │8001 │ │8002 │ │8003 │
   └──┬──┘ └──┬──┘ └──┬──┘
      │       │       │
      └───────┼───────┘
              ▼
       ┌────────────┐
       │   MLflow   │
       │  (5000)    │
       └────────────┘
```

## 🚀 Quick Start

### Prerrequisitos
```bash
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.10+
- 8GB RAM mínimo
```

### Instalación y Ejecución

**1. Clonar repositorio:**
```bash
git clone https://github.com/usuario/mlops-final-project.git
cd mlops-final-project
```

**2. Configurar variables de entorno:**
```bash
cp .env.example .env
# Editar .env según necesidad
```

**3. Desplegar en desarrollo:**
```bash
cd infra/scripts
chmod +x deploy-dev.sh
./deploy-dev.sh
```

**4. Acceder a los servicios:**
- 🎨 **Gradio UI**: http://localhost:7860
- 📊 **MLflow**: http://localhost:5000
- 🗣️ **LLM API**: http://localhost:8001/docs
- 📈 **Sklearn API**: http://localhost:8002/docs
- 🖼️ **CNN API**: http://localhost:8003/docs

## 📦 Servicios

### 1. LLM Connector
**Puerto**: 8001  
**Descripción**: Interfaz para interactuar con modelos de lenguaje (Ollama/LLaMA)

**Endpoints principales:**
- `POST /chat`: Conversación con el LLM
- `GET /health`: Estado del servicio

### 2. Sklearn Model
**Puerto**: 8002  
**Descripción**: Clasificador Random Forest entrenado con dataset Iris

**Endpoints principales:**
- `POST /predict`: Predicción sobre nuevos datos
- `GET /health`: Estado del servicio

### 3. CNN Image
**Puerto**: 8003  
**Descripción**: Clasificador de imágenes con 4 capas convolucionales

**Capacidades:**
- Clasificación de 3 clases: perro, gato, ave
- Aplicación de filtros: Gaussian Blur, Edge Detection, Sharpening, Emboss

**Endpoints principales:**
- `POST /classify`: Clasificar imagen
- `POST /apply-filters`: Aplicar filtros convolucionales
- `GET /info`: Información y limitaciones

### 4. Gradio Frontend
**Puerto**: 7860  
**Descripción**: Interfaz web interactiva que integra todos los servicios

## 🧪 Testing

### Ejecutar todos los tests:
```bash
cd infra/scripts
./run-tests.sh
```

### Tests individuales:
```bash
# Sklearn
cd sklearn_model
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=app --cov=pipeline
```

##
