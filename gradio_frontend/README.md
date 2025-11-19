# Gradio Frontend

## Descripción
Interfaz web interactiva que integra los tres servicios del pipeline MLOps.

## Características
- Chat con LLM
- Clasificación con modelo sklearn
- Clasificación de imágenes con CNN
- ℹInformación del sistema

## Variables de Entorno
```bash
LLM_SERVICE_URL=http://llm-connector:8001
SKLEARN_SERVICE_URL=http://sklearn-model:8002
CNN_SERVICE_URL=http://cnn-image:8003
```

## Ejecución
```bash
pip install -r requirements.txt
python app.py
```

Acceder a: http://localhost:78