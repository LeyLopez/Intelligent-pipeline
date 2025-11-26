# README — Intelligent Pipeline (Proyecto MLOps)

Este repositorio contiene una pipeline modular de servicios para experimentación MLOps. Está pensado como un ejemplo práctico que integra modelos de lenguaje (LLM), modelos clásicos de machine learning (scikit-learn), y modelos de visión por computador (CNN), con una interfaz unificada en Gradio y tracking con MLflow.

Contenido principal del repositorio
- `cnn_image/` — Servicio FastAPI que sirve un clasificador CNN (TensorFlow/Keras). Incluye `app/` con `main.py`, `model.py`, y filtros en `filters/`.
- `sklearn_model/` — Servicio FastAPI que sirve un modelo de scikit-learn (Random Forest). Contiene `app/`, `models/` y tests en `tests/`.
- `llm_connector/` — Servicio FastAPI que actúa como conector a LLMs (configurable para Ollama u otros proveedores).
- `gradio_frontend/` — Frontend en Gradio que agrupa las llamadas a los distintos servicios y ofrece una UI interactiva.
- `infra/` — Infraestructura y despliegue: `docker-compose.yml`, stack para swarm y scripts útiles en `infra/scripts/`.
- `README.md` — Este documento.

Objetivo
 - Proveer un entorno para experimentar con varios tipos de modelos y demostrar una arquitectura básica de MLOps: servicios independientes, frontend, y tracking (MLflow).

Cómo ejecutar el proyecto
Las instrucciones muestran dos formas: desplegar todo con Docker Compose (recomendado para reproducibilidad) o ejecutar servicios localmente en modo desarrollo.

1) Ejecutar todo con Docker Compose (recomendado)

Desde la raíz del proyecto, usa el `docker-compose.yml` en `infra/` para levantar todos los servicios (ollama, mlflow, llm-connector, sklearn-model, cnn-image, gradio-frontend).

PowerShell:

```powershell
# Desde la raíz del repositorio
cd infra
docker-compose -f docker-compose.yml up --build -d

# Ver logs (ejemplo)
docker-compose -f docker-compose.yml logs -f gradio-frontend

# Para parar y eliminar contenedores + volúmenes
docker-compose -f docker-compose.yml down -v
```

Notas:
- MLflow estará en http://localhost:5000
- Gradio UI en http://localhost:7860
- LLM Connector en http://localhost:8001
- Sklearn API en http://localhost:8002
- CNN API en http://localhost:8003

2) Ejecutar cada servicio localmente (modo desarrollo)

Recomendación general: usar un entorno virtual por servicio.

Ejemplo (PowerShell) — servicio CNN:

```powershell
# Ir al directorio del servicio (desde la raíz del repo)
cd cnn_image

# Crear y activar entorno virtual (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Levantar el servicio con uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

Servicio Sklearn (puerto 8002):

```powershell
cd sklearn_model
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Servicio LLM Connector (puerto 8001):

```powershell
cd llm_connector
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Gradio frontend (local):

```powershell
cd gradio_frontend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
# O abrir http://localhost:7860
```

3) Comandos útiles para desarrollo y despliegue

- Construir imágenes localmente (script provisto):

```powershell
cd infra\scripts
./build-images.sh
```

- Desplegar en desarrollo con script (si existe):

```powershell
cd infra\scripts
./deploy-dev.sh
```

- Ejecutar tests (script):

```powershell
cd infra\scripts
./run-tests.sh
```

O ejecutar pytest directo en el servicio que contiene tests (ej. `sklearn_model`):

```powershell
cd sklearn_model
.\.venv\Scripts\Activate.ps1
pytest tests/ -v
```

Detalles por componente

- `cnn_image/` — Servicio FastAPI con endpoints principales:
        - `GET /health` — estado
        - `GET /info` — info del modelo y limitaciones
        - `POST /classify` — enviar imagen para clasificar
        - `POST /apply-filters` — aplicar filtros convolucionales

- `sklearn_model/` — Servicio FastAPI:
        - `GET /health` — estado
        - `POST /predict` — predecir con el modelo cargado (espera JSON con features)

- `llm_connector/` — Servicio FastAPI:
        - `GET /health` — chequea disponibilidad del motor LLM (p.ej. Ollama)
        - `POST /chat` — generar respuesta a partir de un prompt

- `gradio_frontend/` — Interfaz de usuario que llama a los servicios anteriores. Está diseñada para ejecutarse como aplicación local o en un contenedor.

Consideraciones y notas
- Variables de entorno: muchos servicios leen configuraciones desde variables (ver `app/config.py` en cada servicio). Si ejecutas con Docker Compose, las variables en `infra/docker-compose.yml` ya configuran los servicios principales.
- Modelos en disco: las rutas de modelos y preprocesadores se configuran en cada servicio (por ejemplo `settings.model_path`). Asegúrate de tener los artefactos en las rutas esperadas si vas a servir modelos preentrenados.
- Recursos: algunos servicios (especialmente Ollama y modelos de ML) requieren memoria/CPU; ajustar límites en `docker-compose.yml` si trabajas en una máquina con recursos limitados.

Siguientes pasos sugeridos (opcionales)
- Documentar variables de entorno en un `.env.example` raíz (si aún no existe).
- Añadir instrucciones de entrenamiento reproducible y un ejemplo de dataset si quieres compartir cómo generar los modelos locales.


