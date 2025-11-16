# MLOps Final Project: Pipeline Inteligente

## Descripción
Sistema modular que integra:
- LLM (Language Model)
- ML Clásico (scikit-learn)
- CNN (Convolutional Neural Network)
- Frontend (Gradio)

#Quick Start

### Desarrollo Local
\`\`\`bash
docker-compose -f infra/docker-compose.yml up --build
\`\`\`

### Producción (Swarm)
\`\`\`bash
docker stack deploy -c infra/swarm-stack.yml mlops-stack
\`\`\`

## Servicios

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Gradio Frontend | 7860 | Interfaz de usuario |
| LLM Connector | 8001 | API de lenguaje |
| Sklearn Model | 8002 | Clasificación/Regresión |
| CNN Image | 8003 | Clasificación de imágenes |
| MLflow | 5000 | Tracking de experimentos |

## 👥 Autores
- LeyLopez
- 16/11/2025


