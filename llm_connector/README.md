# LLM Connector

## Descripción
Servicio para interactuar con modelos de lenguaje (Ollama/LLaMA).

## Configuración

### Variables de Entorno
\`\`\`
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_API_URL=http://localhost:11434
\`\`\`

## Endpoints

### POST /generate
**Request:**
\`\`\`json
{
  "prompt": "¿Qué es MLOps?",
  "context": "opcional",
  "max_tokens": 500
}
\`\`\`

**Response:**
\`\`\`json
{
  "response": "MLOps es...",
  "model": "llama2",
  "status": "success"
}
\`\`\`

### GET /health
Verifica estado del servicio.

## Ejecución Local
\`\`\`bash
pip install -r requirements.txt
python -m app.main
\`\`\`

## Docker
\`\`\`bash
docker build -t llm-connector .
docker run -p 8001:8001 llm-connector
\`\`\`