set -e

echo "Building Docker images..."

echo "Building LLM Connector image..."
docker build -t mlops-llm-connector:latest ../llm_connector/

echo "Building Sklearn Model image..."
docker build -t mlops-sklearn-model:latest ../sklearn_model/

echo "Building CNN Image Model image..."
docker build -t mlops-cnn-image:latest ../cnn_image/

echo "Building Gradio Frontend image..."
docker build -t mlops-gradio-frontend:latest ../gradio_frontend/

echo "All Docker images built successfully."
docker images | grep mlops