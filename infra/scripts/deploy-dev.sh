set -e

echo "Deploying development environment..."

./build-images.sh

cd ..
docker-compose -f infra/docker-compose.yml up -d


echo "Waiting for services to be ready..."
sleep 15 

echo "Waiting for Ollama..."
for i in {1..60}; do
    if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready"
        break
    fi
    echo "  Trying... ($i/60)"
    sleep 2
done

echo ""
echo "Waiting for other services..."
sleep 15

echo "Verifying services health..."
curl -f http://localhost:5000/health && echo "MLflow is up."
curl -f http://localhost:8001/health && echo "LLM Connector is up."
curl -f http://localhost:8002/health && echo "Sklearn Model is up."
curl -f http://localhost:8003/health && echo "CNN Image Model is up."
curl -f http://localhost:7860 && echo "Gradio Frontend is up."


echo ""
echo "Deployment completed successfully!"
echo "Ollama:  http://localhost:11434"
echo "MLflow: http://localhost:5000"
echo "Gradio: http://localhost:7860"
echo ""
echo "Note: The llama2 model is downloading in the background."
echo "First run may take 5-10 minutes."
echo ""
echo "View logs: docker-compose logs -f ollama"