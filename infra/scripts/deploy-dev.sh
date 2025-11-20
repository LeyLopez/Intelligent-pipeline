set -e

echo "Deploying development environment..."

./build-images.sh

cd ..
docker-compose -f infra/docker-compose.yml up -d


echo "Waiting for services to be ready..."
sleep 15 

echo "Verifying services health..."
curl -f http:localhost:5000/health && echo "MLflow is up."
curl -f http:localhost:8001/health && echo "LLM Connector is up."
curl -f http:localhost:8002/health && echo "Sklearn Model is up."
curl -f http:localhost:8003/health && echo "CNN Image Model is up."
curl -f http:localhost:7860/health && echo "Gradio Frontend is up."


echo ""
echo "Deployment completed successfully. Access the Gradio frontend at http://localhost:7860"
echo "Access MLflow UI at http://localhost:5000"
