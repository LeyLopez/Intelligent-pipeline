set -e

echo "Deploying production environment..."

if ! docker info | grep -1 "Swarm: active"; then
    echo "Docker Swarm is not initialized. Initializing now..."
    docker swarm init
fi


./build-images.sh

cd ..
docker stack deploy -c infra/swarm-stack.yml mlops-stack

echo "Waiting for services to be ready..."
sleep 20

echo "Verifying services status..."
docker stack services mlops-stack

echo ""
echo "Deployment completed successfully."
echo "Monitor with: docker stack ps mlops-stack"
