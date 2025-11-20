set -e

echo "Waiting for ollama to be available..."
for i in {1..60}; do
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is available!"
        break
    fi
        echo "Trying... ($i/60)"
        sleep 1
done


echo ""
echo "Verifiying ollama installation..."
models=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)


if echo "$models" | grep -q "llama2"; then
    echo "Llma2 model already exists. Skipping download."
else
    echo "Downloading llama2 model (first run, may take 5-10 minutes)..."
    curl -X POST http://localhost:11434/api/pull -d '{"name": "llama2"}' --no-buffer
    echo ""
    echo "Llama2 model downloaded successfully"
fi

echo ""
echo "Ollama initialized successfully"
echo "Available models: $models"