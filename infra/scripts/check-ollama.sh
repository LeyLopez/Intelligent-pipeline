echo "Verifying Ollama..."
echo ""


if docker ps | grep -q "mlops-ollama"; then
    echo "Ollama container is running"
else
    echo "Ollama container is NOT running"
    exit 1
fi


echo ""
echo "Verifying HTTP endpoint..."
if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Endpoint accessible"
else
    echo "Endpoint not accessible"
    exit 1
fi


echo ""
echo "Available models:"
curl -s http://localhost:11434/api/tags | jq '.models[] | .name'

 
echo ""
echo "Testing request..."
response=$(curl -s -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "llama2",
    "prompt": "Hola",
    "stream": false
  }')

if echo "$response" | grep -q "response"; then
    echo "Ollama responding correctly"
    echo "Response: $(echo $response | jq -r '.response' | head -c 30)..."
else
    echo "Ollama not responding"
fi

echo ""
echo "Verification completed"