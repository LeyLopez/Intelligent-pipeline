set -e

echo "Running tests..."

echo "Testing Sklearn Model Service..."
cd ../sklearn_model
python -m pytest tests/ -v

echo ""
echo "All tests completed."