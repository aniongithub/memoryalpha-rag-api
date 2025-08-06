# wait-for-ollama.sh
#!/bin/bash
echo "✋ Waiting for Ollama at $OLLAMA_URL..."

until curl -s "$OLLAMA_URL/api/tags" > /dev/null; do
  sleep 1
done

echo "✅ Ollama is ready."

# Check if DEFAULT_MODEL is available, if not, pull it
echo "🔍 Checking if model '$DEFAULT_MODEL' is available..."
if curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$DEFAULT_MODEL\""; then
  echo "✅ Model '$DEFAULT_MODEL' is already available."
else
  echo "📥 Model '$DEFAULT_MODEL' not found. Pulling it now..."
  curl -X POST "$OLLAMA_URL/api/pull" -H "Content-Type: application/json" -d "{\"name\":\"$DEFAULT_MODEL\"}"
  echo ""
  echo "✅ Model '$DEFAULT_MODEL' has been pulled successfully."
fi

exec "$@"

# Warm up ollama with the default model
echo "🤖 Warming up Ollama with $DEFAULT_MODEL..."
curl -s "$OLLAMA_URL/api/generate" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$DEFAULT_MODEL\", \"prompt\":\"Hello, Ollama!\"}" > /dev/null
echo "✅ Ollama is warmed up."