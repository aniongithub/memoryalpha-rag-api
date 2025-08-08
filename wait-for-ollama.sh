#!/bin/bash
echo "✋ Waiting for Ollama at $OLLAMA_URL..."

until curl -s "$OLLAMA_URL/api/tags" > /dev/null; do
  sleep 1
done

echo "✅ Ollama is ready."

pull_model() {
  local model_name="$1"
  echo "🔍 Checking if model '$model_name' is available..."
  if curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$model_name\""; then
    echo "✅ Model '$model_name' is already available."
  else
    echo "📥 Model '$model_name' not found. Pulling it now..."
    curl -X POST "$OLLAMA_URL/api/pull" -H "Content-Type: application/json" -d "{\"name\":\"$model_name\"}"
    echo ""
    echo "✅ Model '$model_name' has been pulled successfully."
  fi
}

# Pull the default models
pull_model "$DEFAULT_MODEL"
pull_model "$DEFAULT_IMAGE_MODEL"

# Warm up ollama with the default model
echo "🤖 Warming up Ollama with $DEFAULT_MODEL..."
curl -s "$OLLAMA_URL/api/generate" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$DEFAULT_MODEL\", \"prompt\":\"Hello, Ollama!\"}" > /dev/null
echo "✅ Ollama is warmed up."

exec "$@"