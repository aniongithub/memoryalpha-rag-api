#!/bin/bash
echo "✋ Waiting for Ollama at $OLLAMA_URL..."

until curl -s "$OLLAMA_URL/api/tags" > /dev/null; do
  sleep 1
done

echo "✅ Ollama is ready."

pull_model() {
  local model_name="$1"
  if [ -z "$model_name" ]; then
    echo "⏭️  No model name provided, skipping."
    return 0
  fi
  echo "🔍 Checking if model '$model_name' is available..."
  if curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$model_name\""; then
    echo "✅ Model '$model_name' is already available."
  else
    echo "📥 Model '$model_name' not found. Pulling it now..."
    local response
    response=$(curl -s -X POST "$OLLAMA_URL/api/pull" -H "Content-Type: application/json" -d "{\"name\":\"$model_name\"}")
    echo ""
    if echo "$response" | grep -q '"error"'; then
      echo "❌ Failed to pull model '$model_name': $response"
      return 1
    fi
    echo "✅ Model '$model_name' has been pulled successfully."
  fi
}

# Pull the default models (DEFAULT_IMAGE_MODEL is optional and skipped if unset)
pull_model "$DEFAULT_MODEL"
pull_model "$DEFAULT_IMAGE_MODEL"

# Warm up ollama with the default model
echo "🤖 Warming up Ollama with $DEFAULT_MODEL..."
curl -s "$OLLAMA_URL/api/generate" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$DEFAULT_MODEL\", \"prompt\":\"Hello, Ollama!\"}" > /dev/null
echo "✅ Ollama is warmed up."

exec "$@"