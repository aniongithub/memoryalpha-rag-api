#!/bin/bash

# Get the default model from environment variable
DEFAULT_MODEL=${DEFAULT_MODEL:-"qwen3:0.5b"}

echo "🛠️ Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
until curl -s http://localhost:11434 > /dev/null; do
  echo "⏳ Waiting for Ollama to start..."
  sleep 1
done

echo "📥 Pulling model ${DEFAULT_MODEL}..."
ollama pull "${DEFAULT_MODEL}"

echo "✅ Model ready. Keeping Ollama running..."
wait $OLLAMA_PID
