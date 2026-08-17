#!/bin/bash

# Interactive chat script for MemoryAlpha RAG API
# Load vars from .env. chat.sh runs *inside* the container, so it targets the
# container-internal APP_PORT (not the host-published API_PORT).
if [ -f .env ]; then set -a; . ./.env; set +a; fi
# Container listens on APP_PORT (default 8000). Set RAG_API_URL to override
# entirely (e.g. http://localhost:${API_PORT} when running from the host).
BASE_URL="${RAG_API_URL:-http://localhost:${APP_PORT:-8000}}"
MAX_TOKENS=2048
TOP_K=10
TOP_P=0.8
TEMPERATURE=0.3

echo "🖖 Welcome to MemoryAlpha RAG Chat"
echo "Type 'quit' or 'exit' to end the session"
echo "----------------------------------------"

# Function to handle text question
ask_question() {
    local question="$1"
    local encoded_question
    encoded_question=$(printf '%s' "$question" | jq -sRr @uri)
    echo "🤖 LCARS Response:"
    echo "----------------------------------------"
    local response
    response=$(curl -s \
        "${BASE_URL}/memoryalpha/rag/ask?question=${encoded_question}&max_tokens=${MAX_TOKENS}&top_k=${TOP_K}&top_p=${TOP_P}&temperature=${TEMPERATURE}")
    
    # Check if response is valid JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        printf "Error: Invalid response received.\n"
        printf "Raw response: %s\n" "$response"
        echo "----------------------------------------"
        return
    fi
    
    local answer
    answer=$(echo "$response" | jq -r '.answer // empty')
    if [[ -n "$answer" ]]; then
        printf "%s\n" "$answer"
        
        # Display token usage if available
        local input_tokens output_tokens total_tokens
        input_tokens=$(echo "$response" | jq -r '.token_usage.input_tokens // empty')
        output_tokens=$(echo "$response" | jq -r '.token_usage.output_tokens // empty')
        total_tokens=$(echo "$response" | jq -r '.token_usage.total_tokens // empty')
        
        if [[ -n "$input_tokens" && -n "$output_tokens" && -n "$total_tokens" ]]; then
            echo
            printf "📊 Token Usage: Input: %s | Output: %s | Total: %s\n" "$input_tokens" "$output_tokens" "$total_tokens"
        fi
    else
        local error
        error=$(echo "$response" | jq -r '.error // empty')
        if [[ -n "$error" ]]; then
            printf "Error: %s\n" "$error"
        else
            printf "No response received.\n"
        fi
    fi
    echo "----------------------------------------"
}

# Main question loop
while true; do
    echo -n "❓ Enter your Star Trek question (or 'quit' to exit): "
    read -r question
    if [[ "$question" == "quit" || "$question" == "exit" ]]; then
        echo "🖖 Live long and prosper!"
        break
    fi
    if [[ -z "$question" ]]; then
        continue
    fi
    ask_question "$question"
done
