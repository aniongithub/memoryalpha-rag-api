#!/bin/bash

# Interactive chat script for MemoryAlpha RAG API
BASE_URL="http://localhost:8000"
THINKING_MODE="DISABLED"
MAX_TOKENS=2048
TOP_K=5
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
        "${BASE_URL}/memoryalpha/rag/ask?question=${encoded_question}&thinkingmode=${THINKING_MODE}&max_tokens=${MAX_TOKENS}&top_k=${TOP_K}&top_p=${TOP_P}&temperature=${TEMPERATURE}")
    local answer
    answer=$(echo "$response" | jq -r '.response // empty')
    if [[ -n "$answer" ]]; then
        printf "%s\n" "$answer"
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
