#!/bin/bash

# Interactive chat script for MemoryAlpha RAG API
BASE_URL="http://localhost:8000"
THINKING_MODE="DISABLED"
MAX_TOKENS=512
TOP_K=5
TOP_P=0.8
TEMPERATURE=0.3

echo "🖖 Welcome to MemoryAlpha RAG Chat"
echo "Type 'quit' or 'exit' to end the session"
echo "----------------------------------------"

while true; do
    # Prompt for user input
    echo -n "❓ Ask about Star Trek: "
    read -r question
    
    # Check for exit commands
    if [[ "$question" == "quit" || "$question" == "exit" || "$question" == "q" ]]; then
        echo "🖖 Live long and prosper!"
        break
    fi
    
    # Skip empty questions
    if [[ -z "$question" ]]; then
        continue
    fi
    
    # URL encode the question
    encoded_question=$(printf '%s' "$question" | jq -sRr @uri)
    
    echo "🤖 LCARS Response:"
    echo "----------------------------------------"
    
    # Make the streaming request
    curl -s -N -H "Accept: text/event-stream" \
        "${BASE_URL}/memoryalpha/rag/stream?question=${encoded_question}&thinkingmode=${THINKING_MODE}&max_tokens=${MAX_TOKENS}&top_k=${TOP_K}&top_p=${TOP_P}&temperature=${TEMPERATURE}" \
        | while IFS= read -r line; do
            if [[ $line == data:* ]]; then
                chunk=$(echo "${line#data: }" | jq -r '.chunk // empty')
                if [[ -n "$chunk" ]]; then
                    printf "%s" "$chunk"
                fi
            fi
        done
    
    echo -e "\n----------------------------------------"
done
