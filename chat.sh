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

# Function to handle image identification
identify_image() {
    local image_path="$1"
    local tmpfile=""
    # Check if local file exists
    if [[ -f "$image_path" ]]; then
        tmpfile="$image_path"
    else
        # Try to download
        echo "Attempting to download image from URL: $image_path"
        tmpfile="/tmp/maimg_$$.img"
        if ! curl -sSL "$image_path" -o "$tmpfile"; then
            echo "Failed to download image. Returning to menu."
            [[ -f "$tmpfile" ]] && rm -f "$tmpfile"
            return
        fi
    fi
    echo "🤖 LCARS Image Identification:"
    echo "----------------------------------------"
    local response
    response=$(curl -s -X POST \
        -F "file=@${tmpfile}" \
        "${BASE_URL}/memoryalpha/rag/identify?top_k=${TOP_K}")
    local answer
    answer=$(echo "$response" | jq -r '.model_answer // empty')
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
    # Clean up temp file if downloaded
    if [[ "$tmpfile" != "$image_path" ]]; then
        rm -f "$tmpfile"
    fi
}

while true; do
    echo "Choose an option:"
    echo "  1) Ask a Star Trek question"
    echo "  2) Identify an image"
    echo "  q) Quit"
    echo -n "Enter choice [1/2/q]: "
    read -r choice
    case "$choice" in
        1)
            echo -n "❓ Enter your question: "
            read -r question
            if [[ -z "$question" ]]; then
                continue
            fi
            ask_question "$question"
            ;;
        2)
            echo -n "🖼️ Enter local image path or image URL: "
            read -r image_path
            if [[ -z "$image_path" ]]; then
                continue
            fi
            identify_image "$image_path"
            ;;
        q|quit|exit)
            echo "🖖 Live long and prosper!"
            break
            ;;
        *)
            echo "Invalid choice. Please enter 1, 2, or q."
            ;;
    esac
done
