#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Shutting down..."
    kill $API_PID
    exit
}

trap cleanup SIGINT

# Start API Server
echo "Starting API Server..."
./.venv/bin/uvicorn api_server.main:app --reload --port 8000 &
API_PID=$!

# Wait for API to be ready (simple sleep for now)
sleep 3

# Start Streamlit UI
echo "Starting Web UI..."
echo "--------------------------------------------------"
echo "👉 OPEN THIS LINK FOR THE APP: http://localhost:8501"
echo "--------------------------------------------------"
echo "(API Server is running in background on port 8000)"
./.venv/bin/streamlit run web_ui.py
