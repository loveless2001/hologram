#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Shutting down..."
    kill $SERVER_PID
    exit
}

trap cleanup SIGINT

# Start Hologram Server
echo "Starting Hologram Server..."
./.venv/bin/python -m hologram.server &
SERVER_PID=$!

# Wait for server to be ready
sleep 3

# Start Streamlit UI
echo "Starting Web UI..."
echo "--------------------------------------------------"
echo "👉 OPEN THIS LINK FOR THE APP: http://localhost:8501"
echo "--------------------------------------------------"
echo "(Hologram Server is running in background on port 8000)"
./.venv/bin/streamlit run web_ui.py
