#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1

# Check if .env file exists, if not copy from example
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
fi

# Enhanced UI with AI and Visualizations
echo "Starting Smart Provider Credentialing Analytics Platform..."
echo "Enhanced with AI-powered insights and interactive dashboards"
streamlit run ui/dashboard.py