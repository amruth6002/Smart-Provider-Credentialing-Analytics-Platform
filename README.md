# Smart Provider Credentialing Analytics Platform

A comprehensive healthcare data quality analytics platform that combines data validation, local AI-powered natural language processing, and interactive visualizations to help healthcare organizations monitor, audit, and improve provider data quality.

## Table of Contents
- [Key Benefits](#key-benefits)
- [Architecture](#architecture)
  - [Core Components](#core-components)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [AI Model Pipeline](#ai-model-pipeline)
  - [Models Used (Defaults)](#models-used-defaults)
- [Quick Start](#quick-start)
- [Features & Capabilities](#features--capabilities)
  - [Data Quality Analytics](#data-quality-analytics)
  - [Interactive Visualizations](#interactive-visualizations)
  - [AI-Powered Features](#ai-powered-features)
  - [Export & Reporting](#export--reporting)
  - [Scoring Methodology](#scoring-methodology)
- [Configuration](#configuration)
  - [Optional Environment Variables](#optional-environment-variables)
  - [Model Customization](#model-customization)
  - [Data Sources](#data-sources)
- [Performance](#performance)
- [Use Cases](#use-cases)
- [Technical Stack](#technical-stack)
- [Troubleshooting](#troubleshooting)


## Demo Video

Watch a quick demo of the platform in action:

[Demo video](https://www.youtube.com/watch?v=m1SQw90M-SM)

## Key Benefits
- No API keys required — runs fully local
- Cost-free usage with no external API charges
- Privacy-first — data never leaves your environment
- Works offline after initial model download

## Architecture

### Core Components
- Data Quality Engine (`src/engine.py`): Core analytics and validation
- Visualization Module (`src/visualizations.py`): Charts and graphs
- Gen-AI Processor (`src/genai.py`): Local NLP for intents and responses
- Dashboard (`ui/dashboard.py`): Streamlit-based UI

### Data Processing Pipeline
1. Ingestion: Load provider, license, and NPI data
2. Standardization: Clean and normalize formats
3. Validation: Apply rules and cross-reference sources
4. Quality Scoring: Compute composite metrics
5. AI Analysis: Natural language query understanding
6. Visualization: Interactive dashboards

### AI Model Pipeline
1. Rule-based processing for common queries
2. Semantic enhancement for intent detection
3. Context-aware response generation
4. Fallback system for robustness

### Models Used (Defaults)
- Sentence Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Text Generation (optional): `microsoft/DialoGPT-small`
  - https://huggingface.co/microsoft/DialoGPT-small
- Fallback: Template-based responses if generation model is unavailable
- Initialization: see `GenAIProcessor._initialize_local_models` in `src/genai.py`

## Quick Start

Quick Start index:
- Choose your path:
- Manual installation: [Manual Index](#installation) 
    - Go to [Method A: Streamlit (recommended)](#method-a-streamlit-recommended)
    - Go to [Method B: Run script](#method-b-run-script)
- Docker (containerized): [Docker](#docker) 


### Prerequisites
- Python 3.8+
- pip
- Git (to clone)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/amruth6002/Smart-Provider-Credentialing-Analytics-Platform.git
   cd Smart-Provider-Credentialing-Analytics-Platform
   ```

2. (Recommended) Create and activate a virtual environment:
   - Windows (Command Prompt or PowerShell):
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Key AI dependencies: `sentence-transformers`, `transformers`, `torch`.

### Run the App (Two Methods)

#### Method A: Streamlit (recommended)
```bash
streamlit run ui/dashboard.py
```

#### Method B: Run script
```bash
chmod +x run.sh
./run.sh
```

After starting, open your browser at:
http://localhost:8501

Note: A `.env` file is not required for default settings. You may add one if you want to override settings (see Configuration).

### Docker
Build and run:
```bash
docker build -f docker/Dockerfile -t provider-analytics .
docker run -p 8501:8501 provider-analytics
```
Then open http://localhost:8501

### First Run (Models)
On first use of AI features, the app will:
1. Download models (~100 MB, one-time)
2. Cache models locally
3. Fallback to rule-based processing if models are unavailable

### Load Sample Data
- Start the app, open the sidebar, and choose “Load sample data.”
- Explore the dashboard, then try example queries:
  - “How many providers have expired licenses?”
  - “Show me quality issues by specialty.”
  - “What’s our overall data quality score?”
  - “Show me phone formatting issues.”


## Features & Capabilities

### Data Quality Analytics
- License expiration tracking and compliance
- NPI validation and missing data detection
- Phone number format validation
- Duplicate provider detection
- Multi-state licensing analysis
- Specialty-based quality metrics

### Interactive Visualizations
- Quality score gauge
- Issue distribution charts
- State-wise analysis
- License expiration timelines
- Specialty performance analytics
- Duplicate analysis insights

### AI-Powered Features
- Natural language query processing
- Intelligent, context-aware responses
- Smart follow-up suggestions
- Conversational interface

### Export & Reporting
- CSV exports
- Compliance and analytics summaries
- Custom filtered views

### Scoring Methodology

Individual Score: For each provider, it's computed as 100 minus penalties for issues like expired licenses, missing NPI, phone formatting errors, duplicates, and state mismatches. See scoring.py for the formula.

Overall Score: The average of all individual scores across the dataset.

 Weights: Penalties are weighted as follows (from src/config.py):
- License issues: 35%
- Missing NPI: 25%
- Duplicates: 15%
- Contact format (e.g., phone): 15%
- Mismatches: 10%

## Configuration

### Optional Environment Variables
```bash
# Cache directory for downloaded models
AI_CACHE_DIR=./models_cache

# Enable debug logging
DEBUG=true
```

### Model Customization
Edit `src/genai.py` to:
- Swap `all-MiniLM-L6-v2` with another sentence-transformer
- Replace `microsoft/DialoGPT-small` with another generator
- Adjust similarity thresholds for intent classification

### Data Sources
- Provider Roster (primary dataset)
- License Databases (e.g., state-specific)
- NPI Registry

## Performance
- RAM: ~500 MB (models loaded)
- Storage: ~100 MB (model cache)
- Cold start: ~2–3 s (first query with model load)
- Warm queries: typically <100 ms
- Rule-based fallback: typically <10 ms

## Use Cases
- Credentialing compliance monitoring
- Pre-audit data quality checks
- Provider directory accuracy
- License expiration tracking
- Trend monitoring for data quality teams

## Technical Stack
- Backend: Python, Pandas, DuckDB
- Frontend: Streamlit
- Visualizations: Plotly, Altair
- AI/ML: `sentence-transformers`, `transformers` (local)
- Data Processing: Pandas, NumPy

## Troubleshooting

### Models Not Loading
- Check your internet connection (first download only)
- Ensure enough disk space (~500 MB)
- Set `AI_CACHE_DIR` to a writable location if needed

### Performance Issues
- Ensure adequate RAM (2 GB+ recommended)
- Consider model quantization for constrained environments
- Use rule-based fallback for fastest responses

### Error Handling
- The system provides clear feedback, graceful degradation, and automatic fallback to rule-based responses when AI models aren’t available.
- If the error persists and unable to resolve kindly raise an issue or contact the authors/contributors.



