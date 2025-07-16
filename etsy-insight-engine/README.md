# Etsy Insight Engine

A modular Python system to identify high-potential product trends for Etsy sellers.

## Features
- Collects Etsy marketplace data (API/scraping)
- Integrates with Google Keywords, Alura, and other sources
- Analyzes listing performance (favorites-per-day, reviews-per-day, etc.)
- Extracts and ranks keywords from titles/tags
- Detects emerging trends by tracking keyword popularity
- Suggests new listing ideas based on market gaps
- Generates daily/weekly reports
- Optional: Streamlit dashboard, scheduled jobs, AI-powered listing generation

## Getting Started
1. Clone this repo and create a Python virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
2. Add your Etsy API keys and any other credentials to `.env` (see `.env.example`).
3. Run the main pipeline:
   ```sh
   python main.py
   ```

## Folder Structure
- `main.py` — Entry point for the pipeline
- `data_collection.py` — Scraping/API logic for Etsy and other sources
- `analysis.py` — Trend and keyword analysis
- `reporting.py` — Report generation
- `utils.py` — Helper functions
- `data/` — Collected and processed data (CSV/SQLite)
- `outputs/` — Generated reports

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Extending
- Add new data sources by extending `data_collection.py`
- Add new analysis modules in `analysis.py`
- Add dashboard or AI features as needed

## License
MIT
