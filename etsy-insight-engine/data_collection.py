import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load API keys and config from .env
load_dotenv()
ETSY_API_KEY = os.getenv('ETSY_API_KEY')
ETSY_SHOP_ID = os.getenv('ETSY_SHOP_ID')

# --- Data Collection Functions ---
def collect_etsy_listings():
    """Collects active listings from Etsy shop via API."""
    if not ETSY_API_KEY or not ETSY_SHOP_ID:
        print("[WARN] Etsy API key or shop ID not set. Skipping Etsy API collection.")
        return pd.DataFrame()
    url = f"https://openapi.etsy.com/v3/application/shops/{ETSY_SHOP_ID}/listings/active"
    headers = {"x-api-key": ETSY_API_KEY}
    params = {"limit": 100}
    listings = []
    while url:
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"[ERROR] Etsy API error: {resp.status_code}")
            break
        data = resp.json()
        listings.extend(data.get('results', []))
        url = data.get('pagination', {}).get('next_page_url')
        params = None  # Only needed for first request
    return pd.DataFrame(listings)

def collect_google_trends():
    """Stub for Google Trends collection (implement with pytrends or scraping)."""
    return pd.DataFrame()

def collect_alura_trends():
    """Stub for Alura trends scraping (implement as needed)."""
    return pd.DataFrame()

def collect_all_data():
    """Collects all relevant data sources and returns as a dict of DataFrames."""
    print("[Data Collection] Gathering Etsy listings...")
    etsy_df = collect_etsy_listings()
    print(f"[Data Collection] {len(etsy_df)} listings collected from Etsy.")
    # Add more sources as needed
    google_trends_df = collect_google_trends()
    alura_df = collect_alura_trends()
    return {
        "etsy": etsy_df,
        "google_trends": google_trends_df,
        "alura": alura_df
    }
