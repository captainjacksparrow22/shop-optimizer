import pandas as pd
from utils import extract_keywords

def analyze_trends_and_keywords(data):
    """
    Analyze Etsy and other data for trends and keyword performance.
    Returns a dict with trend and keyword analysis results.
    """
    etsy_df = data.get('etsy', pd.DataFrame())
    if etsy_df.empty:
        print("[Analysis] No Etsy data to analyze.")
        return {}

    # Example: Calculate favorites-per-day and reviews-per-day
    if 'creation_tsz' in etsy_df.columns and 'num_favorers' in etsy_df.columns:
        etsy_df['favorites_per_day'] = etsy_df['num_favorers'] / ((pd.Timestamp.now() - pd.to_datetime(etsy_df['creation_tsz'], unit='s')).dt.days + 1)
    if 'num_reviews' in etsy_df.columns and 'creation_tsz' in etsy_df.columns:
        etsy_df['reviews_per_day'] = etsy_df['num_reviews'] / ((pd.Timestamp.now() - pd.to_datetime(etsy_df['creation_tsz'], unit='s')).dt.days + 1)

    # Extract and rank keywords from titles and tags
    etsy_df['keywords'] = etsy_df.apply(lambda row: extract_keywords(row), axis=1)
    keyword_counts = pd.Series([kw for kws in etsy_df['keywords'] for kw in kws]).value_counts()
    top_keywords = keyword_counts.head(30)

    # TODO: Add trend detection and time-based keyword analysis

    return {
        'etsy': etsy_df,
        'top_keywords': top_keywords
    }
