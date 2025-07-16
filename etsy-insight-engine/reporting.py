import pandas as pd
import os

def generate_report(results):
    """
    Generate a CSV and simple text report of hot trends and product suggestions.
    """
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    etsy_df = results.get('etsy', pd.DataFrame())
    top_keywords = results.get('top_keywords', pd.Series())

    # Save full Etsy data with calculated metrics
    etsy_path = os.path.join(outputs_dir, 'etsy_listings_with_metrics.csv')
    etsy_df.to_csv(etsy_path, index=False)
    print(f"[Reporting] Saved Etsy listings with metrics to {etsy_path}")

    # Save top keywords
    kw_path = os.path.join(outputs_dir, 'top_keywords.csv')
    top_keywords.to_csv(kw_path, header=['count'])
    print(f"[Reporting] Saved top keywords to {kw_path}")

    # Print a simple text summary
    print("\n[Hot Keywords]")
    print(top_keywords.head(20))
    print("\n[Suggested Next Steps]")
    print("- Consider creating listings for trending keywords you are not currently offering.")
    print("- Review the etsy_listings_with_metrics.csv for high favorites/reviews per day.")
