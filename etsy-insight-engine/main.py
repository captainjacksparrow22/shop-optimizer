"""
Etsy Insight Engine - Main Pipeline
"""
from data_collection import collect_all_data
from analysis import analyze_trends_and_keywords
from reporting import generate_report

if __name__ == "__main__":
    print("[Etsy Insight Engine] Starting pipeline...")
    all_data = collect_all_data()
    analysis_results = analyze_trends_and_keywords(all_data)
    generate_report(analysis_results)
    print("[Etsy Insight Engine] Done! See outputs folder for reports.")
