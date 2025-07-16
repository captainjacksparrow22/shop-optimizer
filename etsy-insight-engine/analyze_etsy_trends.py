import requests
from bs4 import BeautifulSoup
import csv
from collections import Counter
from PIL import Image
from io import BytesIO
import re
import google.generativeai as genai

# Install required libraries
try:
    import requests
    from bs4 import BeautifulSoup
    from PIL import Image
    import google.generativeai as genai
except ImportError:
    print("Installing required libraries...")
    import os
    os.system("pip install requests beautifulsoup4 pillow google-generativeai")

# Add your Gemini API key here
API_KEY = "AIzaSyDQxvpunp7e9ZwA-XmSfcR-ZGUiGeV9BGU"
genai.configure(api_key=API_KEY)

# Function to fetch and analyze Etsy page
def analyze_etsy_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch the page. Status code:", response.status_code)
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract listing titles
    titles = [title.text.strip() for title in soup.find_all('h3', class_='v2-listing-card__title')]

    # Extract image URLs
    image_urls = [img['src'] for img in soup.find_all('img', class_='v2-listing-card__img') if 'src' in img.attrs]

    # Initialize AI model (using a placeholder model name)
    # You might need to choose a specific model depending on your needs (e.g., 'gemini-pro-vision' for images)
    try:
        model = genai.GenerativeModel('gemini-pro-vision') # Or another suitable model
    except Exception as e:
        print(f"Failed to initialize AI model: {e}")
        print("Proceeding with basic analysis without AI.")
        model = None

    # Analyze keywords and themes in titles using AI (placeholder)
    ai_keywords = Counter()
    if model:
        print("Analyzing titles with AI...")
        for title in titles:
            try:
                # Placeholder for AI call to analyze title text
                # response = model.generate_content(f"Analyze keywords and themes in this Etsy listing title: {title}")
                # Process AI response and update ai_keywords counter
                ai_keywords['placeholder_ai_keyword'] += 1 # Example update
            except Exception as e:
                print(f"AI analysis failed for title '{title}': {e}")

    # Analyze image styles and themes using AI (placeholder)
    ai_image_styles = Counter()
    if model:
        print("Analyzing images with AI...")
        for url in image_urls:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                # Placeholder for AI call to analyze image
                # response = model.generate_content(["Analyze the style and theme of this Etsy listing image.", img])
                # Process AI response and update ai_image_styles counter
                ai_image_styles['placeholder_ai_style'] += 1 # Example update
            except Exception as e:
                print(f"AI analysis failed for image '{url}': {e}")

    # Combine traditional keyword analysis with AI insights
    all_keywords = Counter(re.findall(r'\b\w+\b', ' '.join(titles).lower()))
    # You might want to merge or compare all_keywords, ai_keywords, and ai_image_styles

    # Create trend report CSV
    with open('trend_report_ai.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Update header to include potential AI insights
        writer.writerow(['Keyword', 'Frequency', 'AI Theme (Titles)', 'AI Style (Images)'])

        # Placeholder: Write combined data to CSV
        # This part needs to be adapted based on how you process and combine AI results
        for keyword, freq in all_keywords.most_common():
             writer.writerow([keyword, freq, '', '']) # Example: just writing traditional keywords

        # Example: Writing AI keywords (you'd need to decide how to structure this)
        # for ai_kw, ai_freq in ai_keywords.most_common():
        #     writer.writerow([ai_kw, '', ai_freq, ''])

        # Example: Writing AI image styles (you'd need to decide how to structure this)
        # for ai_style, ai_style_freq in ai_image_styles.most_common():
        #     writer.writerow(['', '', '', ai_style, ai_style_freq])

    print("AI-enhanced trend report created: trend_report_ai.csv")

# Example usage
if __name__ == "__main__":
    etsy_url = input("Enter Etsy page URL: ")
    analyze_etsy_page(etsy_url)
