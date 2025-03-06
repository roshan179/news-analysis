import re
import html
import psycopg2
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
from psycopg2.extras import execute_batch

def clean_text(text):
    #clean text without webscraped tags
    """Cleans raw HTML/text data for NLP processing."""
    text = html.unescape(text)  # Decode HTML entities
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric characters
    # text = text.lower()  # Convert to lowercase
    return text

def processing_ready_text(text):
    # Tokenization & Stopword Removal
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# URL = f"https://content.guardianapis.com/search?api-key={API_KEY}&show-fields=body"

API_KEY = "845a8aa3-afd9-448d-9c4a-c01383a9352c" #API KEY for The Guardian
BASE_URL = "https://content.guardianapis.com/search"

# 🟢 Fetch News Function
def fetch_news(start_date=None, end_date=None, num_articles=50):
    articles = []
    page = 1
    page_size = 50

    if not start_date:
        start_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")

    while len(articles) < num_articles:
        params = {
            "api-key": API_KEY,
            "show-fields": "body",
            "page-size": page_size,
            "page": page,
            "from-date": start_date,
            "to-date": end_date
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "response" not in data or "results" not in data["response"]:
            print("❌ Error fetching news.")
            break

        news_results = data["response"]["results"]

        for article in news_results:
            if len(articles) >= num_articles:
                break

            articles.append({
                "title": article.get("webTitle", "No Title"),
                "raw_content": article.get("fields", {}).get("body", "No Content"),
                "url": article.get("webUrl", "No URL"),
                "published_date": article.get("webPublicationDate", "Unknown"),
                "processed_content": processing_ready_text(clean_text(article.get("fields", {}).get("body", "No Content"))) 
            })

        print(f"✅ Fetched {len(articles)} articles so far...")

        if len(news_results) < page_size:
            break  # No more pages left

        page += 1

    print(f"🎯 Total {len(articles)} articles fetched.")
    return articles
