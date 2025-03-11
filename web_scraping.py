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
from collections import defaultdict
from datetime import datetime, timedelta

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
PAGE_SIZE = 50  # Maximum allowed per API request

def generate_fetch_plan(start_date=None, end_date=None, total_articles=None):
    """Distribute the total_articles evenly across the date range using defaultdict."""
    if not start_date:
        start_date = datetime.today() - timedelta(days=7)
    if not end_date:
        end_date = datetime.today() - timedelta(days=1)
    
    if not total_articles:
        total_articles=70

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    num_days = (end_date - start_date).days + 1
    articles_per_day = total_articles // num_days
    remainder = total_articles % num_days  

    fetch_plan = defaultdict(int)
    current_date = start_date

    while current_date <= end_date:
        fetch_plan[current_date.strftime("%Y-%m-%d")] = articles_per_day
        current_date += timedelta(days=1)

    # Distribute remaining articles evenly
    for i, date in enumerate(fetch_plan.keys()):
        if i < remainder:
            fetch_plan[date] += 1

    return fetch_plan

def fetch_news(date=None, num_articles=50):
    """Fetches exactly num_articles for a given date with default date handling."""
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")

    articles = []
    page = 1
    page_size = min(50, num_articles)  # Fetch in chunks of 50 or less

    while len(articles) < num_articles:
        params = {
            "api-key": API_KEY,
            "show-fields": "body",
            "page-size": page_size,
            "page": page,
            "from-date": date,
            "to-date": date
        }

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()  # Raise error for HTTP failures
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            break  # Stop fetching if there's an API failure

        if "response" not in data or "results" not in data["response"]:
            print(f"⚠️ No valid response received for {date}, page {page}.")
            break  # Stop if the structure is unexpected

        news_results = data["response"]["results"]  # Corrected response path
        if not news_results:
            print(f"⚠️ No more articles found for {date} on page {page}.")
            break  # Stop if no articles were returned

        # articles.extend(news_results[:num_articles - len(articles) + 1])  # Fetch only required count
        for article in news_results:
            if len(articles) > num_articles:
                break

            articles.append({
                "title": article.get("webTitle", "No Title"),
                "raw_content": article.get("fields", {}).get("body", "No Content"),
                "url": article.get("webUrl", "No URL"),
                "published_date": article.get("webPublicationDate", "Unknown")
                # "processed_content": processing_ready_text(clean_text(article.get("fields", {}).get("body", "No Content")))
            })

        print(f"✅ {len(articles)} articles fetched for {date}, page {page}.")
        
        if len(news_results) < page_size:
            print(f"✅ No more articles available for {date}. Ending fetch.")
            break  # Stop if fewer articles were returned than requested (pagination exhausted)

        page += 1

    return articles

# 🟢 Fetch News Function - old function 
# def fetch_news(start_date=None, end_date=None, num_articles=50):
#     articles = []
#     page = 1
#     page_size = 50

#     if not start_date:
#         start_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
#     if not end_date:
#         end_date = datetime.today().strftime("%Y-%m-%d")

#     while len(articles) < num_articles:
#         params = {
#             "api-key": API_KEY,
#             "show-fields": "body",
#             "page-size": page_size,
#             "page": page,
#             "from-date": start_date,
#             "to-date": end_date
#         }

#         response = requests.get(BASE_URL, params=params)
#         data = response.json()

#         if "response" not in data or "results" not in data["response"]:
#             print("❌ Error fetching news.")
#             break

#         news_results = data["response"]["results"]

#         for article in news_results:
#             if len(articles) >= num_articles:
#                 break

            # articles.append({
            #     "title": article.get("webTitle", "No Title"),
            #     "raw_content": article.get("fields", {}).get("body", "No Content"),
            #     "url": article.get("webUrl", "No URL"),
            #     "published_date": article.get("webPublicationDate", "Unknown"),
            #     "processed_content": processing_ready_text(clean_text(article.get("fields", {}).get("body", "No Content"))) 
            # })

#         print(f"✅ Fetched {len(articles)} articles so far...")

#         if len(news_results) < page_size:
#             break  # No more pages left

#         page += 1

#     print(f"🎯 Total {len(articles)} articles fetched.")
#     return articles
