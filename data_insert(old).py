import re
import html
import psycopg2
import requests
import nltk
from bs4 import BeautifulSoup
import web_scraping
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from web_scraping import fetch_news
from datetime import datetime, timedelta
from psycopg2.extras import execute_batch

# Download necessary NLTK data
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

# PostgreSQL Connection Details
DB_CONFIG = {
    "dbname": "news_db",
    "user": "postgres",
    "password": "postgrespass123",
    "host": "localhost",
    "port": "5432",
}

# cluster url : postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/yugabyte?ssl=true&sslmode=verify-full&sslrootcert=E:\Data Engineering\Project - Fake News Detection\root.crt

# News API Endpoint (Example: The Guardian API)
# URL = "https://content.guardianapis.com/search?api-key=your_api_key&show-fields=body"

# def clean_text(text):
#     """Cleans raw HTML/text data for NLP processing."""
#     text = html.unescape(text)  # Decode HTML entities
#     text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
#     text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric characters
#     text = text.lower()  # Convert to lowercase

#     # Tokenization & Stopword Removal
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words("english"))
#     words = [word for word in words if word not in stop_words]

#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     words = [lemmatizer.lemmatize(word) for word in words]

#     return " ".join(words)

# def fetch_news():
#     """Fetches news articles and cleans content before storing in PostgreSQL."""
#     response = requests.get(URL)
#     data = response.json()

#     articles = []
#     for article in data["response"]["results"]:
#         raw_content = article["fields"]["body"]
#         cleaned_content = clean_text(raw_content)  # Clean the text

#         articles.append({
#             "title": article["webTitle"],
#             "raw_content": raw_content,
#             "cleaned_content": cleaned_content,
#             "url": article["webUrl"],
#         })
    
#     return articles

def create_table():
    """Creates a PostgreSQL table if it does not exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS news_articles (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        raw_content TEXT NOT NULL,
        cleaned_content TEXT NOT NULL,
        url TEXT UNIQUE NOT NULL
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    conn.close()

def insert_articles(articles):
    """Inserts news articles into PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for article in articles:
        try:
            cur.execute(
                """
                INSERT INTO news_articles (title, raw_content, cleaned_content, url)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING;
                """,
                (article["title"], article["raw_content"], article["cleaned_content"], article["url"]),
            )
        except Exception as e:
            print(f"Error inserting article: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table()  # Ensure table exists
    num_articles = int(input("Enter the number of articles to fetch: "))  # User input
    news_articles = fetch_news(num_articles)  # Fetch news & clean text
    insert_articles(news_articles)  # Store in PostgreSQL
    # print("✅ News articles fetched and stored successfully!")