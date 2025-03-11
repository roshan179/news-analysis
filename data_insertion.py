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
from web_scraping import *
from datetime import datetime, timedelta
from psycopg2.extras import execute_batch# 🟢 Store News Function (Now Using YugabyteDB)

DB_URL = "postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/my_database?ssl=true&sslmode=verify-full&sslrootcert=Usfeul info\\root.crt"

def store_articles(articles):
    try:
        conn = psycopg2.connect(DB_URL)
    except Exception as err:
        print(f"Faced an error while connecting to the DB: {err}")
    cursor = conn.cursor()

    # Create Table (If Not Exists)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS news (
        news_id SERIAL PRIMARY KEY,
        title TEXT,
        raw_content TEXT,
        url TEXT UNIQUE,        
        published_date TIMESTAMP
    );
    """
    try:
        cursor.execute(create_table_query)
        print("Table Successfully created or it already exists!")
    except Exception as e:
        print(f"Faced an error while creating the table:{e}")


    # Batch Insert Data (Ignoring Duplicates)
    if articles:
        insert_query = """
        INSERT INTO news (title, raw_content, url, published_date) 
        VALUES (%s, %s, %s, %s) ON CONFLICT (url) DO NOTHING;
        """
        execute_batch(cursor, insert_query, [(a["title"], a["raw_content"], a["url"], a["published_date"]) for a in articles])

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ News articles stored in YugabyteDB.")
    else:
        print("No Articles present to be inserted into the database.")

# 🟢 Run the Script
if __name__ == "__main__":
    print("---------------Kindly enter below the date range and total number of articles required; code will fetch [NO. OF ARTICLES]/[NO.OF DAYS IN DATERANGE] Articles for each day------------")
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for last 7 days: ") or None
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for today: ") or None
    num_articles_input = input("Enter number of articles to fetch or press Enter to accept 70 articles: ")
    num_articles = int(num_articles_input) if num_articles_input.isdigit() else 70

        # Generate the fetch plan
    fetch_plan = generate_fetch_plan(start_date, end_date, num_articles)

    # Fetch articles based on the optimized plan
    all_articles = []
    for date, num_articles in fetch_plan.items():
        all_articles.extend(fetch_news(date, num_articles))

    print(f"🎯 Total articles fetched: {len(all_articles)}")

    # articles = fetch_news(start_date, end_date, num_articles)
    store_articles(all_articles)
