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
from psycopg2.extras import execute_batch# 🟢 Store News Function (Now Using YugabyteDB)

DB_URL = "postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/my_database?ssl=true&sslmode=verify-full&sslrootcert=./Useful info/root.crt"

def store_articles(articles):
    try:
        conn = psycopg2.connect(DB_URL)
    except Exception as err:
        print(f"Faced an error while connecting to the DB: {err}")
    cursor = conn.cursor()

    # Create Table (If Not Exists)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS news (
        id SERIAL PRIMARY KEY,
        title TEXT,
        raw_content TEXT,
        processed_content TEXT,
        url TEXT UNIQUE,        
        published_date TIMESTAMP
    );
    """
    try:
        cursor.execute(create_table_query)
        print("Table Successfully created!")
    except Exception as e:
        print(f"Faced an error while creating the table:{e}")


    # Batch Insert Data (Ignoring Duplicates)
    insert_query = """
    INSERT INTO news (title, raw_content, processed_content, url, published_date) 
    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (url) DO NOTHING;
    """
    execute_batch(cursor, insert_query, [(a["title"], a["raw_content"], a["processed_content"], a["url"], a["published_date"]) for a in articles])

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ News articles stored in YugabyteDB.")

# 🟢 Run the Script
if __name__ == "__main__":
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for last 7 days: ") or None
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for today: ") or None
    num_articles = int(input("Enter number of articles to fetch: "))

    articles = fetch_news(start_date, end_date, num_articles)
    store_articles(articles)
