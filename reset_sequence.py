import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch values from environment
DB_URL = os.getenv("DB_URL")

def reset_sequences():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    queries = [
        "SELECT setval('news_id_seq', COALESCE((SELECT MAX(news_id) FROM news), 1), true);",
        "SELECT setval('news_summaries_id_seq', COALESCE((SELECT MAX(id) FROM news_summaries), 1), true);",
        "SELECT setval('news_sentiments_id_seq', COALESCE((SELECT MAX(id) FROM news_sentiments), 1), true);",
        "SELECT setval('news_similarity_id_seq', COALESCE((SELECT MAX(id) FROM news_similarity), 1), true);"
    ]

    try:
        for query in queries:
            cursor.execute(query)

        conn.commit()
        conn.close()
        print("✅ Database ready for New Load: Sequences Reset successfully!")
    except Exception as e:
        print(f"Faced an error while resetting the index - {e}")

#First Step - resetting indexes to ensure continuity