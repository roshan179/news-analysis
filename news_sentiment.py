import psycopg2
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer

# ✅ Download necessary NLP tools
# nltk.download("vader_lexicon")
# nltk.download("stopwords")
# nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch values from environment
DB_URL = os.getenv("DB_URL")
# API_KEY = os.getenv("API_KEY")

sent_pro_id = [] #list of IDs successfully processed for sentiment analsyis, to be updated at last
# sim_pro_id = [] #list of IDs successfully processed for similarity analsyis, to be updated at last

def fetch_sentiment_news_data():
    """Fetch raw news content from the database for sentiment analysis."""
    conn = psycopg2.connect(DB_URL)
    query = "SELECT news_id, raw_content FROM news where sentiment_flag=FALSE order by news_id;"  
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Processing Function 1: For Sentiment Analysis
def preprocess_for_sentiment(text):
    """Cleans text for sentiment analysis (preserving sentence structure)."""
    soup = BeautifulSoup(text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])  # Extract <p> content
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = text.strip()
    return text if text else "Content not available."



# ✅ Function 1: Sentiment Analysis
def analyze_sentiment(text):
    """Analyzes sentiment and returns Positive, Neutral, or Negative."""
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)["compound"]

    # Classify sentiment based on score
    if sentiment_score > 0.05:
        return "Positive", sentiment_score
    elif sentiment_score < -0.05:
        return "Negative", sentiment_score
    else:
        return "Neutral", sentiment_score
    


def perform_sentiment_analysis():
    """Processes news sentiment and stores it in the database."""
    print("\n🚀 Performing Sentiment Analysis...")
    
    # ✅ Fetch Data
    df = fetch_sentiment_news_data()
    
    # ✅ Process Raw Content for Sentiment
    df["cleaned_sentiment_text"] = df["raw_content"].apply(preprocess_for_sentiment)
    
    # ✅ Apply Sentiment Analysis
    df[["sentiment", "sentiment_score"]] = df["cleaned_sentiment_text"].apply(lambda x: pd.Series(analyze_sentiment(x)))

    # ✅ Store Sentiments in Database
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    for i, row in df.iterrows():
        query = """
        INSERT INTO news_sentiments (news_id, sentiment, sentiment_score) VALUES (%s, %s, %s)
        ON CONFLICT (news_id) DO UPDATE SET sentiment = EXCLUDED.sentiment, sentiment_score = EXCLUDED.sentiment_score;
        """
        try:
            cursor.execute(query, (row["news_id"], row["sentiment"], row["sentiment_score"]))
            sent_pro_id.append(row["news_id"])
            print(f"✅ Successfully inserted sentiment data for news_id: {row['news_id']}!")
        except Exception as e:
            print(f"❌ Error inserting sentiment data for news_id {row['news_id']}: {e}")

    # ✅ Update processed flag for successful records - sentiment analysis
    if sent_pro_id:
        update_query = f"""
        UPDATE news 
        SET sentiment_flag = TRUE 
        WHERE news_id IN ({','.join(map(str, sent_pro_id))});
        """
        cursor.execute(update_query)
        print(f"✅ Updated sentiment_flag for {len(sent_pro_id)} records.")

    conn.commit()
    conn.close()
    print("✅ Sentiment Analysis Completed!")

