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

sim_pro_id = [] #list of IDs successfully processed for similarity analsyis, to be updated at last



def fetch_similarity_news_data():
    """Fetch raw news content from the database for similarity analysis."""
    conn = psycopg2.connect(DB_URL)
    query = "SELECT news_id, raw_content FROM news where similarity_flag=FALSE order by news_id;"  
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ✅ Processing Function: For News Similarity
def preprocess_for_similarity(text):
    """Cleans and normalizes text for similarity detection (focus on meaning)."""
    soup = BeautifulSoup(text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])  # Extract meaningful text
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in word_tokenize(text) if word not in stopwords.words("english")])  # Remove stopwords
    text = text.strip()
    return text if text else "Content not available."


# ✅ Function: News Similarity Detection
def compute_news_similarity(df):
    """Computes cosine similarity and returns top 3 similar articles for each news."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_similarity_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    news_similarity_results = []

    for idx, row in df.iterrows():
        similar_indices = similarity_matrix[idx].argsort()[-4:-1][::-1]  # Get top 3 similar articles
        similar_news_ids = [(df.iloc[i]["news_id"], similarity_matrix[idx, i]) for i in similar_indices]
        
        for similar_id, similarity_score in similar_news_ids:
            news_similarity_results.append((row["news_id"], similar_id, similarity_score))

    return news_similarity_results


def perform_news_similarity():
    """Computes and stores news similarity."""
    print("\n🚀 Computing News Similarity...")

    # ✅ Fetch Data
    df = fetch_similarity_news_data()

    # ✅ Process Raw Content for Similarity
    df["cleaned_similarity_text"] = df["raw_content"].apply(preprocess_for_similarity)

    # ✅ Compute News Similarity
    news_similarity_results = compute_news_similarity(df)

    # ✅ Store Similarity Data in Database
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    for news_id, similar_id, score in news_similarity_results:
        query = """
        INSERT INTO news_similarity (news_id, similar_news_id, similarity_score) VALUES (%s, %s, %s)
        ON CONFLICT (news_id, similar_news_id) DO UPDATE SET similarity_score = EXCLUDED.similarity_score;
        """
        try:
            cursor.execute(query, (int(news_id), int(similar_id), float(score)))
            sim_pro_id.append(news_id)
            print(f"✅ Successfully inserted similarity data for news_id: {news_id}!")
        except Exception as e:
            print(f"❌ Error inserting similarity data for news_id {news_id}: {e}")

    # ✅ Update processed flag for successful records - similarity analysis
    if sim_pro_id:
        update_query = f"""
        UPDATE news 
        SET similarity_flag = TRUE 
        WHERE news_id IN ({','.join(map(str, sim_pro_id))});
        """
        cursor.execute(update_query)
        print(f"✅ Updated similarity flag for {len(sim_pro_id)} records.")

    conn.commit()
    conn.close()
    print("✅ News Similarity Computation Completed!")