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

# ✅ Database Connection
DB_URL = "postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/my_database?ssl=true&sslmode=verify-full&sslrootcert=Usfeul info\\root.crt"

def fetch_news_data():
    """Fetch raw news content from the database."""
    conn = psycopg2.connect(DB_URL)
    query = "SELECT news_id, raw_content FROM news order by news_id;"  
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

# ✅ Processing Function 2: For News Similarity
def preprocess_for_similarity(text):
    """Cleans and normalizes text for similarity detection (focus on meaning)."""
    soup = BeautifulSoup(text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])  # Extract meaningful text
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in word_tokenize(text) if word not in stopwords.words("english")])  # Remove stopwords
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

# ✅ Function 2: News Similarity Detection
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

# # ✅ Fetch Data
# df = fetch_news_data()

# # ✅ Process Raw Content for Sentiment & Similarity
# df["cleaned_sentiment_text"] = df["raw_content"].apply(preprocess_for_sentiment)
# df["cleaned_similarity_text"] = df["raw_content"].apply(preprocess_for_similarity)

# # ✅ Apply Sentiment Analysis
# df[["sentiment", "sentiment_score"]] = df["cleaned_sentiment_text"].apply(lambda x: pd.Series(analyze_sentiment(x)))

# # ✅ Compute News Similarity
# news_similarity_results = compute_news_similarity(df)

# # ✅ Store Sentiments in Database
# conn = psycopg2.connect(DB_URL)
# cursor = conn.cursor()

# for i, row in df.iterrows():
#     query = """
#     INSERT INTO news_sentiments (news_id, sentiment, sentiment_score) VALUES (%s, %s, %s)
#     ON CONFLICT (news_id) DO UPDATE SET sentiment = EXCLUDED.sentiment, sentiment_score = EXCLUDED.sentiment_score;
#     """
#     try:
#         cursor.execute(query, (row["news_id"], row["sentiment"], row["sentiment_score"]))
#         print(f"Successfully inserted sentiment data for news_id:{row["news_id"]}!")
#     except Exception as e:
#         print(f"Faced an error while inserting sentiment data for news_id - {row["news_id"]} : {e}")

# # ✅ Store Similarity Data in Database
# for news_id, similar_id, score in news_similarity_results:
#     query = """
#     INSERT INTO news_similarity (news_id, similar_news_id, similarity_score) VALUES (%s, %s, %s)
#     ON CONFLICT (news_id, similar_news_id) DO UPDATE SET similarity_score = EXCLUDED.similarity_score;
#     """
#     try:
#         cursor.execute(query, (int(news_id), int(similar_id), float(score)))  # ✅ Convert to Python int & float
#         print(f"Successfully inserted similarity data for news_id:{news_id}!")
#     except Exception as e:
#         print(f"Faced an error while inserting similarity data for news_id - {news_id} : {e}")

# conn.commit()
# conn.close()
# print("✅ Sentiment analysis and news similarity stored in the database!")

def perform_sentiment_analysis():
    """Processes news sentiment and stores it in the database."""
    df = fetch_news_data()
    df["cleaned_sentiment_text"] = df["raw_content"].apply(preprocess_for_sentiment)  # ✅ Use sentiment-specific preprocessing
    df["sentiment"], df["sentiment_score"] = zip(*df["cleaned_sentiment_text"].apply(analyze_sentiment))

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    for i, row in df.iterrows():
        query = """
        INSERT INTO news_sentiments (news_id, sentiment, sentiment_score) VALUES (%s, %s, %s)
        ON CONFLICT (news_id) DO UPDATE SET sentiment = EXCLUDED.sentiment, sentiment_score = EXCLUDED.sentiment_score;
        """
        cursor.execute(query, (row["news_id"], row["sentiment"], row["sentiment_score"]))

    conn.commit()
    conn.close()
    print("✅ Sentiment Analysis Completed!")

def perform_news_similarity():
    """Computes and stores news similarity."""
    df = fetch_news_data()
    df["cleaned_similarity_text"] = df["raw_content"].apply(preprocess_for_similarity)  # ✅ Use similarity-specific preprocessing

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_similarity_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    for idx, row in df.iterrows():
        similar_indices = similarity_matrix[idx].argsort()[-4:-1][::-1]
        for i in similar_indices:
            query = """
            INSERT INTO news_similarity (news_id, similar_news_id, similarity_score) VALUES (%s, %s, %s)
            ON CONFLICT (news_id, similar_news_id) DO UPDATE SET similarity_score = EXCLUDED.similarity_score;
            """
            cursor.execute(query, (row["news_id"], df.iloc[i]["news_id"], similarity_matrix[idx, i]))

    conn.commit()
    conn.close()
    print("✅ News Similarity Computation Completed!")

