import psycopg2
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import html

# ✅ Download NLTK dependencies
# nltk.download("stopwords")
# nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch values from environment
DB_URL = os.getenv("DB_URL")

def fetch_news_data():
    """Fetch processed news content from the database."""
    conn = psycopg2.connect(DB_URL)
    query = "SELECT news_id,title, raw_content FROM news order by 1;"  # ✅ Using raw_content and cleaning before summarization
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def clean_text(text):
    """Cleans unwanted words, hyperlinks, and HTML while preserving useful content."""

    if not text or pd.isna(text):
        return "Content not available."

    # ✅ Parse HTML properly
    soup = BeautifulSoup(text, "html.parser")

    # ✅ Extract text from <p> tags (ignore <aside>, <blockquote>, <figure>)
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = " ".join(paragraphs)

    # ✅ Remove extra spaces, unwanted words, and links
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove links
    text = re.sub(r"\b(gmt|related|click|read more|subscribe)\b", "", text, flags=re.IGNORECASE)  # Remove noise
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces

    if not text.strip():
        return "Content not available."

    return text


def summarize_text(text, news_id, title, word_limit=60):
    """Summarizes text while ensuring a minimum valid output."""
    sentences = sent_tokenize(text)

    # ✅ Print which news article is being processed
    print(f"\n🔹 Processing News ID: {news_id} | Title: {title}")

    if len(sentences) == 0:
        print("❌ Debug: No sentences detected after tokenization.")
        return "Summary not available."

    if len(word_tokenize(text)) <= word_limit:
        print("✅ Debug: Text is already within 60 words, returning full cleaned text.")
        return text  

    # ✅ Debug: Check number of sentences
    print(f"🔍 Debug: Sentences detected for summarization: {len(sentences)}")

    # ✅ Convert sentences to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    
    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        print("❌ Debug: TF-IDF failed due to very short input.")
        return sentences[0]  

    # ✅ Compute sentence similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)
    
    # ✅ Apply PageRank algorithm
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # ✅ Debug: Check sentence scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print(f"🔍 Debug: Ranked sentences count: {len(ranked_sentences)}")

    # ✅ Select sentences within 60 words
    summary = []
    word_count = 0
    
    for _, sentence in ranked_sentences:
        sent_word_count = len(word_tokenize(sentence))
        if word_count + sent_word_count > word_limit:
            break
        summary.append(sentence.strip())
        word_count += sent_word_count

    if not summary:
        print("❌ Debug: No sentences added to summary.")
        return sentences[0]  

    print("✅ Debug: Final summary generated successfully!\n")
    return " ".join(summary)

def summarize_news():
    """Main function to fetch news, summarize, and store in DB."""
    df = fetch_news_data()
    df["summary"] = df.apply(lambda row: summarize_text(clean_text(row["raw_content"]), row["news_id"], row["title"], word_limit=60), axis=1)

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    for i, row in df.iterrows():
        query = """
        INSERT INTO news_summaries (news_id, summary) VALUES (%s, %s)
        ON CONFLICT (news_id) DO UPDATE SET summary = EXCLUDED.summary;
        """
        cursor.execute(query, (row["news_id"], row["summary"]))

    conn.commit()
    conn.close()
    print("✅ News summaries stored in the database with a strict 60-word limit!")


