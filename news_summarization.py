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

summarize_pro_id=[] #list of news_ids successfully summarized

def fetch_news_data():
    """Fetch processed news content from the database."""
    conn = psycopg2.connect(DB_URL)
    query = "SELECT news_id,title, raw_content FROM news where summarization_flag=FALSE order by 1;"  # ✅ Using raw_content and cleaning before summarization
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def clean_text(text):
    """Cleans unwanted words, hyperlinks, time stamps, and HTML while preserving useful content."""

    if not text or pd.isna(text):
        return "Content not available."

    # ✅ Parse HTML properly
    soup = BeautifulSoup(text, "html.parser")

    # ✅ Extract text from <p> tags (ignore <aside>, <blockquote>, <figure>)
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = " ".join(paragraphs)

    # ✅ Remove hyperlinks
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # ✅ Remove common noise words
    text = re.sub(r"\b(gmt|related|click|read more|subscribe)\b", "", text, flags=re.IGNORECASE)

    # ✅ Remove **standalone time expressions** like "4.58pm", "10:30 AM", "22:15 GMT"
    text = re.sub(r"\b\d{1,2}[:.]\d{2}\s?(?:am|pm|AM|PM|GMT)?\b", "", text)

    # ✅ Remove **punctuation-only lines**
    text = re.sub(r"^[\W_]+$", "", text, flags=re.MULTILINE)

    # ✅ Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text.strip():
        return "Content not available."

    return text





def summarize_text(text, news_id, title, word_limit=60):
    """Summarizes text while ensuring a strict 60-word limit and handling edge cases."""
    sentences = sent_tokenize(text)

    print(f"\n🔹 Processing News ID: {news_id} | Title: {title}")

    if len(sentences) == 0:
        print(f"⚠️ Debug: No sentences found, returning 'Summary not available.'")
        return "Summary not available.", 0

    print(f"🔍 Debug: {len(sentences)} sentences detected.")

    # ✅ If text is already within word limit, return as-is
    if len(word_tokenize(text)) <= word_limit:
        print("✅ Debug: Text is within limit, returning full text.")
        return text, len(word_tokenize(text))

    # ✅ Convert sentences to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        print(f"⚠️ Debug: TF-IDF failed, returning first sentence.")
        return (sentences[0], len(word_tokenize(sentences[0]))) if sentences else ("Summary not available.", 0)

    # ✅ Compute sentence similarity and rank using PageRank
    similarity_matrix = cosine_similarity(sentence_vectors)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    print(f"🔍 Debug: {len(ranked_sentences)} sentences ranked.")

    # ✅ Try picking sentences that fit within the limit
    summary = []
    word_count = 0

    for _, sentence in ranked_sentences:
        sent_word_count = len(word_tokenize(sentence))

        if word_count + sent_word_count > word_limit:
            print(f"⚠️ Debug: Stopping at {word_count} words (Next = {sent_word_count} words).")
            break

        summary.append(sentence.strip())
        word_count += sent_word_count

    # ✅ If all sentences are too long, truncate the first one
    if not summary and ranked_sentences:
        print(f"⚠️ Debug: All sentences too long. Truncating first one.")
        truncated_summary = " ".join(word_tokenize(ranked_sentences[0][1])[:word_limit]) + "..."
        return truncated_summary, word_limit

    final_summary = " ".join(summary)
    final_summary_word_count = len(word_tokenize(final_summary))

    print(f"✅ Debug: Final summary contains {final_summary_word_count} words.\n")

    return final_summary, final_summary_word_count


def summarize_news():
    """Main function to fetch news, summarize, and store in DB."""
    df = fetch_news_data()

    # ✅ Apply summarization and get both summary + word count
    df[["summary", "summary_word_count"]] = df.apply(
        lambda row: pd.Series(summarize_text(clean_text(row["raw_content"]), row["news_id"], row["title"], word_limit=60)), axis=1
    )

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    for i, row in df.iterrows():
        query = """
        INSERT INTO news_summaries (news_id, summary, summary_word_count) 
        VALUES (%s, %s, %s)
        ON CONFLICT (news_id) DO UPDATE 
        SET summary = EXCLUDED.summary, summary_word_count = EXCLUDED.summary_word_count;
        """
        try:
            cursor.execute(query, (row["news_id"], row["summary"], row["summary_word_count"]))
            summarize_pro_id.append(row["news_id"])
            print(f"✅ Successfully inserted summarization data for news_id: {row['news_id']}!")
        except Exception as e:
            print(f"❌ Error inserting summarization data for news_id {row['news_id']}: {e}")

        # ✅ Update processed flag for successful records - sentiment analysis
    if summarize_pro_id:
        update_query = f"""
        UPDATE news 
        SET summarization_flag = TRUE 
        WHERE news_id IN ({','.join(map(str, summarize_pro_id))});
        """
        cursor.execute(update_query)
        print(f"✅ Updated summarization_flag for {len(summarize_pro_id)} records.")

    conn.commit()
    conn.close()
    print("✅ News summaries stored in the database with a strict 60-word limit!")


