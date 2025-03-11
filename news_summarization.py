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

# ✅ Database Connection
DB_URL = "postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/my_database?ssl=true&sslmode=verify-full&sslrootcert=Usfeul info\\root.crt"

def fetch_news_data():
    """Fetch processed news content from the database."""
    conn = psycopg2.connect(DB_URL)
    query = "SELECT news_id, raw_content FROM news order by 1;"  # ✅ Using raw_content and cleaning before summarization
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


def summarize_text(text, word_limit=60):
    """Summarizes text while ensuring a minimum valid output."""
    print(f"🔍 Debug: Cleaned text for summarization: {text[:200]}")
    sentences = sent_tokenize(text)

    if len(word_tokenize(text)) <= word_limit:
        return text  # ✅ If already short, return full cleaned text

    if not sentences:  
        return "Summary not available."  # ✅ Prevents blank summaries

    # ✅ Rank sentences based on importance
    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(sentence_vectors)
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # ✅ Select sentences within 60 words
    summary = []
    word_count = 0
    
    for _, sentence in ranked_sentences:
        sent_word_count = len(word_tokenize(sentence))
        if word_count + sent_word_count > word_limit:
            break
        summary.append(sentence.strip())
        word_count += sent_word_count

    return " ".join(summary)


# ✅ Fetch Data & Apply Summarization
df = fetch_news_data()
df["summary"] = df["raw_content"].apply(lambda x: summarize_text(clean_text(x), word_limit=60))

# ✅ Store Summaries in Database
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

for i, row in df.iterrows():
    query = """
    INSERT INTO news_summaries (news_id, summary) VALUES (%s, %s)
    ON CONFLICT (news_id) DO UPDATE SET summary = EXCLUDED.summary;
    """
    cursor.execute(query, (row["news_id"], row["summary"]))
    print(f"Successfully inserted news summary for news_id : {i+1}")

conn.commit()
conn.close()
print("✅ News summaries stored in the database with a strict 60-word limit!")
