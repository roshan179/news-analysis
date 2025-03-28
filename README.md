# 📰 News Aggregation & Analysis Pipeline

🔹 Automated pipeline for fetching, storing, summarizing, analyzing sentiment, and finding similar news articles.

## 📌 Project Overview

This project is a full-fledged data pipeline that:

✅ Fetches latest news articles using an API.

✅ Stores them in a cloud-based PostgreSQL-compatible DB (YugabyteDB).

✅ Summarizes news articles into concise 60-word summaries.

✅ Analyzes sentiment using VADER sentiment analysis.

✅ Finds similar news articles using TF-IDF & Cosine Similarity.

### **Tech Stack**

🐍 Python | NLP | PostgreSQL | YugabyteDB | TF-IDF | VADER

---

## 📌 Features

### **1️⃣ News Fetching & Storage**

- Fetches news articles from an API (e.g., The Guardian API).

- Implements pagination to collect multiple pages of news.

- Stores title, raw content, URL, and publication date in YugabyteDB.

### **2️⃣ News Summarization**

- Cleans text (removes HTML, unwanted words, and hyperlinks).

- Extracts important sentences using TF-IDF vectorization & TextRank.

- Summarizes within 60 words while ensuring complete sentences.

### **3️⃣ Sentiment Analysis**

- Uses VADER Sentiment Analyzer (Lexicon-based NLP model).

- Classifies news into Positive, Negative, or Neutral sentiment.

- Stores sentiment scores in the database.

### **4️⃣ News Similarity Detection**

- Uses TF-IDF Vectorization to convert news into numerical representations.

- Computes Cosine Similarity between articles.

- Identifies top 3 most similar articles for each news item.

### **5️⃣ Error Handling & Logging**

- Modularized functions for better debugging & maintenance.

- Try-Except blocks ensure partial failures don’t crash the pipeline.

- Logs success & error messages at each step.

### 📌 Updated Flag Functionality

The pipeline utilizes flag columns (`summarization_flag`, `sentiment_flag`, `similarity_flag`) in the `news` table to ensure a structured and efficient incremental processing workflow:

- **summarization_flag**: Ensures summarization is performed only once per article.
- **sentiment_flag**: Prevents redundant sentiment analysis on already processed news.
- **similarity_flag**: Avoids unnecessary reprocessing for similarity detection.

These flags help identify new or unprocessed records, ensuring that the pipeline only operates on necessary data, improving efficiency and reducing redundant computations.

### 📌 Reset Sequence Functionality

The `reset_sequence.py` script is used to maintain sequence integrity for primary keys in the database. It ensures that:

1. The sequence values are updated correctly even after manual truncation - TRUNCATE TABLE RESTART IDENTITY.
2. The incremental load continues smoothly without key conflicts.

The script runs the following query format for each table:

```sql
SELECT setval('sequence_name', COALESCE((SELECT MAX(column_name) FROM table_name), 0), TRUE);
```

Sequence Reset Logic: The setval function uses

```sql
COALESCE((SELECT MAX(id) FROM table), 0), TRUE
```

to ensure continuity in incremental loads. This prevents primary key conflicts by setting the sequence to the highest

existing value and correctly handling cases where the table is manually truncated.

---

## 📌 Project Architecture

```txt
            +----------------------+
            |     Fetch News       |
            +----------------------+
                        ||
                        vv
       +------------------------------------+
       |        Store in Database           |
       +------------------------------------+
           /             |               \
          /              |                \
         v               v                 v
+----------------+  +-----------------+  +----------------+
| Summarization  |  | Sentiment       |  | Similarity     |
| (TF-IDF & LDA) |  | Analysis (VADER)|  | Detection      |
+----------------+  +-----------------+  +----------------+
         \                  |                /
          \                 |               /
           v                v              v
          +----------------------------------+
          |     Store in Database            |
          +----------------------------------+
```

## _📌 Installation & Setup_

### _1️⃣ Clone the Repository_

git clone https://github.com/your-username/news-analysis.git

cd news-analysis

### _2️⃣ Set Up Virtual Environment_

python -m venv venv

source venv/bin/activate # On macOS/Linux

venv\Scripts\activate # On Windows

### _3️⃣ Install Dependencies_

pip install -r requirements.txt

### _4️⃣ Configure Database_

Use YugabyteDB (or PostgreSQL).

Create the required tables using:

```txt
CREATE TABLE news (

    news_id SERIAL PRIMARY KEY,

	title TEXT,

	raw_content TEXT,

	url TEXT UNIQUE,

	published_date TIMESTAMP,

    summarization_flag BOOLEAN DEFAULT FALSE,

    sentiment_flag BOOLEAN DEFAULT FALSE,

    similarity_flag BOOLEAN DEFAULT FALSE

);


CREATE TABLE news_summaries (

    id SERIAL PRIMARY KEY,

    news_id INT REFERENCES news(news_id),

	summary TEXT

);


CREATE TABLE news_sentiments (

    id SERIAL PRIMARY KEY,

    news_id INT REFERENCES news(news_id),

    sentiment TEXT,

    sentiment_score FLOAT

);


CREATE TABLE news_similarity (

    news_id INT REFERENCES news(news_id),

    similar_news_id INT REFERENCES news(news_id),

    similarity_score REAL NOT NULL,

    PRIMARY KEY (news_id, similar_news_id)
);
;
```

### _5️⃣ Set Database Connection_

Create a .env file and add:

DB_URL="your_postgresql_connection_string_here"

API_KEY="your_news_api_key_here"

## 📌 Running the Pipeline

python main.py

Prompts you to enter the date range & number of articles.

Automatically fetches, stores, processes & analyzes news data.

Results are stored in the database.

## _📌 Project Structure_

```t
📂 news-analysis
├── 📄 main.py # Main script to run the entire pipeline
├── 📄 web_scraping.py # Fetches the news article from The Guradian - via API calls
├── 📄 data_insertion.py # Stores the fetched news articles
├── 📄 news_summarization.py # Summarizes news articles
├── 📄 news_sentiment.py # Sentiment analysis
├── 📄 news_similarity.py # similarity analysis
├── 📄 reset_sequence.py # Reset the sequence of keys before every run to ensure continuity
├── 📄 requirements.txt # Python dependencies
├── 📄 README.md # Project documentation
├── 📂 models # (Optional) Stores trained models
└── 📂 logs # (Optional) Stores log files
```

## _📌 How It Works (Step-by-Step)_

1️⃣ User inputs date range & number of articles.

2️⃣ News articles are fetched from API & stored in DB.

3️⃣ Summarization module extracts key sentences.

4️⃣ Sentiment analysis determines tone of articles.

5️⃣ Cosine similarity finds related news articles.

6️⃣ Results are stored in the database.

## 📌 Expected Output

### **1️⃣ News Summarization Output (Database)**

```txt
+----------------------------------------------------------------------------------+
| news_id | summary                                                                |
|---------|------------------------------------------------------------------------|
| 1       | "Government announces economic reforms to boost growth..."             |
| 2       | "Tech giant unveils new AI product with enhanced security features..." |
+----------------------------------------------------------------------------------+
```

### **2️⃣ Sentiment Analysis Output**

```txt
+---------------------------------------+
| news_id | sentiment | sentiment_score |
|---------|-----------|-----------------|
| 1       | Positive  | 0.78            |
| 2       | Neutral   | 0.05            |
+---------------------------------------+
```

### **3️⃣ News Similarity Output**

```txt
+----------------------------------------------+
| news_id | similar_news_id | similarity_score |
|---------|----------------|-------------------|
| 1       | 3              | 0.87              |
| 1       | 5              | 0.76              |
+----------------------------------------------+
```

## _📌 Future Enhancements_

✅ Deploy as a Web App using Flask or FastAPI.

✅ Integrate Machine Learning-based Fake News Detection.

✅ Build a Dashboard using Streamlit for real-time visualization.

✅ Add Political Bias Classification to analyze media bias.

## _📌 Contributors_

💡 Created by Roshan Somayajula

## _📌 License_

📜 MIT License – Free to use, modify, and distribute.
