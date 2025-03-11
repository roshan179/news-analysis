import psycopg2
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

#package download for sentiment analysis
nltk.download('vader_lexicon')


DB_URL = "postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/my_database?ssl=true&sslmode=verify-full&sslrootcert=Usfeul info\\root.crt"

try:
    conn = psycopg2.connect(DB_URL)
except Exception as err:
    print(f"Faced an error while connecting to the DB: {err}")

#declare cursor
cur = conn.cursor()

# Fetch all articles
cur.execute("SELECT id, processed_content FROM news")
articles = cur.fetchall()

# Load BERT model & tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Function to classify news
def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    return "Fake" if prediction == 1 else "Real"


sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"
    
if __name__ == "__main__":    
#Calling these functions and inserting results into respective tables in the database
    for article_id, text in articles:
        classification = classify_news(text)
        sentiment = analyze_sentiment(text)
        try:
            cur.execute("INSERT INTO news_sentiment (news_id, sentiment) VALUES (%s, %s)", (article_id, sentiment))
            cur.execute("INSERT INTO news_classification (news_id, classification) VALUES (%s, %s)", (article_id, classification))
            print(f"Successfully analyzed and inserted values into respective tables for article no. {article_id}!")
            conn.commit() 
        except Exception as e:
            print("Faced an error while inserting the values into DB:",e)
    conn.close()
        

