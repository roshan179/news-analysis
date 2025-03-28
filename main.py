import time
from data_insertion import fetch_news, store_articles, generate_fetch_plan
from news_summarization import summarize_news
from news_sentiment_similarity import perform_sentiment_analysis, perform_news_similarity
from reset_sequence import reset_sequences

#Resetting Sequences and readying database for incremental load
reset_sequences()


def log_message(step):
    print("\n" + "=" * 50)
    print(f"🚀 {step}")
    print("=" * 50)

# ✅ Track overall success status
pipeline_success = True

try:
    # ✅ Step 1: Take User Input for Date & Number of Articles
    log_message("STEP 1: Fetching & Storing News")
    print("---------------Kindly enter below the date range and total number of articles required;")
    print("Code will fetch [NO. OF ARTICLES]/[NO.OF DAYS IN DATERANGE] Articles for each day------------")

    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 7 days ago: ") or None
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for yesterday: ") or None
    num_articles_input = input("Enter number of articles to fetch or press Enter to accept 70 articles: ")
    num_articles = int(num_articles_input) if num_articles_input.isdigit() else 70

    fetch_plan = generate_fetch_plan(start_date, end_date, num_articles)

    all_articles = []
    for date, num_articles in fetch_plan.items():
        all_articles.extend(fetch_news(date, num_articles))

    print(f"🎯 Total articles fetched: {len(all_articles)}")

    store_articles(all_articles)
    log_message("✅ News articles successfully stored in the database.")

except Exception as e:
    print(f"❌ ERROR in STEP 1 (Fetching & Storing News): {e}")
    pipeline_success = False

try:
    # ✅ Step 2: Summarize News
    log_message("STEP 2: Summarizing News")
    summarize_news()

except Exception as e:
    print(f"❌ ERROR in STEP 2 (News Summarization): {e}")
    pipeline_success = False

try:
    # ✅ Step 3: Perform Sentiment Analysis
    log_message("STEP 3: Performing Sentiment Analysis")
    perform_sentiment_analysis()

except Exception as e:
    print(f"❌ ERROR in STEP 3 (Sentiment Analysis): {e}")
    pipeline_success = False

try:
    # ✅ Step 4: Compute News Similarity
    log_message("STEP 4: Computing News Similarity")
    perform_news_similarity()

except Exception as e:
    print(f"❌ ERROR in STEP 4 (News Similarity Computation): {e}")
    pipeline_success = False

# ✅ Final Completion Message
if pipeline_success:
    log_message("🎯 PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
else:
    log_message("❌ PROJECT EXECUTION FAILED! Check the errors above.")
