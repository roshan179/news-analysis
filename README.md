# news-analysis
A project to design and deploy a data pipeline, to fetch news data via APIs, and leverage it for analyses like political bias, text sentiment, visualization of word count etc.
The first step is to hit the API for the news website, The Guardian, for a custom user-specified timeframe and number of articles, and ingest it to a cloud based database - in our case it is YugabyteDB Aeon (free version) - for enchanced scalability in future.
Next step is to pick the main news data with slight text processing like text cleansing and lemmatization, from the required columns, performing analysis - like sentiment analysis, political bias analysis and visualization.
