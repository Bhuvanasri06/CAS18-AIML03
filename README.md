# CAS18-AIML03: AI-Powered Social Media Sentiment Analysis for Brands

## Data Dreamers - Mithun - Sentisphere

### Problem Statement:
AI-Powered Social Media Sentiment Analysis for Brands (AIML03)

This project aims to analyze Twitter (X) data to assess sentiments surrounding various brands. By categorizing tweets as positive, negative, or neutral, this tool can help brands understand public opinion, track their reputation, and identify areas for improvement.

### Tech Stack:
- **Tweepy**: Used to connect to the Twitter API (now X API) and collect tweets.
- **spaCy**: Advanced NLP tools for text processing, including entity recognition.
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A sentiment analysis tool for categorizing tweets into positive, negative, or neutral sentiments.
- **BERT** **(Bidirectional encoder representation from transformers)**: it is used for fine-tuning and mainly used for search engine and text classification.
- **Plotly**: Interactive visualizations for sentiment trends and analysis.
- **WordCloud**: For generating visualizations of frequent words from positive/negative tweets.
- **Pandas & NumPy**: For data manipulation and analysis.
  
---
## Project Structure

### 1. Data Collection
- **Tweepy API**: Connect to the Twitter API and gather tweets mentioning target brands using specific search queries.
- **Pagination**: Ensure that the system is capable of pulling a large volume of tweets by implementing pagination.
- **Metadata Storage**: Tweets will be stored along with essential metadata such as timestamps, user metrics, and geolocation (if available).

### 2. Preprocessing Pipeline
- **Cleaning Tweets**: Remove unnecessary content such as URLs, special characters, and other irrelevant data.
- **Tokenization**: Split the tweet text into smaller, meaningful parts (tokens).
- **Stopwords Removal**: Filter out common words (such as "the", "and", "is") that don't contribute to sentiment analysis.
- **Emoji Handling**: Emojis are crucial in sentiment analysis and will be considered for emotional context.
- **Text Normalization**: Lowercase conversion and lemmatization to standardize the text for better analysis.

### 3. Sentiment Analysis
- **VADER Sentiment Scoring**: Use the VADER sentiment analyzer to categorize tweets as positive, negative, or neutral based on predefined thresholds.
- **spaCy for Advanced NLP**: Utilize spaCy for tasks like entity recognition to identify and tag mentions of brands within the tweets.
- **Classification**: Implement a sentiment classification system that can effectively separate tweets into three categories (positive, negative, neutral).

### 4. Visualization & Reporting
- **Time-series Sentiment Trends**: Visualize sentiment changes over time to observe trends.
- **Brand Sentiment Comparison**: Compare sentiment data across different brands to determine their relative public perception.
- **Word Clouds**: Generate word clouds to visualize the most common positive and negative terms mentioned in tweets.
- **Interactive Dashboards**: Create dashboards using Plotly for interactive and user-friendly sentiment insights.

### 5. Potential Enhancements
- **Advanced Machine Learning Models**: Integrate machine learning models (e.g., BERT, GPT) to improve sentiment prediction accuracy.
- **Topic Modeling**: Implement topic modeling (e.g., Latent Dirichlet Allocation) to understand the key themes driving sentiment.
- **Automated Alerting**: Set up notifications to alert users to significant changes in sentiment, indicating potential PR crises or opportunities.

---

## Getting Started

### Prerequisites:
- Python 3.x
- Required libraries: `tweepy`, `pandas`, `numpy`, `spacy`, `vaderSentiment`, `plotly`, `wordcloud`, etc.
- Twitter API access credentials (you will need to create an app on the Twitter Developer portal to get your API keys).

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/AI-Sentiment-Analysis.git
