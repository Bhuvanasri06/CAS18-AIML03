!pip install vaderSentiment
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy model for NLP preprocessing
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """Preprocess text using spaCy."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def get_vader_sentiment(text):
    """Get sentiment scores using VADER."""
    return vader.polarity_scores(text)

def get_bert_sentiment(text):
    """Get sentiment prediction using BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    scores = output.logits.numpy()[0]
    probabilities = softmax(scores)
    sentiment = ["very negative", "negative", "neutral", "positive", "very positive"]
    return sentiment[probabilities.argmax()], probabilities

# Sample social media texts
data = [
    "I love this new phone! The camera quality is amazing!",
    "Terrible customer service experience. I'm never buying from this brand again.",
    "The product is okay, but it could be better.",
    "Absolutely fantastic! Highly recommended!",
    "Worst purchase ever, completely disappointed."
]

# Analyze sentiments
vader_scores = [get_vader_sentiment(preprocess_text(text)) for text in data]
bert_results = [get_bert_sentiment(preprocess_text(text)) for text in data]

# Visualization
labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
bert_counts = [sum(1 for res in bert_results if res[0] == label.lower()) for label in labels]

plt.figure(figsize=(8, 5))
plt.bar(labels, bert_counts, color=["red", "orange", "gray", "lightgreen", "green"])
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.title("BERT Sentiment Analysis on Social Media Texts")
plt.show()

# Print results
for i, text in enumerate(data):
    print(f"Text: {text}")
    print(f"VADER Sentiment: {vader_scores[i]}")
    print(f"BERT Sentiment: {bert_results[i][0]}, Probabilities: {bert_results[i][1]}")
    print("-" * 50)
