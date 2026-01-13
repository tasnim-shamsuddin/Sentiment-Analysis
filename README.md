# Sentiment Analysis

## Project Overview
This project performs sentiment analysis on text data (reviews) using Natural Language Processing (NLP) techniques. The goal is to classify reviews as **positive (1)** or **negative (0)** based on their rating.



## Dataset
- Reviews dataset with ratings from 1 to 5
- Ratings < 3 are labeled as 0 (negative), ratings ≥ 3 are labeled as 1 (positive)
- Dataset is mildly imbalanced (approx. 2:1)

## Data Preprocessing
- Lowercasing all text
- Removal of special characters, punctuation, emojis, URLs, HTML tags, and extra spaces
- Stopwords removal
- Lemmatization using NLTK WordNet

## Feature Engineering
Two types of vectorizations were used to convert text into numerical features:
1. **Bag of Words (CountVectorizer)** – word counts in each document
2. **TF-IDF Vectorizer** – weighted word importance based on frequency across all documents

## Models Used
- **Naive Bayes** (Gaussian/Bernoulli)

## Model Performance
| Vectorizer | Accuracy |
|------------|---------|
| Bag of Words | 0.6164 |
| TF-IDF      | 0.6169 |

**Interpretation:**  
Both models achieved similar accuracy. TF-IDF slightly improved accuracy because it gives more weight to rare but informative words, while common words are down-weighted. Bag of Words is simpler and almost as effective for this dataset.

## Tools & Libraries
- Python
- pandas, numpy
- NLTK (stopwords, WordNet)
- scikit-learn (CountVectorizer, TfidfVectorizer, Naive Bayes)
- BeautifulSoup (for HTML cleaning)

## Platform
- Google Colab
