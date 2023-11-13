# Import necessary packages

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt') # Punkt Tokenizer Model
nltk.download('stopwords') # Stopword Token List
nltk.download('wordnet') # WordNet

def load_data(filename):

# Load the dataset from a CSV file into a Pandas DataFrame.

    df = pd.read_csv(filename)
    return df

def clean_text(text):

# Clean the text by removing special characters, numbers, and other non-essential elements.

    # Remove HTML tags using regex
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def tokenize_text(text):

# Tokenize the text into individual words/tokens.

    tokens = word_tokenize(text)
# Filter out stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

def lemmatize_tokens(tokens):

# Lemmatize the tokens - converting them to their base or root form.

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def preprocess_text(text):

# Fully preprocess text, including cleaning, tokenization, and lemmatization.

    text = clean_text(text)
    tokens = tokenize_text(text)
    lemmas = lemmatize_tokens(tokens)
# Combine tokens back into a single string
    preprocessed_text = ' '.join(lemmas)
    return preprocessed_text

def vectorize_text(texts):

# Convert a collection of text documents into a matrix of TF-IDF features.
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

if __name__ == "__main__":
    filename = '/home/orion/Geo/Projects/Emotion-Detection-Project/imdb_dataset.csv'
    df = load_data(filename)
    df['processed_text'] = df['text'].apply(preprocess_text)
    tfidf_matrix, vectorizer = vectorize_text(df['processed_text'])
    # You can now use 'tfidf_matrix' as input for your machine learning models.
    # And 'vectorizer' can be used later to transform any new text based on the learned vocabulary.
