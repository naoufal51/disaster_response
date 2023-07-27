import sys
import string
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.utils import parallel_backend
from xgboost import XGBClassifier
import re
import contractions
from bs4 import BeautifulSoup


def tokenize(text):
    """Normalize, tokenize, and lemmatize the messages text
    
    Args:
        text: string, message text to be processed (tokenized)  
        
    Returns:
        tokens: list of strings, tokenized text
    Notes:
    Processing step as referenced in lesson
    """
    # expand contractions
    text = contractions.fix(text)

    # Remove html tags using beautifulsoup (from lesson)
    soup = BeautifulSoup(text, 'html.parser', from_encoding='utf-8')
    text = soup.get_text()

    # Remove urls (from lesson)
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # Normalize text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens
