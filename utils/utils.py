import string
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
import contractions
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import ssl
from textblob import TextBlob
import spacy
from spacy import cli

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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


class TextLengthExtractor(BaseEstimator, TransformerMixin):

    """
    Compute the length of the text in terms of words.
    """

    def __init__(self):
        pass

    def compute_text_length(self, text):
        """
        Compute length of the text.

        Args:
            text (str): text to be processed

        Returns:
            text_len (np.array): array of text length for our text
        """

        text_len = []
        for t in text:
            text_len.append(len(t))
        text_len = np.array(text_len).reshape(-1, 1)
        return text_len
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        return self.compute_text_length(X)


class VerbCountExtractor(BaseEstimator, TransformerMixin):
    """Count the number of verbs for each text entry """

    def __init__(self):
        pass

    def count_verbs(self, text):
        """
        Count the number of verbs in text.

        Args:
            text (str): text to be processed

        Returns:
            verb_count (np.array): array of verb count for our text

        """
        texts_tags = [nltk.pos_tag(nltk.word_tokenize(t)) for t in text]
        verb_count = []
        for t in texts_tags:
            verb_count.append(len([w for w, tag in t if tag.startswith('VB')]))

        verb_count = np.array(verb_count).reshape(-1, 1)
        return verb_count
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return self.count_verbs(X)

    
class NegationCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def count_negations(self, text):
        """
        Count the number of negation in the entry text

        Args:
            text (str): text to be processed

        Returns:
            neg_count (np.array): number of negations in each message in the dataset
        
        """
        neg_count = np.array([len([word for word in mark_negation(t.lower().split()) if "_NEG" in word]) for t in text]).reshape(-1, 1)
        return neg_count

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return self.count_negations(X)


class EmotionWordCountExtractor(BaseEstimator, TransformerMixin):
    """ Count the number of words for each emotion and for each text entry
    
        Args:
            lexicon_filepath (str): path to the emotion lexicon (https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
              
    """
    def __init__(self, lexicon_filepath):
        self.lexicon_filepath = lexicon_filepath
        self.emotion_lexicon = pd.read_csv(self.lexicon_filepath, sep='\t', header=None, names=['word', 'emotion', 'association'])
        self.emotions = self.emotion_lexicon['emotion'].unique()

    def load_words(self, emotion):
        return set(self.emotion_lexicon[(self.emotion_lexicon['emotion'] == emotion) & (self.emotion_lexicon['association'] == 1)]['word'])
        
    def count_emotion_words(self, text, emotion_words):
        """
        Count the number of words for the chosen emotion
        
        Args:
            text (str): text to be processed
            emotion_words (set): set of words associated with the selected emotion
            
        Returns:
            emotion_count (np.array): number of instance of the selected emotion in the text.
        """
        return np.array([len([word for word in t.lower().split() if word in emotion_words]) for t in text]).reshape(-1, 1)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        features = np.hstack([self.count_emotion_words(X, self.load_words(emotion)) for emotion in self.emotions])
        return features


class PunctuationCountExtractor(BaseEstimator, TransformerMixin):
    """ Count the number of punctuation marks for each text entry
    
        Args:
            punctuation_marks (list): list of punctuation marks to be counted
        Notes:
            For our disaster response pipeline we only care about exclamation 
            marks, question marks.
              
    """
    def __init__(self, punctuation_marks):
        self.punctuation_marks = punctuation_marks

    def punctuation_count(self, text):
        punctuation_list = [p for p in text if p in self.punctuation_marks]
        return len(punctuation_list)


    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_punct = pd.Series(X).apply(self.punctuation_count)
        return pd.DataFrame(X_punct)
    
class CapitalizationCountExtractor(BaseEstimator, TransformerMixin):
    """
    Count the number of capital letters for each text entry.
    """
    def __init__(self):
        pass

    def count_capital(self, X):
        count_cap = []
        for t in X:
            count_cap.append(sum([1 for c in t if c.isupper()]))
        count_cap = np.array(count_cap).reshape(-1,1)
        return count_cap
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        return self.count_capital(X)
    

class SubjectivityExtractor(BaseEstimator, TransformerMixin):
    """
    Extract subjectivity score for text using textblob sentiment analysis.
    """
    def __init__(self):
        pass

    def extract_subjectivity(self, X):
        subjectivity = []
        for t in X:
            subjectivity.append(TextBlob(t).sentiment.subjectivity)
        subjectivity = np.array(subjectivity).reshape(-1,1)
        return subjectivity
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        return self.extract_subjectivity(X)
    

class PolarityExtractor(BaseEstimator, TransformerMixin):
    """
    Extract polarity score for text using textblob sentiment analysis.
    """

    def __init__(self):
        pass

    def extract_polarity(self, X):
        polarity = []
        for t in X:
            polarity.append(TextBlob(t).sentiment.polarity)
        polarity = np.array(polarity).reshape(-1,1)
        return polarity
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        return self.extract_polarity(X)


class NamedEntityCounter(BaseEstimator, TransformerMixin):
    """
    Apply Name entity recognition on text and coun the number of occurences for each entity and line.

    Args:
        entities (list): list of entities to be counted (e.g. ['LOC', 'ORG'])
        batch_size (int): batch size for efficient processing using spacy pipeline

    Exception:
        OSError: if spacy model is not found, download it from spacy server.
    """
    def __init__(self, entities=['LOC', 'ORG'], batch_size=1000):
        self.entities = entities
        self.batch_size = batch_size
        self.nlp = self.load_spacy_model('en_core_web_sm')

    def load_spacy_model(self, model_name='en_core_web_sm'):
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"{model_name} not found. Downloading...")
            cli.download(model_name)
            nlp = spacy.load(model_name)
        return nlp

    def count_named_entities(self, X):
        ent_count = []
        for doc in self.nlp.pipe(X, batch_size=self.batch_size):
            ent_count.append(sum([ent.label_ in self.entities for ent in doc.ents ]))
        ent_count = np.array(ent_count).reshape(-1,1)
        return ent_count

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return self.count_named_entities(X)