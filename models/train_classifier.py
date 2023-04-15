import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
from sklearn.utils import parallel_backend


def load_data(database_filepath):
    """
    Load processed distater messages data from sqlite database
    
    Args:
        database_filepath: path to the sqlite distater messages database
    
    Returns:
        X: pandas dataframe containing the messages (Features)
        y: pandas dataframe containing the categories (Target)
        category_names: list of category names
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('disaster_messages', engine)
    
    # define features and target variables X and Y
    X = df['message']
    Y = df.iloc[:, 4:]
    # replace 2 with 1 in related column
    Y['related'] = Y['related'].replace(2,1)
    
    # get category names
    category_names = Y.columns
    
    return X, Y, category_names
    


def tokenize(text):
    """Normalize, lemmatize, and tokenize the messages text
    
    Args:
        text: string, message text to be processed (tokenized)
        
    Returns:
        tokens: list of strings, tokenized text
    """
    # Normalize text
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation) # This maps punctuation to None
    text = text.translate(translator) # This removes punctuation
    
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # tokenize text
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    
    #lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return tokens
    
    
    


def build_model():
    """
    Build a model pipeline using XGBoost Classifier and GridSearchCV
    
    Args:
        None
        
    Returns:
        cv: GridSearchCV object
    """
    
    # create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('Classifier', MultiOutputClassifier(XGBClassifier()))
    ])
    
    # define the parameter for grid search
    parameters = {    
                'Classifier__estimator__n_estimators': [50, 100, 200]
                }
    
    # create grid search object
    with parallel_backend('threading', n_jobs=-1):
        cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='f1_macro', verbose=2)
    return cv

    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model using test data.
    
    Args:
        model: model to be evaluated
        X_test: test features
        Y_test: test target
        category_names: list of category names
    
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    
    # print the classification report for each category (f1 score, precision, recall)
    number_of_categories = Y_test.shape[1]
    for i in range(number_of_categories):
        print("\nCategory: ", category_names[i])
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Store the classifier into a pickle file.
    
    Args:
        model: model to be saved
        model_filepath: path to where the model will be saved
        
    Returns:
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()