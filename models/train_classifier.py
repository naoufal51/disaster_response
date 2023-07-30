import sys
import os
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.utils import parallel_backend

from xgboost import XGBClassifier
from utils.utils import (tokenize, VerbCountExtractor, NegationCountExtractor, EmotionWordCountExtractor, 
                         PunctuationCountExtractor, TextLengthExtractor, CapitalizationCountExtractor,
                         SubjectivityExtractor, PolarityExtractor, NamedEntityCounter)
from tabulate import tabulate



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
    disaster_messages = pd.read_sql('disaster_messages', engine)

    # define features and target variables X and Y
    X = disaster_messages['message']
    Y = disaster_messages.iloc[:, 4:]
    Y = Y.drop('child_alone', axis=1)

    # replace 2 with 1 in related column
    Y['related'] = Y['related'].replace(2,1)

    # get category names
    category_names = Y.columns

    return X, Y, category_names



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
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('negation_counter', NegationCountExtractor()),
            ('verb_counter', VerbCountExtractor()),
            ('emotion_counter', EmotionWordCountExtractor('data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')),
            ('punctuation_counter', PunctuationCountExtractor(['!', '?'])),
            ('text_length', TextLengthExtractor()),
            ('capitalization_counter', CapitalizationCountExtractor()),
            ('subjectivity', SubjectivityExtractor()),
            ('polarity', PolarityExtractor()),
            ('ner', NamedEntityCounter())
            
        ])),
        ('classifier', MultiOutputClassifier(XGBClassifier(random_state=42)))
            
    ])   

    parameters = {    
        'features__text_pipeline__vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__estimator__n_estimators': [50, 100, 200],
        'classifier__estimator__learning_rate': [0.1, 0.3],
    }

    # define a metric
    f1 = make_scorer(f1_score, average='micro')
    
    # create grid search object
    with parallel_backend('threading', n_jobs=-1):
        cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring=f1, verbose=2)
        
    return cv



def evaluate_model(model, X_test, Y_test, category_names, save_path):
    """
    Evaluate the model using test data.
    We evaluate using f1 score, precision and recall.
    
    Args:
        model: model to be evaluated
        X_test: test features
        Y_test: test target
        category_names: list of category names
        save_path: path to md file for report saving
    
    Returns:
        None
    """
    y_pred = model.predict(X_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as file:
        number_of_categories = Y_test.shape[1]
        for i in range(number_of_categories):
            file.write(f"\n## Category: {category_names[i]}\n")
            class_report =  pd.DataFrame(classification_report(Y_test.iloc[:, i], y_pred[:, i], labels=[0,1], output_dict=True)).transpose()
            class_report = class_report.iloc[0:2, :]
            file.write(tabulate(class_report, headers='keys', tablefmt='pipe'))



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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        # Print best parameters
        print("Best Parameters:", model.best_params_)
        
        print('Evaluating model...')
        # evaluate_model(model, X_test, Y_test, category_names)
        rp_path = './reports/'
        evaluate_model(model, X_test, Y_test, category_names, f'{rp_path}classification_report.md')


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