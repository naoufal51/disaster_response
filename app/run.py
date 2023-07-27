import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# def tokenize(text):
#     """
#     Tokenize and lemmatize the disaster messages.
    
#     Args:
#         text: string, disaster message to be processed (tokenized)
    
#     Returns:
#         clean_tokens: list of strings, tokenized text
#     """
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


# load data
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
# df = pd.read_sql_table('disaster_messages', engine)
df = pd.read_csv('disaster_messages_comb.csv.gz', compression='gzip')

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Displays visualizations of the data 
    and allows the user to enter a message to classify.
    
    Args:
        None
        
    Returns:
        Flask response object with rendered template 
    """   
    # Count the number of messages by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Count the number of messages by category
    category_counts = df.iloc[:, 4:].sum()
    category_names = list(category_counts.index)


    # Calculate the percentage of messages in each category by genre
    def calculate_genre_percentage(category):
        category_data = df[df[category] == 1]
        category_genre_counts = category_data['genre'].value_counts()
        return category_genre_counts / category_genre_counts.sum()

    genre_percentage_df = pd.DataFrame([calculate_genre_percentage(category) for category in category_names], index=category_names)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=genre_percentage_df['direct'],
                    name='Direct'
                ),
                Bar(
                    x=category_names,
                    y=genre_percentage_df['news'],
                    name='News'
                ),
                Bar(
                    x=category_names,
                    y=genre_percentage_df['social'],
                    name='Social'
                ),
            ],
            'layout': {
                'title': 'Distribution of messages across categories and genres',
                'yaxis': {
                    'title': "Proporation"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                },
                'barmode': 'stack'
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = [f"graph-{i}" for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handles user query and displays model results.
    
    Args:
        None
        
    Returns:
        Flask response object with rendered go.html template.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run()


if __name__ == '__main__':
    main()
