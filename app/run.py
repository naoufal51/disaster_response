import sys
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from utils.utils import *


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)
# drop child alone column (no positive instances)
df.drop(['child_alone'], axis=1, inplace=True)
print('data loaded')
# load model
model = joblib.load("./models/classifier_light.pkl")


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
    def calculate_genre_percentage(df, category_names):
        """
        Calculate the percentage of messages in each category by genre.

        Args:
            category_names (list): list of names of categories present in the dataset
            df (pd.DataFrame): dataframe with the messages and their categories and genre
            
        Returns:
            genre_percentage_df (pd.DataFrame): dataframe with the percentage of messages in each category by genre
        
        """
        genre_percentage_df = pd.DataFrame()
        for category in category_names:
            category_data = df[df[category] == 1]
            category_genre_counts = category_data['genre'].value_counts()
            genre_percentage_df[category] = category_genre_counts / category_genre_counts.sum()
        genre_percentage_df = genre_percentage_df.T
        return genre_percentage_df
    
    def compute_categoty_message_length(df, category_names):
        """
        Calculate the average message length by category.

        Args:
            category_names (list): list of names of categories present in the dataset
            df (pd.DataFrame): dataframe with the messages and their categories and genre
            
        Returns:
            category_length (pd.DataFrame): dataframe with the average message length by category
        
        """
        category_length ={}
        for category in category_names:
            category_data = df[df[category] == 1]
            category_length[category] = category_data['message_length'].mean()
        category_length = pd.Series(category_length)
        category_length = category_length.sort_values(ascending=False)
        return category_length

    genre_percentage_df = calculate_genre_percentage(df, category_names)

    # Compute message length by genre
    df['message_length'] = df['message'].apply(lambda x: len(x.split()))
    genre_length = df.groupby(['genre'])['message_length'].mean()
    genre_names_l = list(genre_length.index)

    # Comote message length by category
    category_length = compute_categoty_message_length(df, category_names)


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
                'title': 'Distribution of Messages Across Categories and Genres',
                'yaxis': {
                    'title': "Proporation"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                },
                'barmode': 'stack'
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names_l,
                    y=genre_length
                )
            ],
            'layout': {
                'title': 'Average Message Length by Genre',
                'yaxis': {
                    'title': "Message Length",
                    },
                'xaxis': {
                'title': "Message Genre"
                    
            }
        }
        },
        {
            'data': [
                Bar(
                    x=category_length.index,
                    y= category_length.values,
                )
            ],
            'layout': {
                'title': 'Average Message Length by Category',
                'yaxis': {
                    'title': "Message Length",
                    },
                'xaxis': {
                'title': "Message Category",
                'tickangle': 45
                    
            }
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()