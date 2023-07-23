import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and the categories data from csv files and merge them into one dataframe.
    
    Args:
        messages_filepath (str): path to the messages csv file
        categories_filepath (str): path to the categories csv file
        
    Returns:
        df: pandas dataframe, combined dataframe of messages and categories
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge the messages and categories datasets using the common id
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the df dataframe and use one-hot encoding for the categories.
    
    Args:
        df: pandas dataframe, combined dataframe of messages and categories
    
    Returns:
        df: pandas dataframe, cleaned dataframe
    
    """
    
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # get the names of the categories from the first row and delete the last two characters
    category_colnames = categories.iloc[0].str[:-2]
    
    # rename the columns of categories
    categories.columns = category_colnames
    
    #convert category values to just numbers 0 or 1
    for columns in categories:
        # set each value to be the last character of the string
        categories[columns] = categories[columns].str[-1]
        # convert column from string to numeric
        categories[columns] = categories[columns].astype(int)
    
    # Drop the categories column from the df dataframe
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the df dataframe with the categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.
     
    Args:
        df: pandas dataframe, cleaned dataframe
        database_filename: str, database filename
        
    Returns:
        None
    
    """
    # Create an SQLite engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # Save the clean dataset into an sqlite database
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()