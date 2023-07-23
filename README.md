# Disaster Response Pipeline Project

This project is designed to analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages. The provided dataset set contains real messages that were sent during disaster events. For that we build a machine learning pipeline to classify appropriately these events in order to dispatch the messages to the concerned disaster relief agency.

We provide a web app where an emergency worker can input a new message and get classification results in the pre-defined categories. Furthermore, some visualizations are provided to show the data distribution.



## Project Components
### 1. ETL Pipeline
ETL in a Python script, `process_data.py`, to prepare the data. The script performs the following tasks:

    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores the clean data in a SQLite database

### 2. ML Pipeline
ML pipeline in a Python script, `train_classifier.py`, to train and export a classifier. The script performs the following tasks:

    - Loads data from the SQLite database
    - Tokenizes and lemmatizes text data
    - Builds an ML pipeline using `CountVectorizer`, `TfidfTransformer`, and `MultiOutputClassifier`
    - Trains and tunes a model using `GridSearchCV`
    - Outputs results on the test set
    - Exports the final model as a pickle file

### 3. Web App
A Flask web app that shows data visualizations and classifies disaster messages. The web app includes:

    - A data visualization using Plotly in the `go.html` template file
    - A Flask app that runs the web app

## Getting Started
### Dependencies
- Python 3.9
- NumPy
- Pandas
- Scikit-Learn
- NLTK
- Flask
- Plotly
- SQLAlchemy

### Installation
- Clone the repository
```bash
git clone https://github.com/naoufal51/disaster_response.git
```
- Create a virtual environment
```bash
python3 -m venv venv
```
- Activate the virtual environment
```sh
source venv/bin/activate
```
- Install the dependencies
```bash
pip install -r requirements.txt
```

## Running the project

### ETL Pipeline
To run the ETL pipeline that cleans data and stores in database, run the following command:
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
### ML Pipeline
To run the ML pipeline that trains classifier and saves, run the following command:
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
### Flask Web App
To run the Flask web app, run the following command:
```bash
python run.py
```
Then go to http://0.0.0.0:3001/ or http://localhost:3001/

## Web App Screenshots
### Home Page
![Home Page](figures/webapp_interface.png)

### Classification Results
![Classification Results](figures/webapp_classification.png)

## Category Imbalance and Model Performance
![Category Imbalance](figures/category_imbalance.png)
This dataset is highly imbalanced as shown in figure above where a large number of categories have very few samples. This imbalance issue can lead to a biased model that favors the majority classes. This issue is even more critical as we are dealing with disaster response messages where we want to make sure that all categories are well represented. 
### Model Performance
In oder to observe the impact of the imbalance issue, we use the F1 score as a metric to evaluate the model performance. The F1 score is a good metric to assess the balance between precision and recall.


## Acknowledgements
- [Appen](https://appen.com/) for providing the dataset

