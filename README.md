# Disaster Response Pipeline Project

This project aims to build a model for an API that classifies disaster messages based on disaster data from Appen (formally Figure 8). The provided dataset contains real messages that were sent during disaster events. We build a machine learning pipeline to classify possible disaster events. Then, we dispatch the messages to the concerned disaster relief agency.

We provide a web app where an emergency worker can input a new message and get classification results in the pre-defined categories. Furthermore, we provide some visualizations to show the data distribution.


https://github.com/naoufal51/disaster_response/assets/15954923/63695160-ae94-4165-968a-0b93e9e9e035

## Live Web app:
**click on the image below** 

*Note:*  You can visualize the app but the model might not work due to limited resources on the Free tier of `render.com`

<a href="https://disasterapp.onrender.com/">
  <img src="figures/family.png" width="200" height="200" alt="Disaster Response Webapp. Generate by hotpot.ai.">
</a>


## Project Components
### 1. ETL Pipeline
The ETL pipeline, `process_data.py`, is used to to prepare the data:

    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores the clean data in a SQLite database

### 2. ML Pipeline
The ML pipeline ,`train_classifier.py`, is used to train and export a classifier:

    - Loads data from the SQLite database
    - Tokenizes and lemmatizes text data
    - Builds an ML pipeline using `CountVectorizer`, `TfidfTransformer`, and `MultiOutputClassifier`
    - Add features such as `negation_counter`, `verb_counter`, `emotion_counter`, `punctuation_counter`, `text_length`, `capitalization_counter`, `subjectivity`, `polarity`, `ner`(not in the light version).
    - Trains and tunes a model using `GridSearchCV`
    - Outputs results on the test set in `reports/classification_report.md`
    - Exports the final model as a pickle file in `models/classifier.pkl`

### 3. Web App
A Flask web app that shows data visualizations and runs the inference using trained model to classify disaster messages. The web app includes:

    - A data visualization using Plotly in the `go.html` template file
    - A Flask app that runs the web app accessible in `app/run.py`. 
    - Mutliple visualizations are accessible such as:
        * Distribution of Message Genres
        * Distribution of Messages Across Categories and Genres
        * Average Message Length by Genre
        * Average Message Length by Category

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
- SpaCy
- TextBlob
- gunicorn
- contractions

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
You can choose to run the app with the model with or without NER (`classifier_light.pkl`). Go to `app/run.py` to make the necessary changes.

To run the Flask web app, run the following command:
```bash
gunicorn app.run:app 
```
Then go to http://127.0.0.1:8000 or http://localhost:8000/

## Web App Screenshots
### Home Page
![Home Page](figures/webapp_interface.png)

![Home Page cont](figures/webapp_interface_cont.png)
### Classification Results
![Classification Results](figures/webapp_classification.png)

## Category Imbalance and Model Performance
![Category Imbalance](figures/category_imbalance.png)
This dataset is highly imbalanced as shown in the figure above where a large number of categories have very few samples. This imbalance issue can lead to a biased model that favors the majority classes. This issue is even more critical as we are dealing with disaster response messages where we want to make sure that all categories are well represented. 
There are several techniques to mitigate this issue:
1. **Resampling:** Majority undersampling or minority oversampling.
2. **SMOTE (Synthetic Minority Over-Sampling Techniques):** Instead of simple oversampling of minority classes. SMOTE exploits the feature space to create a new synthetic sample based on k-nearest neighbors.
3. **Data augmentation:** There exist multiple strategies to generate text using data augmentation. Synonym replacement, random insertion, and random deletion are among them.
4. **Cost-Sensitive Learning:** Apply a penalty score in misclassified instances to decrease the total misclassification cost.
5. **Ensemble Methods:** Bagging or Boosting can mitigate class imbalance. 

In order to accurately evaluate our model in the presence of class imbalance issues, we use the F1 score as a metric to evaluate the model performance. The F1 score is a good metric to assess the balance between precision and recall.

## (Optional) Advanced Modeling with Transformers
In addition to the models mentioned above, we also explored leveraging state-of-the-art transformer models using the Hugging Face Transformers library in a Kaggle Notebook. This approach allows us to tap into powerful pre-trained models like BERT, which have achieved high performance across a variety of NLP tasks, including text classification.

### Overview:
- **Loading:** We employ `AutoTokenizer` and `AutoModelForSequenceClassification` for tokenizing and loading the ‘bert-base-uncased’ model.
- **Training & Evaluation:** The model is trained using the `Trainer` class, with a weighted loss function for class imbalance, and evaluated using metrics like accuracy and F1 score on a test set.
- **Experimentation:** For those with ample computational resources, the notebook in the repository provides an optional, more sophisticated methodology, offering the possibility for further tuning and experimentation.

### Note:
Given the resource-intensive nature of transformer models, consideration is required for deployment in constrained environments.

You can refer to the provided [kaggle notebook](https://www.kaggle.com/code/naoufal51/disaster-pipeline-transformer) for a detailed walk-through and implementation.

---
## Acknowledgements
- [Appen](https://appen.com/) for providing the dataset
- [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) for providing NRC Emotion Lexicon. 

