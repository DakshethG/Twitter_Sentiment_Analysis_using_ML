Twitter Sentiment Analysis

This project performs sentiment analysis on a large dataset of tweets to classify them as either positive or negative.

Dataset
The dataset used is the Sentiment140 dataset, which contains 1.6 million tweets. The dataset can be downloaded from Kaggle: https://www.kaggle.com/datasets/kazanova/sentiment140

Dependencies
The following libraries are required to run this notebook:

kaggle
swifter
numpy
pandas
re
nltk
sklearn
pickle
You can install these libraries using pip:

!pip install kaggle swifter numpy pandas nltk sklearn pickle

Data Processing

The following steps were performed to process the data:

The dataset was loaded into a pandas DataFrame.
The columns were renamed for better readability.
Missing values were checked (and none were found).
The target variable was converted from '4' to '1' to represent positive sentiment.
Stemming was applied to the tweet text to reduce words to their root form. Stopwords were removed during this process.
The processed text data was converted into numerical data using TfidfVectorizer.
Model Training
A Logistic Regression model was used for sentiment analysis.

The data was split into training and testing sets (80/20 split) using train_test_split.
The Logistic Regression model was initialized with max_iter=1000.
The model was trained on the training data (X_train, Y_train).
Model Evaluation
The trained model was evaluated on both the training and test sets.

Training Data Accuracy: The accuracy on the training data was calculated using accuracy_score.
Test Data Accuracy: The accuracy on the test data was calculated using accuracy_score.
The model achieved an accuracy of approximately 77.8% on the test data.

Saving and Loading the Model
The trained model was saved using pickle for future use without retraining. The saved model can be loaded back into memory for making predictions on new data.

How to Use the Model
Load the saved model using pickle.
Preprocess the new tweet text using the same stemming function and TfidfVectorizer used during training.
Use the loaded model to predict the sentiment of the preprocessed tweet.
