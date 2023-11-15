This repository contains a Python script for sentiment analysis and review classification based on user reviews from a text dataset. The script utilizes Natural Language Processing (NLP) techniques and machine learning models to analyze sentiments and predict review ratings.

# FoodReviewAnalysis_Group4

The Food Review Analysis Project is a data-driven analysis that leverages sentiment analysis and text summarization techniques in Python to gain insights from a large corpus of food reviews.
This project aims to extract valuable information from textual data, providing valuable insights into consumer preferences and restaurant performance.

Key Components of the Project:

Data Collection:
Gathering a diverse dataset of food reviews from various sources, such as social media, review websites, or APIs.
Preprocessing the data to remove noise, including HTML tags, special characters, and irrelevant information.

Sentiment Analysis:
Utilizing Natural Language Processing (NLP) libraries like NLTK, spaCy, or TextBlob to conduct sentiment analysis on the reviews.
Determining the sentiment of each review as positive, negative, or neutral.
Calculating sentiment scores or polarities to quantify the sentiment strength.

Text Summarization:
Applying text summarization techniques, such as extractive or abstractive summarization, to condense the lengthy reviews into concise summaries.
Utilizing libraries or Transformers to generate summaries that capture the essential information.

Visualization and Insights:
Creating visualizations, such as word clouds, sentiment distributions, and summary statistics, to provide a holistic view of the data.
Extracting meaningful insights from the sentiment and summarization results, including identifying common positive and negative aspects of the reviewed foods.


Overview
The script performs the following tasks:

Data Loading and Cleaning:

Reads user reviews from a text file (foods.txt).
Converts the data into a Pandas DataFrame for easier analysis.
Cleans and preprocesses the data by removing duplicates.
Sentiment Analysis:

Combines the 'review' and 'summary' columns.
Applies text preprocessing (lowercasing, removing non-alphabetic characters).
Uses the VADER sentiment analyzer to assign a compound sentiment score to each review.
Classifies reviews as 'Positive,' 'Negative,' or 'Neutral' based on the compound score.
Review Classification (Naive Bayes):

Prepares the data for classification by splitting it into training and testing sets.
Converts text data into a Bag-of-Words representation using CountVectorizer.
Trains a Multinomial Naive Bayes model.

Evaluates the model's accuracy and provides a classification report.
Review Classification (Logistic Regression):

Prepares the data similarly to the Naive Bayes model.
Trains a Logistic Regression model.
Evaluates the model's accuracy and provides a classification report.
