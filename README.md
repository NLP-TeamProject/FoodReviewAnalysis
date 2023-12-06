# FoodReviewAnalysis_Group4

Certainly! Below is a template for documenting your code in a professional manner. This documentation provides an overview of the code, its purpose, usage, and important considerations.

# Sentiment Analysis and Machine Learning Documentation

## Overview

This document serves as documentation for the sentiment analysis code, encompassing text preprocessing, sentiment classification, and machine learning model training. The primary objective is to analyze user reviews and classify them into positive, negative, or neutral sentiments. The code includes data loading, preprocessing, exploratory data analysis, model training using both Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) representations, hyperparameter tuning, and model evaluation.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Data Loading](#data-loading)
3. [Text Preprocessing](#text-preprocessing)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Visualization](#visualization)
6. [Model Training](#model-training)
    - [Logistic Regression with BoW](#logistic-regression-with-bow)
    - [Naive Bayes Classifier with BoW](#naive-bayes-classifier-with-bow)
    - [Logistic Regression with TF-IDF](#logistic-regression-with-tf-idf)
    - [Naive Bayes Classifier with TF-IDF](#naive-bayes-classifier-with-tf-idf)
7. [Model Evaluation](#model-evaluation)
8. [Save and Load Models](#save-and-load-models)
9. [Considerations](#considerations)
10. [Conclusion](#conclusion)

## Dependencies <a name="dependencies"></a>

- Google Colab
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- nltk
- beautifulsoup4
- tqdm
- scikit-learn

## Data Loading <a name="data-loading"></a>

The code loads user reviews from a text file using Google Colab. Each review is parsed, and relevant information is extracted, resulting in a Pandas DataFrame.

## Text Preprocessing <a name="text-preprocessing"></a>

The preprocessing steps include:
- Removal of HTML tags and URLs
- Conversion to lowercase
- Removal of non-alphabetic characters
- Removal of stopwords
- Stemming

## Sentiment Analysis <a name="sentiment-analysis"></a>

The sentiment analysis utilizes the VADER sentiment intensity analyzer to assign a compound score to each review. The compound score is used to categorize reviews as positive, negative, or neutral.

## Visualization <a name="visualization"></a>

- Distribution of sentiment categories
- Word clouds for positive and negative reviews
- Distribution of review scores
- Distribution of review lengths
- Time series plot for the number of reviews over time

## Model Training <a name="model-training"></a>

The code trains machine learning models for sentiment classification using both BoW and TF-IDF representations. It includes:

### Logistic Regression with BoW <a name="logistic-regression-with-bow"></a>

- Hyperparameter tuning with different values of C
- Model training and evaluation
- Model saving to a pickle file

### Naive Bayes Classifier with BoW <a name="naive-bayes-classifier-with-bow"></a>

- Hyperparameter tuning with different values of alpha
- Model training and evaluation
- Model saving to a pickle file

### Logistic Regression with TF-IDF <a name="logistic-regression-with-tf-idf"></a>

- Hyperparameter tuning with different values of C
- Model training and evaluation
- Model saving to a pickle file

### Naive Bayes Classifier with TF-IDF <a name="naive-bayes-classifier-with-tf-idf"></a>

- Hyperparameter tuning with different values of alpha
- Model training and evaluation
- Model saving to a pickle file

## Model Evaluation <a name="model-evaluation"></a>

The final model, Logistic Regression with TF-IDF, is evaluated using accuracy scores and a confusion matrix.

## Save and Load Models <a name="save-and-load-models"></a>

The code includes functionality to save both the TF-IDF vectorizer and the final model to pickle files. This is crucial for model deployment and future use.

## Considerations <a name="considerations"></a>

- Handling imbalanced classes: Down-sampling is applied to balance the class distribution.
- Handling NaN values: Rows with NaN scores are removed.
- Model selection: Logistic Regression and Naive Bayes models are explored for their suitability in sentiment analysis.

## Conclusion <a name="conclusion"></a>

This code provides a comprehensive pipeline for sentiment analysis, encompassing data loading, preprocessing, model training, and evaluation. It demonstrates the use of both BoW and TF-IDF representations with different machine learning models. The final model achieves satisfactory accuracy in sentiment classification.

---

Feel free to customize this documentation based on your specific code and requirements.
