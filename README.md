# FoodReviewAnalysis_Group4

# Sentiment Analysis and Machine Learning Documentation

## Overview

This document serves as documentation for the sentiment analysis code, encompassing text preprocessing, sentiment classification, and machine learning model training. The primary objective is to analyze user reviews and classify them into positive, negative, or neutral sentiments. The code includes data loading, preprocessing, exploratory data analysis, model training using both Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) representations, hyperparameter tuning, and model evaluation.

## Table of Contents

1. Dependencies
2. Data Loading
3. Text Preprocessing
4. Sentiment Analysis
5. Visualization
6. Model Training
    - Logistic Regression with BoW
    - Naive Bayes Classifier with BoW
    - Logistic Regression with TF-IDF
    - Naive Bayes Classifier with TF-IDF
7. Model Evaluation
8. Save and Load Models
9. Considerations
10. Conclusion

## Dependencies

### Google Colab
Google Colab is used as the development environment, allowing for easy collaboration and access to GPU resources.

### pandas, numpy
Pandas and NumPy are fundamental libraries for data manipulation and numerical operations.

### matplotlib, seaborn
Matplotlib and Seaborn are used for data visualization, aiding in the analysis of distribution patterns and trends.

### wordcloud
The WordCloud library is employed to create visually appealing word clouds for positive and negative reviews.

### nltk
The Natural Language Toolkit (NLTK) is utilized for natural language processing tasks, including stopword removal and stemming.

### beautifulsoup4
Beautiful Soup is employed to remove HTML tags from the text.

### tqdm
TQDM is used to display progress bars, enhancing the user experience during lengthy operations.

### scikit-learn
Scikit-learn provides tools for machine learning, including model training, evaluation, and hyperparameter tuning.

## Data Loading

The code loads user reviews from a text file using Google Colab. Each review is parsed, and relevant information is extracted, resulting in a Pandas DataFrame. This step is essential for subsequent analysis and model training.

## Text Preprocessing

### Steps
1. **HTML and URL Removal:** HTML tags and URLs are removed from the reviews.
2. **Lowercasing:** The text is converted to lowercase for consistency.
3. **Non-Alphabetic Characters Removal:** Non-alphabetic characters are removed, retaining only letters.
4. **Stopword Removal:** Common English stopwords are removed to focus on meaningful words.
5. **Stemming:** Words are reduced to their root form using stemming.

These preprocessing steps ensure that the text data is clean and ready for analysis.

## Sentiment Analysis

The VADER sentiment intensity analyzer is employed to assign a compound score to each review. The compound score is used to categorize reviews as positive, negative, or neutral.

## Visualization

### Plots
1. **Distribution of Sentiment Categories:** A bar plot illustrates the distribution of positive, negative, and neutral sentiments.
2. **Word Clouds:** Word clouds visually represent the most frequent words in positive and negative reviews.
3. **Distribution of Review Scores:** A bar plot showcases the distribution of review scores.
4. **Distribution of Review Lengths:** A bar plot displays the distribution of review lengths.
5. **Time Series Plot:** A time series plot illustrates the number of reviews over time.

These visualizations provide insights into the data distribution and trends.

## Model Training

### Logistic Regression with BoW

1. **Hyperparameter Tuning:** Different values of C are tested.
2. **Model Training and Evaluation:** The logistic regression model is trained and evaluated.
3. **Model Saving:** The trained model is saved to a pickle file.

### Naive Bayes Classifier with BoW

1. **Hyperparameter Tuning:** Different values of alpha are tested.
2. **Model Training and Evaluation:** The Naive Bayes classifier is trained and evaluated.
3. **Model Saving:** The trained model is saved to a pickle file.

### Logistic Regression with TF-IDF

1. **Hyperparameter Tuning:** Different values of C are tested.
2. **Model Training and Evaluation:** The logistic regression model with TF-IDF is trained and evaluated.
3. **Model Saving:** The trained model is saved to a pickle file.

### Naive Bayes Classifier with TF-IDF

1. **Hyperparameter Tuning:** Different values of alpha are tested.
2. **Model Training and Evaluation:** The Naive Bayes classifier with TF-IDF is trained and evaluated.
3. **Model Saving:** The trained model is saved to a pickle file.

## Model Evaluation

The final model, Logistic Regression with TF-IDF, is evaluated using accuracy scores and a confusion matrix. This step assesses the model's performance on the test dataset.

## Save and Load Models

The code includes functionality to save both the TF-IDF vectorizer and the final model to pickle files. This is crucial for model deployment and future use.

## Considerations

### Handling Imbalanced Classes
Down-sampling is applied to balance the class distribution, ensuring the model is not biased toward the majority class.

### Handling NaN Values
Rows with NaN scores are removed to prevent issues during model training.

### Model Selection
Logistic Regression and Naive Bayes models are explored for their suitability in sentiment analysis. These models are known for their simplicity and effectiveness in text classification tasks.

## Conclusion

The code provides a comprehensive pipeline for sentiment analysis, covering data loading, preprocessing, model training, and evaluation. By utilizing both BoW and TF-IDF representations with different machine learning models, the code achieves satisfactory accuracy in sentiment classification. The final model can be saved and deployed for real-world applications.
