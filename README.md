# Food Review Analysis

## Overview

This project focuses on sentiment analysis and classification of food product reviews using Natural Language Processing (NLP) techniques. The goal is to gain insights into user sentiments, classify reviews into positive or negative categories, and explore relationships between review characteristics and user engagement.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Sentiment Analysis](#sentiment-analysis)
4. [Review Classification](#review-classification)
    - [Naive Bayes Model](#naive-bayes-model)
    - [Logistic Regression Model](#logistic-regression-model)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
6. [Review Characteristics Analysis](#review-characteristics-analysis)
7. [Conclusion](#conclusion)

## Introduction

The project involves analyzing a dataset of food product reviews. The main objectives include preprocessing the data, conducting sentiment analysis, implementing review classification models, performing exploratory data analysis (EDA), and analyzing review characteristics.

## Data Loading and Preprocessing

- The dataset is loaded from a text file containing food product reviews.
- Relevant columns such as 'productid', 'userid', 'helpfulness', 'score', 'time', 'summary', and 'review' are extracted.
- Data is cleaned by handling missing values, removing duplicates, and combining relevant columns.

## Sentiment Analysis

- Text preprocessing techniques such as removing stopwords and stemming are applied to the review text.
- Sentiment intensity is evaluated using the VADER sentiment analysis tool.
- Reviews are categorized into positive, negative, or neutral sentiments based on the compound scores.

## Review Classification

### Naive Bayes Model

- A Naive Bayes model is implemented to classify reviews into positive or negative categories.
- The review text is converted into a Bag-of-Words representation using CountVectorizer.
- The model is trained on the training set and evaluated on the testing set.

### Logistic Regression Model

- A Logistic Regression model is implemented for review classification.
- Stop words are removed during CountVectorizer processing.
- The model is trained and evaluated on the training and testing sets, respectively.

## Exploratory Data Analysis (EDA)

- Visualizations include bar plots for sentiment distribution, count plots for percentage upvotes, and histograms for helpfulness ratio.
- Word clouds are generated for positive and negative reviews, as well as an overall word cloud.

## Review Characteristics Analysis

- Review scores are analyzed through a bar plot to showcase the distribution of scores.
- A scatter plot illustrates the relationship between the number of helpful votes and review scores.
- A box plot depicts the distribution of review scores based on sentiment.

## Conclusion

This project provides a comprehensive analysis of sentiment in food product reviews, including the implementation of classification models for positive and negative reviews. The EDA visualizations offer insights into sentiment distribution, helpfulness ratio, and review characteristics. The project aims to enhance understanding of user sentiments and preferences in the context of food product reviews.
