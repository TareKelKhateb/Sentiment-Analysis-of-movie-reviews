# Sentiment Analysis of Movie Reviews

## Overview
This project aims to analyze the sentiment of movie reviews using natural language processing (NLP) techniques. It includes a dataset of 1000 positive movie reviews and 1000 negative movie reviews.

## Dataset
The dataset comprises 2000 movie reviews, divided into two categories:

Positive reviews: 1000 reviews  
Negative reviews: 1000 reviews

## Preprocessing


Removal of punctuation and special characters.  
Conversion of text to lowercase.  
Tokenization of text.  
Removal of stopwords.  
Lemmatization of tokens.  

## Model Training
The sentiment analysis model is trained using the preprocessed movie review data. Key steps in model training include:

-Vectorization of text data using TF-IDF (Term Frequency-Inverse Document Frequency).  

-Splitting the dataset into training and testing sets.  

-Training a machine learning model (e.g., Support Vector Machine, Naive Bayes) on the training data.  

-Evaluating the model's performance on the testing data.


## Evaluating
we've found Random forest fits data well    
<img src="https://github.com/TareKelKhateb/Sentiment-Analysis-of-movie-reviews/assets/110000941/a300a628-4fd9-4dba-8bc8-388122cf3285" width="350">  


and it's confusion matrix    
<img src="https://github.com/TareKelKhateb/Sentiment-Analysis-of-movie-reviews/assets/110000941/5a27a54b-d2f4-4539-b9b0-93618184009c" width="350">  
