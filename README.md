# Fake News Detection System

This project detects whether a news article is real or fake using natural language processing and machine learning.

## Features

- Text preprocessing
- Feature extraction using TF-IDF
- Classification of news articles

## Tech Stack

- Python
- NLP
- Scikit-learn

## Project Structure

Fake-News-Detection/
- data/
- preprocessing/
- model/
- app.py

## Included Files:
- fake_news_detection.py : Script for preprocessing, training, and saving model
- app.py                : Streamlit web app for fake news detection
- model.pkl             : Trained Naive Bayes model (generated after running fake_news_detection.py)
- vectorizer.pkl        : TF-IDF vectorizer (generated after running fake_news_detection.py)


Dataset source: Fake and Real News Dataset - Kaggle
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download

## How It Works

1. Input news text is cleaned and processed
2. Features are extracted using TF-IDF
3. Model classifies the news as real or fake

## Future Improvements

- Use of deep learning models
- Real-time news verification
- Web interface for user input

