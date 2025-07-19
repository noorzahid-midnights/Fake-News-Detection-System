# app.py - Streamlit Web App for Fake News Detection
import streamlit as st
import pickle
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it's Real or Fake.")

input_text = st.text_area("News Article Text")

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        processed_text = preprocess(input_text)
        vector = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(vector)[0]
        result = "🔴 Fake News" if prediction == 0 else "🟢 Real News"
        st.success(f"Prediction: {result}")
