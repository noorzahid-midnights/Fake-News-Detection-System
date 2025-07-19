# Fake News Detection Project

# 1. Import libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 2. Load dataset (Kaggle fake and real news dataset assumed)
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

df_fake["label"] = 0  # Fake
df_real["label"] = 1  # Real

df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

# 3. Preprocess text
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(tokens)

df['content'] = df['title'] + " " + df['text']
df['content'] = df['content'].apply(preprocess)

# 4. Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['label'].values

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
