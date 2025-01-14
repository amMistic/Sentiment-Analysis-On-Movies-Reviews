import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(random_state=42)

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def prepare_data(self, df):
        print("Preprocessing data...")
        # Create copy of dataframe
        processed_df = df.copy()
        
        # Preprocess reviews
        processed_df['processed_review'] = processed_df['review'].apply(self.preprocess_text)
        
        # Convert sentiment to numerical values
        processed_df['sentiment'] = processed_df['sentiment'].map({'positive': 1, 'negative': 0})
        
        return processed_df

    def train(self, X_train, y_train):
        print("Vectorizing text...")
        # Transform text to TF-IDF features
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        print("Training model...")
        # Train the model
        self.model.fit(X_train_vectorized, y_train)

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        # Transform test data
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Print evaluation metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, text):
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return sentiment, confidence

