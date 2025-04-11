import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
from textblob import TextBlob

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2
        )
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.label_encoder = LabelEncoder()
        
        # Sentiment lexicons
        self.positive_words = set(['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful', 'fantastic', 
                                 'beautiful', 'enjoy', 'enjoying', 'pleased', 'amazing', 'superb', 'nice', 'best',
                                 'positive', 'perfect', 'fun', 'excited', 'exciting', 'blessed', 'fantastic'])
        
        self.negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'disappointing', 'upset',
                                 'poor', 'worse', 'worst', 'negative', 'angry', 'mad', 'frustrated', 'disappointing',
                                 'useless', 'wrong', 'never', 'problem', 'terrible', 'horrible', 'injustice'])
        
        # Simple stopwords list
        self.stopwords = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                         'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'} - {'not', 'no', 'never'}
        
    def get_additional_features(self, text):
        # Get TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Count positive and negative words
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Check for exclamation marks and question marks
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        return [polarity, subjectivity, positive_count, negative_count, 
                exclamation_count, question_count]
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep exclamation and question marks
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)
        
        # Simple word splitting
        words = text.split()
        
        # Filter out stopwords
        words = [word for word in words if word not in self.stopwords]
        
        return ' '.join(words)
    
    def determine_sentiment(self, text):
        # Rule-based initial classification
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if pos_count > neg_count and polarity > 0:
            return 'Positive'
        elif neg_count > pos_count and polarity < 0:
            return 'Negative'
        elif abs(polarity) < 0.1 and pos_count == neg_count:
            return 'Neutral'
        elif polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    
    def load_and_prepare_data(self, filename):
        df = pd.read_csv(filename)
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_text'] = df['Text'].apply(self.preprocess_text)
        
        # Get additional features
        print("Extracting additional features...")
        additional_features = df['Text'].apply(self.get_additional_features)
        additional_features_df = pd.DataFrame(additional_features.tolist(), 
                                           columns=['polarity', 'subjectivity', 'positive_count', 
                                                  'negative_count', 'exclamation_count', 'question_count'])
        
        # Determine sentiment
        print("Determining sentiments...")
        df['Sentiment'] = df['Text'].apply(self.determine_sentiment)
        
        # Prepare features
        print("Preparing final features...")
        X_text = self.vectorizer.fit_transform(df['processed_text'])
        X_additional = additional_features_df.values
        X = np.hstack((X_text.toarray(), X_additional))
        
        y = self.label_encoder.fit_transform(df['Sentiment'])
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def predict_sentiment(self, text):
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        
        # Get text features
        text_features = self.vectorizer.transform([processed_text]).toarray()
        
        # Get additional features
        additional_features = np.array(self.get_additional_features(text)).reshape(1, -1)
        
        # Combine features
        X = np.hstack((text_features, additional_features))
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = max(self.model.predict_proba(X)[0])
        
        # Get predicted sentiment
        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        
        # Double-check with rule-based system
        rule_based_sentiment = self.determine_sentiment(text)
        
        # If confidence is low, use rule-based sentiment
        if probability < 0.6:
            return rule_based_sentiment, probability
        
        return sentiment, probability

def main():
    analyzer = EnhancedSentimentAnalyzer()
    
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = analyzer.load_and_prepare_data('sentimentdataset.csv')
    
    print("Training the model...")
    analyzer.model.fit(X_train, y_train)
    
    # Calculate and print accuracy
    accuracy = analyzer.model.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    print("\nSentiment Analysis Program")
    print("Enter a sentence to analyze its sentiment (or 'quit' to exit)")
    print("--------------------------------------------------------")
    
    while True:
        text = input("\nEnter a sentence: ")
        
        if text.lower() == 'quit':
            break
        
        sentiment, confidence = analyzer.predict_sentiment(text)
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()