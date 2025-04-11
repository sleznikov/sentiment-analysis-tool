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
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        )
        self.label_encoder = LabelEncoder()
        
        # Expanded sentiment lexicons
        self.positive_words = {
            'good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful', 'fantastic', 
            'beautiful', 'enjoy', 'enjoying', 'pleased', 'amazing', 'superb', 'nice', 'best',
            'positive', 'perfect', 'fun', 'excited', 'exciting', 'blessed', 'fantastic',
            'delighted', 'grateful', 'thankful', 'happy', 'glad', 'pleasant', 'lovely',
            'outstanding', 'brilliant', 'joyful', 'success', 'successful', 'excellent',
            'magnificent', 'wonderful', 'remarkable', 'fabulous', 'peaceful', 'fortunate',
            'proud', 'delight', 'sunshine', 'kindness', 'kind', 'sweet', 'bright', 'super'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'disappointing', 'upset',
            'poor', 'worse', 'worst', 'negative', 'angry', 'mad', 'frustrated', 'disappointing',
            'useless', 'wrong', 'never', 'problem', 'terrible', 'horrible', 'injustice',
            'awful', 'unpleasant', 'unfair', 'evil', 'wicked', 'corrupt', 'unhappy',
            'miserable', 'unfortunate', 'unacceptable', 'dissatisfied', 'heartbroken',
            'failure', 'failed', 'terrible', 'awful', 'horrible', 'dreadful', 'pathetic',
            'cruel', 'tragic', 'violence', 'violent', 'disaster', 'painful', 'pain', 'hurt',
            'hostile', 'severe', 'suffer', 'suffering', 'disaster', 'cant believe', 'outrageous'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
            'utterly', 'highly', 'incredibly', 'super', 'especially', 'particularly',
            'thoroughly', 'deeply', 'strongly', 'seriously', 'desperately'
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 
            'cant', 'cannot', 'wont', 'wouldnt', 'shouldnt', 'dont', 'doesnt', 'didnt',
            'isnt', 'arent', 'aint'
        }
        
        # Simple stopwords list (excluding important sentiment words)
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
            'will', 'with'
        }
    
    def get_additional_features(self, text):
        text_lower = text.lower()
        words = text_lower.split()
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Count intensifiers and negations
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        negation_count = sum(1 for word in words if word in self.negation_words)
        
        # Check for exclamation marks and question marks
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Check for negation followed by positive/negative words
        negated_positive = 0
        negated_negative = 0
        for i in range(len(words)-1):
            if words[i] in self.negation_words:
                if i+1 < len(words):
                    if words[i+1] in self.positive_words:
                        negated_positive += 1
                    elif words[i+1] in self.negative_words:
                        negated_negative += 1
        
        return [polarity, subjectivity, positive_count, negative_count,
                intensifier_count, negation_count, exclamation_count,
                question_count, negated_positive, negated_negative]
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep exclamation and question marks
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)
        
        # Simple word splitting
        words = text.split()
        
        # Filter out stopwords but keep negations and sentiment words
        words = [word for word in words if word not in self.stopwords or 
                word in self.negation_words or 
                word in self.positive_words or 
                word in self.negative_words]
        
        return ' '.join(words)
    
    def determine_sentiment(self, text):
        text_lower = text.lower()
        words = text_lower.split()
        
        # Get base sentiment scores
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Count sentiment indicators
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        # Check for negations and their position
        negation_indices = [i for i, word in enumerate(words) if word in self.negation_words]
        
        # Adjust sentiment counts based on negations
        if negation_indices:
            for neg_idx in negation_indices:
                # Check words following negation (up to 3 words ahead)
                for i in range(neg_idx + 1, min(neg_idx + 4, len(words))):
                    if words[i] in self.positive_words:
                        pos_count -= 1
                        neg_count += 1
                    elif words[i] in self.negative_words:
                        neg_count -= 1
                        pos_count += 1
        
        # Check for intensifiers and their impact
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        
        # Adjust polarity based on intensifiers
        if intensifier_count > 0:
            polarity = polarity * (1 + 0.5 * intensifier_count)
        
        # Combined approach using multiple factors
        sentiment_score = (pos_count - neg_count) * 0.5 + polarity
        
        # Determine final sentiment using a more nuanced approach
        if sentiment_score > 0.2:
            return 'Positive'
        elif sentiment_score < -0.2:
            return 'Neutral'
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
                                                  'negative_count', 'intensifier_count', 'negation_count',
                                                  'exclamation_count', 'question_count', 
                                                  'negated_positive', 'negated_negative'])
        
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