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
        
        # Expanded emotion words with variations
        self.positive_words = {
            # Basic positive emotions
            'happy', 'joy', 'joyful', 'delighted', 'pleased', 'glad',
            'cheerful', 'content', 'satisfied', 'blessed', 'blissful',
            
            # Excitement and enthusiasm
            'excited', 'thrilled', 'enthusiastic', 'eager', 'energetic',
            'passionate', 'motivated', 'inspired', 'wonderful', 'amazing',
            
            # Love and affection
            'love', 'adore', 'cherish', 'fond', 'caring', 'affectionate',
            
            # Success and achievement
            'proud', 'accomplished', 'successful', 'achieved', 'victorious',
            'triumphant', 'excellent', 'outstanding', 'perfect', 'brilliant',
            
            # Gratitude and appreciation
            'grateful', 'thankful', 'appreciative', 'blessed', 'fortunate',
            
            # Peace and calm
            'peaceful', 'calm', 'relaxed', 'serene', 'tranquil', 'comfortable',
            
            # Positive descriptors
            'good', 'great', 'awesome', 'fantastic', 'superb', 'nice', 'lovely',
            'beautiful', 'wonderful', 'magnificent', 'marvelous', 'splendid',
            
            # Variations and related forms
            'enjoy', 'enjoying', 'enjoyed', 'happiness', 'happier', 'happiest',
            'delight', 'delighting', 'delighted', 'pleasure', 'pleasurable',
            'satisfy', 'satisfying', 'satisfied', 'satisfaction'
        }
        
        self.negative_words = {
            # Basic negative emotions
            'sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'melancholy',
            'upset', 'upsetting', 'distressed', 'troubled', 'disturbed',
            
            # Anger and frustration
            'angry', 'mad', 'furious', 'enraged', 'irritated', 'annoyed',
            'frustrated', 'aggravated', 'agitated', 'bitter', 'hostile',
            
            # Fear and anxiety
            'afraid', 'scared', 'fearful', 'anxious', 'worried', 'concerned',
            'nervous', 'tense', 'stressed', 'panicked', 'terrified',
            
            # Disappointment and regret
            'disappointed', 'disappointing', 'regretful', 'sorry', 'ashamed',
            'guilty', 'remorseful', 'dismayed', 'disheartened',
            
            # Pain and suffering
            'hurt', 'painful', 'suffering', 'aching', 'grieving', 'heartbroken',
            'devastated', 'crushed', 'tormented', 'agonized',
            
            # Negative descriptors
            'bad', 'terrible', 'horrible', 'awful', 'dreadful', 'poor',
            'unpleasant', 'unfair', 'wrong', 'worse', 'worst', 'negative',
            
            # Variations and related forms
            'sadness', 'sadder', 'saddest', 'anger', 'angrier', 'angriest',
            'hate', 'hatred', 'hating', 'hated', 'dislike', 'disliking',
            'upset', 'upsetting', 'upsets', 'frustrate', 'frustrating',
            'frustrated', 'frustration', 'disappoint', 'disappointing',
            'disappointed', 'disappointment'
        }
        
        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
            'utterly', 'highly', 'incredibly', 'super', 'especially', 'particularly',
            'thoroughly', 'deeply', 'strongly', 'seriously', 'desperately', 'more',
            'most', 'quite', 'rather', 'somewhat', 'too', 'so', 'such'
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 
            'cant', 'cannot', 'wont', 'wouldnt', 'shouldnt', 'dont', 'doesnt', 'didnt',
            'isnt', 'arent', 'aint', 'hardly', 'rarely', 'scarcely', 'barely'
        }
        
        self.comparison_words = {
            'but', 'however', 'although', 'though', 'yet', 'still', 'nevertheless',
            'nonetheless', 'despite', 'spite', 'while', 'whereas', 'unlike', 'than',
            'compare', 'compared', 'then'
        }
        
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
            'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
            'were', 'will', 'with'
        }

    def preprocess_text(self, text):
        # Convert to lowercase and strip whitespace
        text = str(text).lower().strip()
        
        # Store original punctuation counts
        self.exclamation_count = text.count('!')
        self.question_count = text.count('?')
        self.period_count = text.count('.')
        
        # Remove all punctuation except apostrophes for contractions
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def get_sentiment_scores(self, text):
        words = text.split()
        
        # Initialize scores
        scores = {
            'positive_score': 0,
            'negative_score': 0,
            'intensifier_count': 0,
            'negation_count': 0,
            'comparison_count': 0
        }
        
        # Process each word
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.intensifiers:
                scores['intensifier_count'] += 1
                continue
                
            # Check for negations
            if word in self.negation_words:
                scores['negation_count'] += 1
                continue
                
            # Check for comparison words
            if word in self.comparison_words:
                scores['comparison_count'] += 1
                continue
            
            # Check for sentiment words
            if word in self.positive_words:
                # Look back for negations and intensifiers
                modifier = 1
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        modifier *= -1
                    elif words[j] in self.intensifiers:
                        modifier *= 1.5
                scores['positive_score'] += modifier
                
            elif word in self.negative_words:
                # Look back for negations and intensifiers
                modifier = 1
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        modifier *= -1
                    elif words[j] in self.intensifiers:
                        modifier *= 1.5
                scores['negative_score'] += modifier
        
        return scores

    def determine_sentiment(self, text):
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get sentiment scores
        scores = self.get_sentiment_scores(processed_text)
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Calculate final sentiment score
        positive_score = scores['positive_score']
        negative_score = scores['negative_score']
        
        # Adjust scores based on comparison words
        if scores['comparison_count'] > 0:
            words = processed_text.split()
            # Find the position of comparison words and weight sentiments accordingly
            for i, word in enumerate(words):
                if word in self.comparison_words:
                    # Words after comparison word get higher weight
                    for j in range(i+1, len(words)):
                        if words[j] in self.positive_words:
                            positive_score += 0.5
                        elif words[j] in self.negative_words:
                            negative_score += 0.5
        
        # Calculate final score
        final_score = (positive_score - negative_score) + polarity
        
        # Determine sentiment with adjusted thresholds
        if abs(final_score) < 0.2:  # Increased neutral threshold
            return 'Neutral'
        elif final_score > 0:
            return 'Positive'
        else:
            return 'Negative'

    def get_additional_features(self, text):
        processed_text = self.preprocess_text(text)
        scores = self.get_sentiment_scores(processed_text)
        
        blob = TextBlob(text)
        
        return [
            blob.sentiment.polarity,
            blob.sentiment.subjectivity,
            scores['positive_score'],
            scores['negative_score'],
            scores['intensifier_count'],
            scores['negation_count'],
            self.exclamation_count,
            self.question_count,
            self.period_count,
            scores['comparison_count']
        ]

    def load_and_prepare_data(self, filename):
        df = pd.read_csv(filename)
        
        print("Preprocessing text...")
        df['processed_text'] = df['Text'].apply(self.preprocess_text)
        
        print("Extracting additional features...")
        additional_features = df['Text'].apply(self.get_additional_features)
        additional_features_df = pd.DataFrame(
            additional_features.tolist(),
            columns=['polarity', 'subjectivity', 'positive_score', 'negative_score',
                    'intensifier_count', 'negation_count', 'exclamation_count',
                    'question_count', 'period_count', 'comparison_count']
        )
        
        print("Determining sentiments...")
        df['Sentiment'] = df['Text'].apply(self.determine_sentiment)
        
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
        
        # Use rule-based sentiment if it differs and has strong signals
        scores = self.get_sentiment_scores(processed_text)
        if (abs(scores['positive_score'] - scores['negative_score']) > 1 or 
            probability < 0.7):
            return rule_based_sentiment, probability
        
        return sentiment, probability

def main():
    analyzer = EnhancedSentimentAnalyzer()
    
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = analyzer.load_and_prepare_data('sentimentdataset.csv')
    
    print("Training the model...")
    analyzer.model.fit(X_train, y_train)
    
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