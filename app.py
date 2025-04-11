from textblob import TextBlob

def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Get the sentiment polarity (range: -1 to 1)
    polarity = blob.sentiment.polarity
    
    # Determine sentiment category based on polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    while True:
        # Get input from user
        text = input("Enter a sentence (or 'quit' to exit): ")
        
        # Check if user wants to quit
        if text.lower() == 'quit':
            break
            
        # Analyze and print the sentiment
        sentiment = analyze_sentiment(text)
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    print("Simple Sentiment Analysis Program")
    print("--------------------------------")
    main()