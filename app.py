from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return f"Sentiment: {sentiment} (polarity: {polarity:.2f})"

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