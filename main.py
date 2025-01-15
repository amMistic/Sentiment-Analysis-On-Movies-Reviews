import pandas as pd
from sklearn.model_selection import train_test_split
from src.sentiment_analyzer import SentimentAnalyzer

# Main execution
def main():
    # Load the dataset
    print("\n\nLoading dataset...")
    df = pd.read_csv('data\\IMDB_Dataset.csv')
    
    # Initialize sentiment analyzer
    print('Load analyzer..')
    analyzer = SentimentAnalyzer()
    
    # Prepare data
    print('Preprocess the input dataframe')
    processed_df = analyzer.prepare_data(df)
    
    # Split the data
    print('Splits the dataset into train and testing')
    X_train, X_test, y_train, y_test = train_test_split(
        processed_df['processed_review'],
        processed_df['sentiment'],
        test_size=0.2,
        random_state=42
    )
    
    # Train the model
    print('Model starts training ...')
    analyzer.train(X_train, y_train)
    
    # Evaluate the model
    print('Evaluating model..')
    analyzer.evaluate(X_test, y_test)
    
    # Example prediction
    sample_review = "This movie was amazing! The acting was superb and the plot was fantastic."
    sentiment, confidence = analyzer.predict(sample_review)
    print(f"\nSample Review: {sample_review}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()