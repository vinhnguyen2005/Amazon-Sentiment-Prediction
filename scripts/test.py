import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_PATH)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_pipeline_logistic.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found at {MODEL_PATH}")
        return None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def predict_sentiment(model, text):
    try:
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = probability[prediction] * 100
        
        return sentiment, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None, None

def main():
    model = load_model()
    if model is None:
        return

    test_reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality, waste of money. Very disappointed.",
        "It's okay, nothing special but works as expected.",
        "I love this! Exceeded all my expectations. Highly recommend!",
        "Horrible experience. Product broke after 2 days.",
        "Great value for money. Very satisfied with the purchase.",
        "Not worth it. Poor quality and bad customer service.",
        "Fantastic! This is exactly what I needed. Five stars!"
    ]
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS TEST RESULTS")
    print("="*80 + "\n")
    
    for i, review in enumerate(test_reviews, 1):
        sentiment, confidence = predict_sentiment(model, review)
        
        if sentiment:
            print(f"Review {i}:")
            print(f"Text: {review}")
            print(f"Prediction: {sentiment} (Confidence: {confidence:.2f}%)")
            print("-" * 80)
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Enter your own reviews (type 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        user_input = input("\nEnter a review to analyze: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            print("Please enter a valid review.")
            continue
        
        sentiment, confidence = predict_sentiment(model, user_input)
        
        if sentiment:
            print(f"\n→ Prediction: {sentiment} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()