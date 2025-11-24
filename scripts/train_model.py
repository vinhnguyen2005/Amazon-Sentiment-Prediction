import pandas as pd
import numpy as np
import os
import logging
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
from supabase import create_client
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_PATH)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

load_dotenv()
SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")

def load_db():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")    
    return create_client(url, key)
    

def load_data(feature_file):
    try:
        supabase = load_db()
        response = supabase.table('amazon_reviews').select("full_review, rating, polarity_score, sentiment_label").limit(50000).execute()
        
        if response.data:
            logging.info("Loaded data from Supabase DB.")
            df = pd.DataFrame(response.data)
            return df
        else:
            logging.warning("Supabase DB returned empty data. Falling back to CSV.")
            
    except Exception as e:
        logging.error(f"Error loading data from DB: {e}. Falling back to CSV.")
        
    if os.path.exists(feature_file):
        logging.info(f"Loading data from CSV file: {feature_file}")
        df = pd.read_csv(feature_file)
        return df
    else:
        logging.error(f"Feature file not found: {feature_file}")
        return None


def train_model(feature_file):
    df = load_data(feature_file)
    if df is None:
        return None
    logging.info(f"Load {len(df)} from supabase successfully")
    
    initial_count = len(df)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['polarity_score'] = pd.to_numeric(df['polarity_score'], errors='coerce')

    consistent_pos = (df['rating'] >= 4) & (df['polarity_score'] > 0.1)
    
    consistent_neg = (df['rating'] <= 2) & (df['polarity_score'] < -0.1)
    
    df_clean = df[consistent_pos | consistent_neg].copy()
    df_clean['target'] = df_clean['rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    dropped_count = initial_count - len(df_clean)
    logging.info(f"Dropped {dropped_count} ambiguous/inconsistent reviews.")
    logging.info(f"Final High-Quality Dataset Size: {len(df_clean)}")
    
    df = df_clean
    X_text = df['full_review'].fillna('').astype(str)
    y = df['target']
    
    logging.info("Splitting Data into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info("Vectorizing text with TF-IDF and train with Logistic Regression...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3), 
            min_df=3,
            max_features=10000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            solver='lbfgs',    
            max_iter=1000,   
            class_weight='balanced' 
        ))
    ])

    param_grid = {
        'clf__C': [0.01, 0.1, 0.5, 1, 5] 
    }
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1, 
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    logging.info(f"Best Params: {grid.best_params_}")
    logging.info(f"Best Cross-Validation Score: {grid.best_score_:.4f}")
    
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    logging.info(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    model_name = "sentiment_model_logistic.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # with open(model_path, 'wb') as f:
    #     pickle.dump(best_model, f)
        
    # vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    # with open(vec_path, 'wb') as f:
    #     pickle.dump(vectorizer, f)
        
    # logging.info(f"Model training completed.")
    # logging.info(f"Saved model to: {model_path}")
    
    return model_path

if __name__ == "__main__":
    sample_path = '/mnt/d/AOIVietNam/Project/BlogModule6W1_2/data/processed/processed_Amazon_Reviews_20251122_182324.csv'
    train_model(sample_path)