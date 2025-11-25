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
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_PATH)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
METADATA_FILE = os.path.join(DATA_DIR, 'metadata', 'kaggle_dataset_processed.json')
os.makedirs(MODEL_DIR, exist_ok=True)

load_dotenv()
SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")

def load_db():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")    
    return create_client(url, key)

def get_files_from_metadata():

    if not os.path.exists(METADATA_FILE):
        logging.warning(f"Metadata file not found at {METADATA_FILE}")
        return None
    
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        latest_files = metadata.get('files', [])
        
        if not latest_files:
            logging.warning("Metadata found but 'files' list is empty.")
            return None
            
        logging.info(f"Metadata indicates {len(latest_files)} files processed at {metadata.get('processed_at')}")
        return latest_files
        
    except Exception as e:
        logging.error(f"Error reading metadata: {e}")
        return None 

def load_data():
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
        
    if df is None:
        logging.info("Checking metadata for latest processed files...")
        
        target_files = get_files_from_metadata() 
        
        if target_files:
            data_frames = []
            for file_path in target_files:
                if os.path.exists(file_path):
                    logging.info(f"Reading file: {file_path}")
                    try:
                        temp_df = pd.read_csv(file_path)
                        data_frames.append(temp_df)
                    except Exception as e:
                        logging.error(f"Failed to read {file_path}: {e}")
                else:
                    logging.warning(f"File listed in metadata but not found on disk: {file_path}")
            
            if data_frames:
                df = pd.concat(data_frames, ignore_index=True)
                logging.info(f"Successfully loaded {len(df)} records from metadata files.")
            else:
                logging.error("No valid files could be loaded from metadata list.")

    return df


def train_model():
    df = load_data()
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
    
    model_path = os.path.join(MODEL_DIR, "sentiment_pipeline_logistic.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    return model_path
