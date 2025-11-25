import glob
import os
import shutil
import sys
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from datetime import datetime
import re
import hashlib
import json
import time
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_PATH)
BASE_DIR = os.path.join(PROJECT_ROOT, 'data')

INPUT_DIR = os.path.join(BASE_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')

for directory in [INPUT_DIR, OUTPUT_DIR, ARCHIVE_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
UPLOAD_TO_CLOUD = os.getenv("UPLOAD_TO_CLOUD", "false").lower() == "true"

BATCH_SIZE = 500
PROCESSED_MARKER = os.path.join(METADATA_DIR, 'kaggle_dataset_processed.json')

def download_nltk_resources():
    resources = ['vader_lexicon', 'stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}') if res == 'punkt' else nltk.data.find(f'corpora/{res}')
        except LookupError:
            logging.info(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

download_nltk_resources()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

try:
    if CURRENT_SCRIPT_PATH not in sys.path:
        sys.path.append(CURRENT_SCRIPT_PATH)
    from constants import country_map
except ImportError:
    logging.warning("Cannot find country_map. Using empty map.")
    country_map = {}

def convert_country(code):
    if pd.isna(code) or code is None or str(code).strip().lower() in ["", "nan", "none"]:
        return "Do not mention"
    code = str(code).strip().upper()
    return country_map.get(code, "Do not mention")

def generate_review_id(row):
    key_components = [
        str(row.get('Reviewer Name', '')),
        str(row.get('Rating', '')),
        str(row.get('Review Date', ''))[:10],
        str(row.get('Review Title', ''))[:50]
    ]
    composite_key = '|'.join(key_components)
    return hashlib.sha256(composite_key.encode('utf-8')).hexdigest()

def connect_to_db():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.error("Supabase credentials not found")
        return None
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("Connected to Supabase!")
        return supabase
    except Exception as e:
        logging.error(f"Cannot connect to db: {e}")
        return None

def is_dataset_already_processed():
    if os.path.exists(PROCESSED_MARKER):
        with open(PROCESSED_MARKER, 'r') as f:
            data = json.load(f)
            logging.info(f"Dataset already processed on {data['processed_at']}")
            return True
    return False

def mark_dataset_as_processed(records_count, file_info_list):
    file_paths = [info['processed_file'] for info in file_info_list]
    with open(PROCESSED_MARKER, 'w') as f:
        json.dump({
            'processed_at': datetime.now().isoformat(),
            'total_records': records_count,
            'files' : file_paths
        }, f, indent=2)
    logging.info(f"Marked dataset as processed ({records_count} records)")

def get_file_from_kaggle(dataset_name, download_path):
    try:
        api = KaggleApi()
        api.authenticate()
        logging.info(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        logging.info(f"Successfully downloaded to {download_path}")
        
        for z in glob.glob(os.path.join(download_path, "*.zip")):
            os.remove(z)
    except Exception as e:
        logging.error(f"Error downloading: {e}")

def extract_data():
    if is_dataset_already_processed():
        logging.info("Dataset already processed. Skipping extraction.")
        from airflow.exceptions import AirflowSkipException
        raise AirflowSkipException("Dataset already processed")
    
    raw_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if raw_files:
        logging.info(f"Found {len(raw_files)} files in INPUT_DIR")
        return raw_files
    
    logging.info("No files found. Downloading from Kaggle...")
    get_file_from_kaggle(
        dataset_name='dongrelaxman/amazon-reviews-dataset',
        download_path=INPUT_DIR
    )
    
    raw_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not raw_files:
        logging.warning("Failed to obtain data")
        return []
    
    logging.info(f"Downloaded {len(raw_files)} files")
    return raw_files

def archive_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f"{timestamp}_{file_name}"
        destination = os.path.join(ARCHIVE_DIR, new_name)
        shutil.move(file_path, destination)
        logging.info(f"Archived: {new_name}")
    except Exception as e:
        logging.error(f"Error archiving {file_path}: {e}")

def clean_text_for_csv(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def transform_data(file_paths):
    if not file_paths:
        logging.info("No files to transform")
        return []
    
    file_info_list = []

    for file_path in file_paths:
        logging.info(f"Processing: {file_path}")
        try:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip', encoding='latin1')
            initial_count = len(df)
            
            col_name = None
            if 'Country' in df.columns:
                col_name = 'Country'
            elif 'country' in df.columns:
                col_name = 'country'
            
            if col_name:
                df['Country'] = df[col_name].apply(convert_country)
                if col_name == 'country':
                    df = df.drop(columns=['country'])
            
            df = df.drop(columns=["Profile Link"], errors='ignore')
            df = df.dropna(subset=["Review Title", "Review Text"])
            
            df = df.drop_duplicates(subset=['Reviewer Name', 'Review Date', 'Review Text'], keep='first')

            df['Review Title'] = df['Review Title'].apply(clean_text_for_csv)
            df['Review Text'] = df['Review Text'].apply(clean_text_for_csv)
            df['full_review'] = df['Review Title'].astype(str) + " " + df['Review Text'].astype(str)

            if "Date of Experience" in df.columns and "Review Date" in df.columns:
                df["Date of Experience"] = df["Date of Experience"].fillna(df["Review Date"])
            
            for col in ['Rating', 'Review Count']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
            
            for col in ['Review Date', 'Date of Experience']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            logging.info("Running sentiment analysis...")
            df["polarity_score"] = df["full_review"].apply(lambda x: sia.polarity_scores(x)["compound"])
            df['sentiment_label'] = df['polarity_score'].apply(
                lambda score: 1 if score >= 0.05 else (0 if score <= -0.05 else -1)
            )
            df = df[df['sentiment_label'] != -1]
            df['sentiment_text'] = df['sentiment_label'].map({1: 'Positive', 0: 'Negative'})

            logging.info("Generating unique IDs...")
            df['id'] = df.apply(generate_review_id, axis=1)
            df = df.drop_duplicates(subset=['id'], keep='first')
            
            df.reset_index(drop=True, inplace=True)

            final_count = len(df)
            logging.info(f"Records: {initial_count} → {final_count} (removed {initial_count - final_count})")
            
            base_name = os.path.basename(file_path).replace('.csv', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"processed_{base_name}_{timestamp}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            df.to_csv(output_path, index=False, quoting=1, escapechar='\\', encoding='utf-8')
            
            logging.info(f"Saved: {output_path}")
            logging.info(f"Sentiment: {df['sentiment_text'].value_counts().to_dict()}")
            
            file_info_list.append({
                'raw_file': file_path,
                'processed_file': output_path,
                'records': final_count
            })
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    return file_info_list

def prepare_dataframe_for_db(df):
    df_db = df.copy()
    df_db.columns = df_db.columns.str.lower().str.replace(' ', '_')
    
    df_db['reviewer_name'] = df_db['reviewer_name'].fillna('')
    df_db['review_title'] = df_db['review_title'].fillna('')
    df_db['review_text'] = df_db['review_text'].fillna('')
    
    cols = [
        'id', 'reviewer_name', 'country', 'review_count', 'review_date',
        'rating', 'review_title', 'review_text', 'date_of_experience',
        'full_review', 'polarity_score', 'sentiment_label', 'sentiment_text'
    ]
    df_db = df_db[[c for c in cols if c in df_db.columns]].copy()
    
    for date_col in ['review_date', 'date_of_experience']:
        if date_col in df_db.columns:
            df_db[date_col] = pd.to_datetime(df_db[date_col], errors='coerce')
            df_db[date_col] = df_db[date_col].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    df_db = df_db.where(pd.notna(df_db), None)
    return df_db

def upload_df_to_supabase(df, supabase):
    if not supabase:
        logging.warning("No Supabase connection")
        return False
    
    try:
        df_db = prepare_dataframe_for_db(df)
        data_list = df_db.to_dict(orient="records")
        
        logging.info(f"Uploading {len(data_list)} records...")
        
        for i in range(0, len(data_list), BATCH_SIZE):
            batch = data_list[i:i+BATCH_SIZE]
            for attempt in range(3):
                try:
                    supabase.table('amazon_reviews').upsert(batch, on_conflict='id').execute()
                    logging.info(f"Batch {i//BATCH_SIZE + 1}: {len(batch)} records uploaded")
                    break
                except Exception as e:
                    logging.error(f"Batch {i//BATCH_SIZE + 1} attempt {attempt+1} failed: {e}")
                    time.sleep(3)
            else:
                return False
        logging.info("Upload successful!")
        return True
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return False

def load_to_cloud(file_info_list):
    if not UPLOAD_TO_CLOUD:
        logging.info("Cloud upload disabled")
        return [info['raw_file'] for info in file_info_list]
    
    if not file_info_list:
        logging.info("No files to upload")
        return []
    
    supabase = connect_to_db()
    if not supabase:
        raise Exception("Cannot connect to Supabase")
    
    successful_raw_files = []
    total_records = 0
    
    for file_info in file_info_list:
        try:
            df = pd.read_csv(file_info['processed_file'])
            success = upload_df_to_supabase(df, supabase)
            
            if success:
                successful_raw_files.append(file_info['raw_file'])
                total_records += file_info['records']
                logging.info(f"Success: {file_info['raw_file']}")
            else:
                logging.error(f"Fail: {file_info['raw_file']}")
                
        except Exception as e:
            logging.error(f"Error: {e}")

    if successful_raw_files:
        mark_dataset_as_processed(total_records, file_info_list)
    
    if len(successful_raw_files) != len(file_info_list):
        raise Exception(f"Only {len(successful_raw_files)}/{len(file_info_list)} uploaded")
    
    logging.info("All files uploaded!")
    return successful_raw_files

def archive_raw_files(raw_file_paths):
    if not raw_file_paths:
        logging.info("No files to archive")
        return
    
    for file_path in raw_file_paths:
        archive_file(file_path)
    
    logging.info(f"Archived {len(raw_file_paths)} files")
    
    
# if __name__ == "__main__":
#     logging.info("Starting ETL + Feature Engineering Process...")
    
#     files = extract_data()
    
#     if files:
#         file_info = transform_data(files)
        
#         if file_info:
#             try:
#                 successful_files = load_to_cloud(file_info)
#                 archive_raw_files(successful_files)
#                 logging.info("ETL Pipeline Complete!")
#             except Exception as e:
#                 logging.error(f"Pipeline failed: {e}")
#                 logging.warning("Raw files NOT archived - can retry later")
#     else:
#         logging.info("No new files to process.")