import glob
import os
import shutil
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from supabase import create_client, Client
import hashlib

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_PATH)
BASE_DIR = os.path.join(PROJECT_ROOT, 'data')

logging.info(f"Project Root detected at: {PROJECT_ROOT}")
logging.info(f"Data Directory set to: {BASE_DIR}")

INPUT_DIR = os.path.join(BASE_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")
UPLOAD_TO_CLOUD = os.getenv("UPLOAD_TO_CLOUD", "false").lower() == "true"

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
lemmatizer = WordNetLemmatizer()
sen = SentimentIntensityAnalyzer()

try:
    from constants import country_map
except ImportError:
    logging.warning("Warning: Cannot find countrymap file. Using empty map.")
    country_map = {}

def convert_country(code):
    if pd.isna(code) or code is None or str(code).strip().lower() in ["", "nan", "none"]:
        return "Do not mention"
    code = str(code).strip().upper()
    return country_map.get(code, "Do not mention")

def generate_id(row):
    content = f"{row['reviewer_name']}_{row['review_date']}_{row['review_title']}_{row['review_text']}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def connect_to_db():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.error("Supabase url and key not found....")
        return None
        
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("Connected to Supabase!")
        return supabase
    except Exception as e:
        logging.error(f"Error: Can not connect to db: {e}")
        return None
    
def prepare_dataframe_for_db(df):
    df_db = df.copy()
    
    df_db.columns = df_db.columns.str.lower().str.replace(' ', '_')
    
    df_db['reviewer_name'] = df_db['reviewer_name'].fillna('')
    df_db['review_title'] = df_db['review_title'].fillna('')
    df_db['review_text'] = df_db['review_text'].fillna('')
    
    cols = [
        'reviewer_name', 'country', 'review_count', 'review_date', 'rating',
        'review_title', 'review_text', 'date_of_experience',
        'full_review', 'polarity_score', 'sentiment_label', 'sentiment_text'
    ]
    df_db = df_db[[c for c in cols if c in df_db.columns]].copy()
    
    for date_col in ['review_date', 'date_of_experience']:
        if date_col in df_db.columns:
            df_db[date_col] = pd.to_datetime(df_db[date_col], errors='coerce')
            df_db[date_col] = df_db[date_col].dt.strftime('%Y-%m-%dT%H:%M:%S')
    logging.info("Generating unique IDs based on content hashing...")
    df_db['id'] = df_db.apply(generate_id, axis=1)
    df_db = df_db.where(pd.notna(df_db), None)
    
    return df_db

def upload_df_to_supabase(df, supabase: Client):
    if not supabase:
        logging.warning("Upload failed - no supabase connection")
        return False
    
    try:
        df_db = prepare_dataframe_for_db(df)
        
        data_list = df_db.to_dict(orient="records")
        
        batch_size = 1000
        logging.info(f"Uploading {len(data_list)} records to Supabase...")
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            try:
                supabase.table('amazon_reviews').upsert(batch, on_conflict='id').execute()
                logging.info(f"Uploaded batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as e:
                logging.error(f"Batch {i//batch_size + 1} failed: {e}")
                return False
                
        logging.info("Load data to supabase successfully...")
        return True
    except Exception as e:
        logging.error(f"Supabase upload error: {e}")
        return False
        

def get_file_from_kaggle(dataset_name, file_name, download_path):
    try:
        api = KaggleApi()
        api.authenticate()
        logging.info(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        logging.info(f"Successfully downloaded data to {download_path}")
        zip_files = glob.glob(os.path.join(download_path, "*.zip"))
        for z in zip_files:
            os.remove(z)
    except Exception as e:
        logging.error(f"Error downloading {file_name} from {dataset_name}: {e}")

def extract_data():
    raw_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if len(raw_files) == 0:
        archived_files = glob.glob(os.path.join(ARCHIVE_DIR, "*.csv"))
        
        if len(archived_files) > 0:
            logging.info(f"Found {len(archived_files)} archived files. Skipping download.")
            logging.info("Please clear the archive folder to redownload.")
            return []
        
        logging.info("No file found in Input or Archive. Downloading from Kaggle...")
        get_file_from_kaggle(
            dataset_name='dongrelaxman/amazon-reviews-dataset',
            file_name='Amazon_Reviews.csv',
            download_path=INPUT_DIR
        )
        raw_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if not raw_files:
        logging.warning("Failed to obtain data or directory is empty.")
        return []
    
    logging.info(f"Found {len(raw_files)} files to process.")
    return raw_files

def archive_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f"{timestamp}_{file_name}"
        destination = os.path.join(ARCHIVE_DIR, new_name)
        shutil.move(file_path, destination)
        logging.info(f"Archived raw file to: {destination}")
    except Exception as e:
        logging.error(f"Error archiving file {file_path}: {e}")

def clean_text_for_csv(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def transform_data(file_paths):
    if not file_paths:
        logging.info("No files to transform.")
        return []
    
    file_info_list = []

    for file_path in file_paths:
        logging.info(f"Processing: {file_path}")
        try:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip', encoding='latin1')
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue

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

        logging.info("Cleaning text fields for CSV compatibility...")
        df['Review Title'] = df['Review Title'].apply(clean_text_for_csv)
        df['Review Text'] = df['Review Text'].apply(clean_text_for_csv)

        logging.info("Concatenating Review Title and Review Text...")
        df['full_review'] = df['Review Title'].astype(str) + " " + df['Review Text'].astype(str)

        if "Date of Experience" in df.columns and "Review Date" in df.columns:
            df["Date of Experience"] = df["Date of Experience"].fillna(df["Review Date"])
            
        for col in ['Rating', 'Review Count']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
        
        for col in ['Review Date', 'Date of Experience']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        logging.info("Calculating VADER Polarity Score...")
        df["polarity_score"] = df["full_review"].apply(
            lambda x: sen.polarity_scores(x)["compound"]
        )

        logging.info("Generating Sentiment Labels...")
        df['sentiment_label'] = df['polarity_score'].apply(
            lambda score: 1 if score >= 0.05 else (0 if score <= -0.05 else -1)
        )

        df = df[df['sentiment_label'] != -1]

        df['sentiment_text'] = df['sentiment_label'].map({
            1: 'Positive',
            0: 'Negative'
        })

        df.reset_index(drop=True, inplace=True)

        base_name = os.path.basename(file_path).replace('.csv', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"processed_{base_name}_{timestamp}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            df.to_csv(
                output_path, 
                index=False,
                quoting=1, 
                escapechar='\\',
                encoding='utf-8'
            )
            
            logging.info(f"Saved processed data: {output_path}")
            logging.info(f"Total records: {len(df)}")
            logging.info(f"Sentiment distribution:\n{df['sentiment_text'].value_counts()}")
            logging.info(f"Polarity score range: {df['polarity_score'].min():.3f} to {df['polarity_score'].max():.3f}")
            file_info_list.append({
                'raw_file': file_path,
                'processed_file': output_path
            })
            
        except Exception as e:
            logging.error(f"Failed to save processed file for {file_path}: {e}")

    return file_info_list


def load_to_cloud(file_info_list):
    if not UPLOAD_TO_CLOUD:
        logging.info("Cloud upload disabled. Skipping...")
        return [info['raw_file'] for info in file_info_list]
    
    if not file_info_list:
        logging.info("No files to upload.")
        return []
    
    supabase = connect_to_db()
    if not supabase:
        raise Exception("Cannot connect to Supabase - upload failed")
    
    successful_raw_files = []
    
    for file_info in file_info_list:
        try:
            df = pd.read_csv(file_info['processed_file'])
            success = upload_df_to_supabase(df, supabase)
            
            if success:
                successful_raw_files.append(file_info['raw_file'])
                logging.info(f"Upload successful: {file_info['raw_file']}")
            else:
                logging.error(f"Upload failed: {file_info['raw_file']}")
                
        except Exception as e:
            logging.error(f"Error uploading {file_info['raw_file']}: {e}")

    if len(successful_raw_files) != len(file_info_list):
        raise Exception(f"Only {len(successful_raw_files)}/{len(file_info_list)} files uploaded successfully")
    
    logging.info("All files uploaded successfully!")
    return successful_raw_files


def archive_raw_files(raw_file_paths):
    if not raw_file_paths:
        logging.info("No files to archive")
        return
    
    for file_path in raw_file_paths:
        archive_file(file_path)
    
    logging.info(f"Archived {len(raw_file_paths)} raw files successfully")



if __name__ == "__main__":
    logging.info("Starting ETL + Feature Engineering Process...")
    
    files = extract_data()
    
    if files:
        file_info = transform_data(files)
        
        if file_info:
            try:
                successful_files = load_to_cloud(file_info)
                archive_raw_files(successful_files)
                logging.info("ETL Pipeline Complete!")
            except Exception as e:
                logging.error(f"Pipeline failed: {e}")
                logging.warning("Raw files NOT archived - can retry later")
    else:
        logging.info("No new files to process.")