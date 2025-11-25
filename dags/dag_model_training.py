from datetime import datetime, timedelta
import logging       
from airflow.sdk import task, dag
from airflow.exceptions import AirflowSkipException
from pytz import timezone
from airflow import Dataset
local_tz = timezone('Asia/Ho_Chi_Minh')

AMAZON_REVIEWS_DB = Dataset("supabase://amazon_reviews")

default_args = {
    'owner': 'cris',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 23, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

PROJECT_PATH = "/mnt/d/AOIVietNam/Project/BlogModule6W1_2"

@dag(
    dag_id='dag_model_training',
    default_args=default_args,
    description='Preprocess data for training',
    schedule=[AMAZON_REVIEWS_DB],
    catchup=False
)
def model_training_pipeline():
    
    @task
    def train_model_task():
        import sys
        if PROJECT_PATH not in sys.path:
            sys.path.append(PROJECT_PATH)
        from scripts.train_model import train_model
        model_path = train_model()
        logging.info(f"New model trained and saved at: {model_path}")
        
    train_model_task()
    
dag_instance = model_training_pipeline()
        