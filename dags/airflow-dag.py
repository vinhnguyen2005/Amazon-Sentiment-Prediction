from datetime import datetime, timedelta
from airflow.sdk import DAG, task
from pytz import timezone
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

local_tz = timezone('Asia/Ho_Chi_Minh')

try:
    from etl import extract_data, transform_data
    from feature_engineering import create_features
    from train_model import train_model
except ImportError as e:
    logging.error(f"Import Error: {e}")
    def extract_data(): pass
    def transform_data(files): pass
    def create_features(file): pass
    def train_model(file): pass

default_args = {
    'owner': 'cris',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 21, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}


with DAG(
    dag_id='amazon_sentiment_pipeline_final', 
    default_args=default_args,
    description='End-to-End ML Pipeline: ETL -> Feature Eng -> Training',
    schedule=None,        
    start_date=datetime(2025, 11, 22),
    catchup=False,                 
    render_template_as_native_obj=True, 
    tags=['mlops', 'amazon', 'sentiment']
) as dag:
    
    @task
    def extract_data():
        logging.info("Bắt đầu extract data...")
        file_paths = extract_data()
        logging.info(f"Extract hoàn tất: {file_paths}")
        return file_paths