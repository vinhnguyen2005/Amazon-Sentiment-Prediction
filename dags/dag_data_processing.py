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
    dag_id='dag_data_processing',
    default_args=default_args,
    description='Preprocess data for training',
    schedule='@hourly',
    catchup=False
)
def data_processing_pipeline():

    @task
    def extract_task():
        import sys
        if PROJECT_PATH not in sys.path:
            sys.path.append(PROJECT_PATH)
        
        from scripts.etl import extract_data
        
        files = extract_data()
        if not files:
            logging.info("No files found.")
            raise AirflowSkipException("No data to process")
        return files

    @task
    def transform_task(raw_file_paths):
        import sys
        if PROJECT_PATH not in sys.path:
            sys.path.append(PROJECT_PATH)
            
        from scripts.etl import transform_data
        
        if not raw_file_paths:
            logging.info("Skipping transform as no files extracted.")
            return []
        return transform_data(raw_file_paths)

    @task(outlets=[AMAZON_REVIEWS_DB])
    def load_task(processed_file_info):
        import sys
        if PROJECT_PATH not in sys.path:
            sys.path.append(PROJECT_PATH)
            
        from scripts.etl import load_to_cloud
        
        if not processed_file_info:
            logging.info("Skipping load.")
            return []
        return load_to_cloud(processed_file_info)

    @task
    def archive_task(files_to_archive):
        import sys
        if PROJECT_PATH not in sys.path:
            sys.path.append(PROJECT_PATH)
            
        from scripts.etl import archive_raw_files
        
        if files_to_archive:
            archive_raw_files(files_to_archive)


    raw_files = extract_task()
    processed_files = transform_task(raw_files)
    uploaded_files = load_task(processed_files)
    archive_task(uploaded_files)


dag_instance = data_processing_pipeline()