# Amazon Reviews Sentiment Analysis Pipeline

<img width="1851" height="879" alt="Screenshot 2025-11-25 153453" src="https://github.com/user-attachments/assets/250ba6bd-66f2-428c-8738-7a9ca46cbf17" />
<img width="1840" height="864" alt="Screenshot 2025-11-25 153524" src="https://github.com/user-attachments/assets/71126cdc-5d7b-41d3-a603-a3a1b91da0fb" />


An automated ETL and Machine Learning pipeline for sentiment analysis of Amazon Reviews using Apache Airflow and Supabase.

## 📋 Overview

This project builds a complete pipeline including:

- **ETL Pipeline**: Automatically download, transform, and load Amazon Reviews data
- **ML Training Pipeline**: Auto-train Logistic Regression model when new data arrives
- **Sentiment Analysis**: Classify reviews into Positive/Negative with confidence scores
- **Data Orchestration**: Use Apache Airflow to manage and schedule workflows

## 🏗️ Architecture

```
┌─────────────────┐
│  Kaggle API     │
│  (Data Source)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   DAG: Data Processing          │
│   Schedule: @hourly             │
├─────────────────────────────────┤
│  1. Extract (Kaggle)            │
│  2. Transform (Clean + Feature) │
│  3. Load (Supabase)             │
│  4. Archive                     │
└────────┬────────────────────────┘
         │ (Dataset Trigger)
         ▼
┌─────────────────────────────────┐
│   DAG: Model Training           │
│   Schedule: Dataset-based       │
├─────────────────────────────────┤
│  1. Load from Supabase          │
│  2. Clean & Prepare             │
│  3. Train (GridSearchCV)        │
│  4. Save Model (.pkl)           │
└─────────────────────────────────┘
```

## 🚀 Key Features

### 1. ETL Pipeline
- **Extract**: Automatically download dataset from Kaggle
- **Transform**: 
  - Data cleaning and deduplication
  - Feature engineering (sentiment scores, country mapping, Vader)
  - Text preprocessing
- **Load**: Batch upload data to Supabase
- **Archive**: Store raw files after successful processing

### 2. Machine Learning
- **Model**: Logistic Regression with TF-IDF vectorization
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Features**: Unigrams, Bigrams, Trigrams (max 10,000 features)
- **Performance**: Accuracy ~85-90% on test set

### 3. Airflow Orchestration
- **Dataset-aware Scheduling**: Model training only runs when new data is available
- **Error Handling**: Retry logic and skip mechanism
- **Monitoring**: Comprehensive logging

## 📁 Project Structure

```
BlogModule6W1_2/
├── dags/
│   ├── dag_data_processing.py    # ETL workflow
│   └── dag_model_training.py     # ML training workflow
├── scripts/
│   ├── etl.py                    # ETL logic
│   ├── train_model.py            # Model training logic
│   ├── test.py                   # Model testing script
│   └── constants.py              # Country code mapping
├── data/
│   ├── raw/                      # Raw CSV files
│   ├── processed/                # Cleaned CSV files
│   ├── archive/                  # Archived files
│   └── metadata/                 # Processing metadata
├── models/
│   └── sentiment_pipeline_logistic.pkl  # Trained model
└── logs/                         # Airflow logs
```

## 🛠️ Installation

### System Requirements
- Python 
- Apache Airflow 
- Supabase account
- Kaggle API credentials

### Step 1: Clone Repository

```bash
git clone https://github.com/vinhnguyen2005/Amazon-Sentiment-Prediction.git
cd BlogModule6W1_2
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create a `.env` file in the root directory:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
UPLOAD_TO_CLOUD=true
```

### Step 4: Configure Kaggle API

Create `~/.kaggle/kaggle.json`:

```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Step 5: Configure Supabase

Create `amazon_reviews` table with schema:

```sql
CREATE TABLE amazon_reviews (
  id TEXT PRIMARY KEY,
  reviewer_name TEXT,
  country TEXT,
  review_count INTEGER,
  review_date TIMESTAMP,
  rating INTEGER,
  review_title TEXT,
  review_text TEXT,
  date_of_experience TIMESTAMP,
  full_review TEXT,
  polarity_score FLOAT,
  sentiment_label INTEGER,
  sentiment_text TEXT
);
```

### Step 6: Start Airflow

```bash
# Initialize Airflow database
airflow db migrate

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start webserver
airflow api-server 

# Start scheduler (in new terminal)
airflow scheduler
```
or just use

```bash
airflow standalone
```


### Step 7: Access Airflow UI

Open browser: `http://localhost:8080`

- Username: `admin`
- Password: `admin`

## 📊 Usage

### 1. Run ETL Pipeline

In Airflow UI:
1. Find DAG `dag_data_processing`
2. Enable the DAG
3. Trigger manually or wait for schedule (@hourly)

The pipeline will:
- Download Amazon Reviews from Kaggle
- Clean and transform data
- Upload to Supabase
- Archive raw files

### 2. Model Training (Automatic)

When ETL pipeline completes, `dag_model_training` will automatically trigger:
- Load 50,000 records from Supabase
- Filter high-quality data
- Train model with GridSearchCV
- Save model to `models/sentiment_pipeline_logistic.pkl`

### 3. Test Model

```bash
python scripts/test.py
```

**Output:**

```
================================================================================
SENTIMENT ANALYSIS TEST RESULTS
================================================================================

Review 1:
Text: This product is absolutely amazing! Best purchase ever!
Prediction: POSITIVE (Confidence: 95.23%)
--------------------------------------------------------------------------------
Review 2:
Text: Terrible quality, waste of money. Very disappointed.
Prediction: NEGATIVE (Confidence: 92.45%)
...

================================================================================
INTERACTIVE MODE - Enter your own reviews (type 'quit' to exit)
================================================================================

Enter a review to analyze: 
```

## 🔍 Technical Details

### Data Processing

**Cleaning Steps:**
- Remove duplicates (by reviewer, date, text)
- Drop missing reviews
- Normalize country codes (150+ countries)
- Clean text (remove newlines, extra spaces)

**Feature Engineering:**
- `full_review`: Concatenation of title + text
- `polarity_score`: VADER sentiment score (-1 to +1)
- `sentiment_label`: 1 (Positive), 0 (Negative), -1 (Neutral - removed)
- `id`: SHA256 hash of key fields

### Model Training

**Data Quality Filter:**
```python
# Only use consistent reviews
positive = (rating >= 4) & (polarity_score > 0.1)
negative = (rating <= 2) & (polarity_score < -0.1)
```

**Model Pipeline:**
```python
Pipeline([
  TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=10000,
    stop_words='english'
  ),
  LogisticRegression(
    C=[0.01, 0.1, 0.5, 1, 5],  # GridSearch
    class_weight='balanced'
  )
])
```

**Performance Metrics:**
- Accuracy: ~97%
- Precision/Recall: Balanced with class_weight
- Cross-validation: 5-fold

### Airflow Dataset Scheduling

```python
# DAG 1: Creates dataset update
@task(outlets=[AMAZON_REVIEWS_DB])
def load_task():
    # Upload to Supabase
    # Triggers dataset update on success

# DAG 2: Listens to dataset
@dag(schedule=[AMAZON_REVIEWS_DB])
def model_training_pipeline():
    # Runs automatically when data is updated
```


## 📈 Monitoring

### View Logs in Airflow UI

1. DAG View → Click on task
2. "Logs" tab for details

### Check Metadata

```bash
cat data/metadata/kaggle_dataset_processed.json
```

```json
{
  "processed_at": "2025-11-25T14:50:33.123456",
  "total_records": 45230,
  "files": ["data/processed/processed_Amazon_Reviews_20251125_145033.csv"]
}
```


## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Dataset: [Amazon Reviews Dataset](https://www.kaggle.com/dongrelaxman/amazon-reviews-dataset) on Kaggle
- Sentiment Analysis: NLTK VADER
- Orchestration: Apache Airflow
- Database: Supabase

