from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.load_data import load_data
from src.feature_engineering import feature_engineering
from src.train_model import train_model
from src.evaluate_model import evaluate_model

# Default arguments for the DAG
default_args = {
    'owner': 'pranav',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'Housing_Price_Prediction_Pipeline',
    default_args=default_args,
    description='Real Estate Price Prediction using Random Forest',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
)

# Task 1: Load Data
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Feature Engineering
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

# Task 3: Train Model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Task 4: Evaluate Model
evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Define task dependencies (pipeline flow)
load_data_task >> feature_engineering_task >> train_model_task >> evaluate_model_task
