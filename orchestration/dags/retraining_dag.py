from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'recommendation_feature_pipeline',
    default_args=default_args,
    description='Batch feature computation and model retraining DAG for Universal Recommender',
    schedule_interval=timedelta(days=1),
)

# 1. Trigger Feast to materialize offline features to online Redis store
materialize_features = BashOperator(
    task_id='materialize_feast_features',
    bash_command='cd /opt/airflow/feature_store && feast materialize $(date -u -d "-1 day" +"%Y-%m-%dT%H:%M:%S") $(date -u +"%Y-%m-%dT%H:%M:%S")',
    dag=dag,
)

# 2. Retrain Two-Tower Embeddings and update FAISS index
retrain_candidate_generator = BashOperator(
    task_id='retrain_candidate_generator',
    bash_command='python /opt/airflow/models/two_tower.py && python /opt/airflow/models/vector_search.py',
    dag=dag,
)

# 3. Retrain XGBoost Ranking Model
retrain_ranker = BashOperator(
    task_id='retrain_xgb_ranker',
    bash_command='python /opt/airflow/models/ranking.py --train',
    dag=dag,
)

materialize_features >> retrain_candidate_generator >> retrain_ranker
