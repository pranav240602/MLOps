# Airflow Lab - Housing Price Prediction Pipeline

**Course:** IE 7374 - MLOps  
**Student:** Pranav Rangbulla  
**Instructor:** Professor Ramin Mohammadi  

## Project Overview

This project implements an **Apache Airflow pipeline** for predicting housing prices using a **Random Forest Regressor**. The pipeline automates the complete machine learning workflow from data loading to model evaluation.

## Pipeline Architecture

The DAG consists of 4 sequential tasks:

1. **load_data**: Loads the California housing dataset
2. **feature_engineering**: Creates new features and handles missing values
3. **train_model**: Trains a Random Forest model
4. **evaluate_model**: Evaluates model performance with metrics
```
load_data → feature_engineering → train_model → evaluate_model
```

## Project Structure
```
airflow_lab/
├── dags/
│   ├── housing_price_dag.py       # Main DAG definition
│   ├── data/
│   │   ├── housing.csv            # Input dataset
│   │   ├── processed_housing.csv  # Processed data
│   │   └── predictions.csv        # Model predictions
│   ├── model/
│   │   └── rf_model.pkl           # Trained model
│   └── src/
│       ├── __init__.py
│       ├── load_data.py           # Data loading
│       ├── feature_engineering.py # Feature creation
│       ├── train_model.py         # Model training
│       └── evaluate_model.py      # Model evaluation
├── config/
├── logs/
├── plugins/
├── docker-compose.yaml
└── README.md
```

## Technologies Used

- **Apache Airflow 2.7.1**: Workflow orchestration
- **Docker & Docker Compose**: Containerization
- **scikit-learn**: Machine learning
- **pandas**: Data manipulation
- **Python 3.8+**: Programming language

## Setup Instructions

### Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM allocated to Docker

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/pranav240602/mlops.git
cd mlops/labs/airflow_lab
```

2. **Start Airflow with Docker Compose**
```bash
docker-compose up -d
```

3. **Wait for all containers to be healthy** (2-3 minutes)

4. **Access Airflow UI**
- URL: `http://localhost:8081`
- Username: `admin`
- Password: `admin`

5. **Run the DAG**
- Find `Housing_Price_Prediction_Pipeline` in the DAG list
- Toggle to unpause the DAG
- Click the play button to trigger the pipeline

## DAG Configuration
```python
default_args = {
    'owner': 'pranav',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
```

- **Schedule**: Manual trigger only
- **Catchup**: Disabled

## Feature Engineering

The pipeline creates the following engineered features:
- `rooms_per_household`: Total rooms divided by households
- `bedrooms_per_room`: Total bedrooms divided by total rooms
- `population_per_household`: Population divided by households

One-hot encoding is applied to the `ocean_proximity` categorical variable.

## Model Details

**Algorithm:** Random Forest Regressor

**Hyperparameters:**
- `n_estimators`: 100
- `max_depth`: 15
- `random_state`: 42
- `n_jobs`: -1

**Train/Test Split:** 80/20

## Results

The model is evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

Results are printed in the Airflow logs and predictions are saved to `predictions.csv`.

## Key Differences from Professor's Lab

This implementation differs from Professor Mohammadi's K-Means clustering lab by:
- Using **supervised learning** (Random Forest) instead of unsupervised (K-Means)
- Predicting **continuous values** (house prices) instead of clustering
- Working with **California housing dataset**
- Implementing **feature engineering** as a separate task
- Using **regression metrics** instead of clustering metrics

## Troubleshooting

**If you see import errors:**
- Ensure `scikit-learn`, `pandas`, and `numpy` are in `_PIP_ADDITIONAL_REQUIREMENTS` in docker-compose.yaml

**If port 8080 is already in use:**
- Change the webserver port mapping in docker-compose.yaml to `8081:8080`

**To restart Airflow:**
```bash
docker-compose restart
```

**To stop Airflow:**
```bash
docker-compose down
```

## Screenshots

All tasks completed successfully:

![Airflow DAG Success](screenshots/dag_success.png)

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Professor Ramin Mohammadi's MLOps Repository](https://github.com/raminmohammadi/MLOps)

---

**Date Submitted:** October 20, 2025
