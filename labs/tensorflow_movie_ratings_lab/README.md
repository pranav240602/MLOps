# Movie Ratings Data Quality & Anomaly Detection Lab

## Overview
This lab demonstrates MLOps data validation and monitoring concepts using the MovieLens dataset. Unlike traditional machine learning labs that focus solely on model training, this lab emphasizes production-ready data quality monitoring, which is critical for maintaining reliable ML systems.

## Lab Objectives
- Perform comprehensive data quality validation
- Detect anomalous user behavior using Machine Learning
- Monitor data drift between training and production data
- Generate automated data quality reports
- Demonstrate MLOps best practices for data monitoring

## Project Structure
```
tensorflow_movie_ratings_lab/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/
│   └── ml-latest-small/               # MovieLens dataset
│       ├── ratings.csv                # User ratings data
│       ├── movies.csv                 # Movie metadata
│       └── ...
├── notebooks/
│   └── movie_ratings_analysis.ipynb   # Main analysis notebook
└── outputs/
    ├── user_statistics_with_anomalies.csv
    ├── anomalous_users.csv
    ├── drift_analysis.csv
    └── summary_statistics.csv
```

## Dataset
**MovieLens Latest Small Dataset**
- 100,000+ movie ratings
- 600+ users
- 9,000+ movies
- Ratings from 0.5 to 5.0
- Source: GroupLens Research (https://grouplens.org/datasets/movielens/)

## Setup Instructions

### 1. Clone/Navigate to Project
```bash
cd ~/Documents/mlops/labs/tensorflow_movie_ratings_lab
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
```bash
cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
cd ..
```

### 5. Run the Notebook
```bash
jupyter notebook
```
Navigate to notebooks/movie_ratings_analysis.ipynb and run all cells.

## What This Lab Does

### 1. Data Quality Validation
- Checks for missing values
- Validates data types and ranges
- Ensures data completeness
- Visualizes data distribution patterns

### 2. Anomaly Detection
- Uses Isolation Forest algorithm to detect unusual user behavior
- Identifies users with suspicious rating patterns
- Flags potential bot accounts or fake reviews
- Visualizes normal vs anomalous users

**Key Features Analyzed:**
- Number of ratings per user
- Average rating per user
- Rating variance (standard deviation)

### 3. Data Drift Detection
- Splits data into "training" (historical) and "production" (recent) sets
- Compares statistical distributions over time
- Detects shifts in user behavior
- Alerts when retraining might be needed

**Metrics Monitored:**
- Mean rating changes
- Standard deviation changes
- Distribution shifts
- Temporal trends

### 4. Automated Reporting
- Generates comprehensive data quality reports
- Provides actionable recommendations
- Exports results to CSV files for further analysis

## Key Results

### Data Quality Metrics
- Zero missing values across all datasets
- Complete data integrity maintained
- All rating values within valid range (0.5 - 5.0)

### Anomaly Detection Results
- Detected approximately 5% of users showing anomalous behavior
- Anomalous patterns include:
  - Extremely high or low average ratings
  - Unusual rating frequency
  - Inconsistent rating patterns

### Data Drift Analysis
- Compared historical vs recent user behavior
- Monitored rating distribution changes over time
- Provided alerts for significant drift

## Technologies Used
- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning (Isolation Forest, StandardScaler)
- **Jupyter Notebook** - Interactive development environment

## Key Differences from Professor's Lab
1. **Different Dataset**: Uses MovieLens movie ratings instead of typical ML datasets
2. **Different Focus**: Emphasizes data quality and monitoring over model training
3. **Additional Features**: 
   - Anomaly detection using Isolation Forest
   - Data drift monitoring
   - Automated quality reporting
4. **Production Perspective**: Demonstrates real-world MLOps data monitoring practices

## MLOps Concepts Demonstrated
- **Data Validation**: Ensuring data quality before model training
- **Anomaly Detection**: Identifying unusual patterns in production data
- **Data Drift Monitoring**: Detecting changes in data distribution over time
- **Automated Reporting**: Creating reproducible quality reports
- **Version Control Ready**: Structured for Git integration

## Output Files
After running the notebook, the following files are generated in the outputs/ directory:

1. **user_statistics_with_anomalies.csv** - User behavior metrics with anomaly flags
2. **anomalous_users.csv** - List of users with suspicious patterns
3. **drift_analysis.csv** - Statistical comparison of training vs production data
4. **summary_statistics.csv** - Overall dataset statistics

## Future Enhancements
- Integration with MLflow for experiment tracking
- Real-time monitoring dashboard
- Automated alerting system for drift detection
- Integration with CI/CD pipeline
- Model training based on validated data

## Author
Pranav Rangbulla
Data Analytics Engineering Graduate Student
Northeastern University
Course: IE 7374 MLOps

## License
This project is for educational purposes as part of the MLOps course at Northeastern University.

## Acknowledgments
- Professor Ramin Mohammadi for course guidance
- GroupLens Research for the MovieLens dataset
- Northeastern University MLOps course materials
