# DVC Pipeline Lab: Stock Market Data Versioning & ML Pipeline

## Overview
This lab demonstrates advanced Data Version Control (DVC) concepts by building a complete ML pipeline for stock market data analysis. Unlike basic DVC tutorials that focus only on tracking files, this lab covers:

- **DVC Pipelines** with multiple stages and dependencies
- **Experiment tracking** with DVC experiments
- **Data metrics** and quality monitoring
- **AWS S3** integration for remote storage
- **Multiple data files** with complex dependencies
- **Feature engineering** pipeline for time-series data

## What Makes This Lab Different?
- **Production-ready pipeline**: Uses `dvc.yaml` with automated stages
- **Real-world data**: Works with live stock market data from multiple companies
- **Feature engineering**: Creates technical indicators (Moving Averages, RSI, MACD)
- **Experiment tracking**: Track different preprocessing approaches
- **Data quality metrics**: Monitor data statistics across versions

## Prerequisites
- Python 3.8+
- Git installed and configured
- AWS account (for S3 remote storage)
- Basic understanding of:
  - Git version control
  - Command line operations
  - Python programming

## Project Structure
```
dvc_pipeline_lab/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── params.yaml                  # Configuration parameters
├── dvc.yaml                     # DVC pipeline definition
├── .dvcignore                   # Files to ignore in DVC
├── .gitignore                   # Files to ignore in Git
├── data/
│   ├── raw/                     # Downloaded stock data
│   ├── processed/               # Cleaned data
│   └── features/                # Engineered features
├── metrics/                     # Data quality metrics
└── src/
    ├── download_data.py         # Download stock data
    ├── preprocess.py            # Data cleaning
    ├── feature_engineering.py   # Create technical indicators
    └── split_data.py            # Train/test split
```

## Setup Instructions

### 1. Install Dependencies
```bash
cd labs/dvc_pipeline_lab
pip install -r requirements.txt
```

### 2. Initialize Git and DVC
```bash
# Initialize Git (if not already done)
git init

# Initialize DVC
dvc init
```

### 3. AWS S3 Setup (Brief)
**Note**: You'll need AWS credentials to complete this section. If you don't have access yet, you can skip remote storage setup and work locally.

To configure AWS S3 as your remote storage:

1. Create an S3 bucket in your AWS console (e.g., `mlops-dvc-lab-<your-name>`)
2. Configure DVC to use S3:
```bash
dvc remote add -d myremote s3://mlops-dvc-lab-<your-name>/dvc-storage
dvc remote modify myremote region us-east-1
```
3. Configure AWS credentials (one of these methods):
   - AWS CLI: `aws configure`
   - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - AWS credentials file: `~/.aws/credentials`

### 4. Configure Parameters
Edit `params.yaml` to customize your pipeline:
```yaml
download:
  tickers:
    - AAPL
    - GOOGL
    - MSFT
    - TSLA
  start_date: "2020-01-01"
  end_date: "2024-12-31"
```

## Lab Exercises

### Exercise 1: Download and Track Stock Data

**Objective**: Download stock data and track it with DVC.
```bash
# Run the download stage
dvc repro download

# Track the raw data
dvc add data/raw/

# Commit the .dvc file
git add data/raw/.gitignore data/.gitignore
git commit -m "Add raw stock data tracking"

# Push to remote storage (if configured)
dvc push
```

**Expected Output**: Raw CSV files for each stock in `data/raw/`

### Exercise 2: Run the Complete Pipeline

**Objective**: Execute all pipeline stages and understand dependencies.
```bash
# Run the entire pipeline
dvc repro

# This will execute:
# 1. download - Get stock data
# 2. preprocess - Clean and prepare data
# 3. feature_engineering - Create technical indicators
# 4. split - Create train/test sets
```

**Expected Output**:
- `data/processed/stock_data_cleaned.csv`
- `data/features/stock_data_features.csv`
- `data/features/train.csv` and `data/features/test.csv`
- `metrics/data_quality.json`

### Exercise 3: Experiment with Different Parameters

**Objective**: Use DVC experiments to test different preprocessing approaches.
```bash
# Experiment 1: Change moving average window
dvc exp run --set-param feature_engineering.ma_window_short=20

# Experiment 2: Change RSI period
dvc exp run --set-param feature_engineering.rsi_period=10

# Show all experiments
dvc exp show

# Compare metrics across experiments
dvc metrics show
```

### Exercise 4: Modify Data and Track Changes

**Objective**: Update data and understand how DVC tracks changes.
```bash
# Modify params.yaml to add a new stock ticker
# Add "AMZN" to the tickers list

# Run pipeline with new data
dvc repro

# See what changed
dvc status

# Commit changes
git add dvc.yaml dvc.lock params.yaml
git commit -m "Add Amazon stock data"

# Push new data version
dvc push
```

### Exercise 5: Revert to Previous Data Version

**Objective**: Practice checking out previous data versions.
```bash
# See Git history
git log --oneline

# Checkout a previous commit
git checkout <commit-hash>

# Get the data version from that commit
dvc checkout

# Return to latest version
git checkout main
dvc checkout
```

## Understanding the Pipeline

### Stage 1: Download (`download`)
- **Input**: Configuration from `params.yaml`
- **Output**: `data/raw/` directory with CSV files
- **Purpose**: Download historical stock data using yfinance

### Stage 2: Preprocess (`preprocess`)
- **Input**: `data/raw/`
- **Output**: `data/processed/stock_data_cleaned.csv`
- **Purpose**: Clean data, handle missing values, normalize

### Stage 3: Feature Engineering (`feature_engineering`)
- **Input**: `data/processed/stock_data_cleaned.csv`
- **Output**: `data/features/stock_data_features.csv`, metrics
- **Purpose**: Create technical indicators (MA, RSI, MACD, Bollinger Bands)

### Stage 4: Split (`split`)
- **Input**: `data/features/stock_data_features.csv`
- **Output**: `data/features/train.csv`, `data/features/test.csv`
- **Purpose**: Split data into training and testing sets

## Key DVC Commands Reference

### Pipeline Management
```bash
dvc repro                    # Run entire pipeline
dvc repro <stage>            # Run specific stage
dvc dag                      # Show pipeline DAG
dvc status                   # Check pipeline status
```

### Data Tracking
```bash
dvc add <file>               # Track file with DVC
dvc push                     # Upload to remote storage
dvc pull                     # Download from remote storage
dvc checkout                 # Restore data to match Git version
```

### Experiments
```bash
dvc exp run                  # Run experiment
dvc exp show                 # Show all experiments
dvc exp diff                 # Compare experiments
dvc exp apply <exp>          # Apply experiment results
```

### Metrics
```bash
dvc metrics show             # Show all metrics
dvc metrics diff             # Compare metrics between commits
```

## Data Quality Metrics

The pipeline tracks these metrics in `metrics/data_quality.json`:

- **Total Records**: Number of data points
- **Date Range**: Start and end dates
- **Missing Values**: Count per column
- **Price Statistics**: Mean, std, min, max for each stock
- **Volume Statistics**: Trading volume statistics

## Troubleshooting

### Issue: AWS credentials not found
**Solution**: Configure AWS credentials using `aws configure` or set environment variables.

### Issue: `dvc repro` fails
**Solution**: Check `dvc status` to see what's out of sync. Run `dvc checkout` to restore data.

### Issue: Large files in Git
**Solution**: Make sure you're using `dvc add` for data files, not `git add`.

### Issue: yfinance download fails
**Solution**: Check internet connection. Some stocks might not be available for the date range specified.

## Learning Outcomes

After completing this lab, you will understand:

1. **DVC Pipelines**: How to define multi-stage ML pipelines
2. **Data Versioning**: Track changes in datasets over time
3. **Experiment Management**: Run and compare different experiments
4. **Remote Storage**: Use cloud storage for large files
5. **Reproducibility**: Recreate exact pipeline results from any commit
6. **Data Quality**: Monitor data statistics across versions

## Next Steps

- Integrate model training and evaluation stages
- Add data validation with Great Expectations
- Set up CI/CD pipeline with GitHub Actions
- Implement automated retraining on data updates
- Add visualization for metrics comparison

## References

- [DVC Documentation](https://dvc.org/doc)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Technical Analysis Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)

## Submission

Include the following in your submission:

1. Screenshot of `dvc dag` showing your pipeline
2. Screenshot of `dvc exp show` with at least 2 experiments
3. `metrics/data_quality.json` file
4. Brief report (1-2 pages) explaining:
   - How DVC pipelines differ from manual data versioning
   - Advantages of using DVC experiments
   - How you would extend this pipeline for production use

---

**Lab created for IE 7374 MLOps Course - Fall 2025**
