# Breast Cancer Prediction API
**Course:** IE 7374 MLOps  
**Student:** Pranav Rangbulla  
**Professor:** Ramin Mohammadi  
**Semester:** Fall 2025  
**Project:** FastAPI Lab 1 - Machine Learning Model Deployment

## Project Overview
A FastAPI-based machine learning system that analyzes medical features from the Wisconsin breast cancer dataset to predict tumor malignancy with confidence scores. The complete pipeline includes data preprocessing, scikit-learn model training with joblib persistence, and REST endpoints for real-time risk assessment predictions.

## Features
- **RESTful API** built with FastAPI framework
- **Machine Learning Model:** Random Forest Classifier with ~95% accuracy
- **Automatic Data Validation:** Input validation using Pydantic models
- **Missing Value Handling:** Automatic imputation for incomplete data
- **Interactive Documentation:** Swagger UI and ReDoc interfaces
- **Model Training Endpoint:** Retrain models via API
- **Performance Metrics:** Track model accuracy and feature importance
- **CORS Support:** Cross-origin resource sharing enabled

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/pranav-rangbulla/breast-cancer-prediction-api.git
cd breast-cancer-prediction-api
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the model (first time only):**
```bash
cd src
python train.py
```

4. **Start the Server:**
```bash
uvicorn main:app --reload
```

5. **Access the API:**
   http://localhost:8000/docs#/

## Project Structure
```
src/
├── data.py          # Data preprocessing and loading
├── train.py         # Model training pipeline
├── predict.py       # Prediction logic
├── main.py          # FastAPI application
├── models/          # Saved model files
└── requirements.txt # Dependencies
```

## Model Information

### Algorithm
- **Random Forest Classifier** with 150 estimators
- Cross-validated for optimal hyperparameters
- Handles imbalanced classes with stratified splitting

### Features Used
- **Mean Radius** - Mean of distances from center to points on perimeter
- **Mean Texture** - Standard deviation of gray-scale values
- **Mean Perimeter** - Mean size of core tumor
- **Mean Area** - Mean area of core tumor
- **Mean Smoothness** - Mean of local variation in radius lengths
- **Mean Compactness** - Mean of perimeter² / area - 1.0
- **Mean Concavity** - Mean of severity of concave portions of contour
- **Mean Concave Points** - Mean number of concave portions of contour
- **Mean Symmetry** - Mean symmetry of tumor
- **Mean Fractal Dimension** - Mean "coastline approximation" - 1

### Performance Metrics
- **Accuracy:** ~95%
- **Precision:** ~94%
- **Recall:** ~96%
- **F1-Score:** ~95%

## Example Usage

### Making a Prediction Request:
```json
POST /predict
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871
}
```

### Response:
```json
{
  "prediction": 0,
  "diagnosis": "Malignant",
  "confidence": 0.92,
  "probabilities": {
    "Benign": 0.08,
    "Malignant": 0.92
  },
  "confidence_level": "High",
  "risk_assessment": "High risk - immediate medical consultation recommended"
}
```

## API Screenshots
Screenshots demonstrating the working API can be found in the `/assets` folder:
- Swagger UI interface
- Successful prediction example
- Model training endpoint
- API health check

## Technologies Used
- **FastAPI:** Modern web framework for building APIs
- **scikit-learn:** Machine learning library
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing
- **Uvicorn:** ASGI server
- **Pydantic:** Data validation
- **Joblib:** Model persistence

## Model Training
To retrain the model with new data:

### Via API:
```bash
POST http://localhost:8000/train
```

### Via Command Line:
```bash
cd src
python train.py
```

## Learning Outcomes
This project demonstrates:
- RESTful API design principles
- Machine learning model deployment
- Data validation and error handling
- API documentation with OpenAPI/Swagger
- Model versioning and persistence
- Production-ready code organization
- Medical data handling best practices

## Acknowledgments
- Wisconsin Breast Cancer Dataset from UCI Machine Learning Repository
- Professor Ramin Mohammadi for course guidance
- FastAPI documentation and community

## License
This project is submitted as part of IE 7374 MLOps coursework.

**For questions or issues, please contact:** rangbulla.p@northeastern.edu
