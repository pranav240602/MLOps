import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data():
    """
    Load the Breast Cancer dataset from CSV file
    """
    # Try different possible locations for the CSV file
    possible_paths = [
        'Breast_cancer_dataset.csv',
        './Breast_cancer_dataset.csv',
        '../Breast_cancer_dataset.csv',
        'assets/Breast_cancer_dataset.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError("Breast_cancer_dataset.csv not found. Please ensure it's in the same directory as your Python files.")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded successfully from: {csv_path}")
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Remove any ID columns
    id_columns = [col for col in df.columns if 'id' in col.lower() or col.lower() in ['id', 'index']]
    if id_columns:
        df = df.drop(columns=id_columns)
        print(f"Removed ID columns: {id_columns}")
    
    # Find the target column (usually 'diagnosis' for breast cancer dataset)
    target_column = None
    possible_targets = ['diagnosis', 'Diagnosis', 'target', 'class', 'label']
    
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break
    
    # If not found, look for binary column
    if target_column is None:
        for col in df.columns:
            if df[col].nunique() == 2:
                target_column = col
                print(f"Found binary column '{col}' - using as target")
                break
    
    # If still not found, use last column
    if target_column is None:
        target_column = df.columns[-1]
        print(f"Using last column '{target_column}' as target")
    
    print(f"Target column: '{target_column}'")
    
    # Separate features and target
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])
    
    # Handle target encoding
    if y.dtype == 'object':
        # Clean string values
        y = y.str.strip()
        
        # Common breast cancer encodings
        if set(y.unique()).issubset({'M', 'B'}):
            y = y.map({'M': 1, 'B': 0})  # M=Malignant=1, B=Benign=0
            target_names = ['Benign', 'Malignant']
            print("Encoded: B (Benign) -> 0, M (Malignant) -> 1")
        elif set(y.unique()).issubset({'Malignant', 'Benign'}):
            y = y.map({'Benign': 0, 'Malignant': 1})
            target_names = ['Benign', 'Malignant']
            print("Encoded: Benign -> 0, Malignant -> 1")
        else:
            # Generic encoding
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_names = list(le.classes_)
            print(f"Encoded labels: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        target_names = ['Benign', 'Malignant']
    
    # Convert features to numeric
    print("Converting features to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Handling missing values with median imputation...")
        imputer = SimpleImputer(strategy='median')
        X_values = imputer.fit_transform(X)
        
        # Save imputer
        os.makedirs('model', exist_ok=True)
        joblib.dump(imputer, 'model/imputer.pkl')
        print("Imputer saved to model/imputer.pkl")
    else:
        X_values = X.values
        print("No missing values found")
    
    # Get feature names
    feature_names = list(X.columns)
    
    # Convert to numpy arrays
    X = X_values
    y = y.values
    
    print(f"Final dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(feature_names)}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for i, count in enumerate(counts):
        label = target_names[i] if i < len(target_names) else f"Class {i}"
        print(f"  {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_names, target_names

def preprocess_data(X_train, X_test):
    """
    Scale the features for better model performance
    """
    print(f"Preprocessing data...")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Scaler saved to model/scaler.pkl")
    
    return X_train_scaled, X_test_scaled

def split_data(X, y):
    """
    Split data into training and testing sets
    """
    print(f"Splitting data...")
    print(f"Total samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def get_feature_names():
    """
    Return feature names from the CSV file
    """
    # Try to load the CSV and get feature names
    possible_paths = [
        'Breast_cancer_dataset.csv',
        './Breast_cancer_dataset.csv',
        '../Breast_cancer_dataset.csv',
        'assets/Breast_cancer_dataset.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Remove ID columns
            id_columns = [col for col in df.columns if 'id' in col.lower() or col.lower() in ['id', 'index']]
            if id_columns:
                df = df.drop(columns=id_columns)
            
            # Find and remove target column
            target_column = None
            possible_targets = ['diagnosis', 'Diagnosis', 'target', 'class', 'label']
            
            for col in possible_targets:
                if col in df.columns:
                    target_column = col
                    break
            
            if target_column is None:
                for col in df.columns:
                    if df[col].nunique() == 2:
                        target_column = col
                        break
            
            if target_column is None:
                target_column = df.columns[-1]
            
            # Return feature names (all columns except target)
            feature_names = [col for col in df.columns if col != target_column]
            return feature_names
    
    # Fallback - return common breast cancer feature names
    return [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]

# Test the data loading
if __name__ == "__main__":
    print("="*70)
    print("TESTING BREAST CANCER DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    try:
        # Test loading
        print("\nStep 1: Loading data...")
        X, y, feature_names, target_names = load_data()
        
        print("\nData loaded successfully!")
        
        # Show some feature names
        print(f"\nSample feature names (first 5 of {len(feature_names)}):")
        for i, name in enumerate(feature_names[:5]):
            print(f"    {i+1}. {name}")
        
        # Test splitting
        print("\nStep 2: Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        print("Split successful!")
        
        # Test preprocessing
        print("\nStep 3: Preprocessing...")
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        print("Preprocessing successful!")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! Ready to train models.")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED:")
        print("="*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()