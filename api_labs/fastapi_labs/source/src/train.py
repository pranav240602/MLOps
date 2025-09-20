import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def ensure_model_directory():
    """Create model directory if it doesn't exist"""
    os.makedirs('model', exist_ok=True)
    print("✅ Model directory ready")

def check_data_quality(X_train, y_train, X_test, y_test):
    """Check for NaN or infinite values in the data"""
    print("\n🔍 Checking data quality...")
    
    issues_found = False
    
    # Check for NaN values
    nan_train_X = np.isnan(X_train).sum()
    nan_test_X = np.isnan(X_test).sum()
    nan_train_y = np.isnan(y_train).sum()
    nan_test_y = np.isnan(y_test).sum()
    
    if nan_train_X > 0 or nan_test_X > 0 or nan_train_y > 0 or nan_test_y > 0:
        print(f"⚠️ Found NaN values:")
        if nan_train_X > 0: print(f"  X_train: {nan_train_X}")
        if nan_test_X > 0: print(f"  X_test: {nan_test_X}")
        if nan_train_y > 0: print(f"  y_train: {nan_train_y}")
        if nan_test_y > 0: print(f"  y_test: {nan_test_y}")
        issues_found = True
    
    # Check for infinite values
    inf_train = np.isinf(X_train).sum()
    inf_test = np.isinf(X_test).sum()
    
    if inf_train > 0 or inf_test > 0:
        print(f"⚠️ Found infinite values:")
        if inf_train > 0: print(f"  X_train: {inf_train}")
        if inf_test > 0: print(f"  X_test: {inf_test}")
        issues_found = True
    
    if not issues_found:
        print("✅ Data quality check passed - no NaN or infinite values")
    
    # Print data shapes
    print(f"\n📊 Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return not issues_found

def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare performance
    """
    # Check data quality first
    data_is_good = check_data_quality(X_train, y_train, X_test, y_test)
    
    if not data_is_good:
        print("⚠️ Data quality issues detected, but proceeding with training...")
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=5000,
            random_state=42,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(
            probability=True,
            kernel='rbf',
            random_state=42,
            gamma='scale'
        )
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = None
    
    print("\n" + "="*70)
    print("📊 TRAINING AND EVALUATING MULTIPLE MODELS")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            # Calculate ROC-AUC if model supports probability
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                # Handle binary classification
                if y_pred_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = None
            else:
                roc_auc = None
            
            # Store results
            results[name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc else 'N/A',
                'status': 'success'
            }
            
            # Print results
            print(f"  ✅ Training successful!")
            print(f"     Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"     Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"     Recall:    {recall:.4f} ({recall*100:.2f}%)")
            print(f"     F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
            if roc_auc:
                print(f"     ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"  ❌ Failed to train {name}")
            print(f"     Error: {str(e)}")
            results[name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Print summary
    if best_model is not None:
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print("="*70)
    else:
        print("\n❌ No model was successfully trained!")
    
    return best_model, best_model_name, results

def train_best_model(X_train, y_train, X_test, y_test, model_name="Random Forest"):
    """
    Train the best performing model with comprehensive metrics
    """
    ensure_model_directory()
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING FINAL MODEL: {model_name}")
    print("="*70)
    
    # Check data quality
    if not check_data_quality(X_train, y_train, X_test, y_test):
        print("⚠️ Warning: Data contains quality issues but proceeding...")
    
    # Create the model based on name
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=5000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'
        )
    else:
        # Default to Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    # Train model
    print(f"\n🔧 Training {model_name}...")
    model.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    # Make predictions
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    # ROC-AUC
    roc_auc = None
    if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create comprehensive metrics dictionary
    metrics = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'confusion_matrix': cm.tolist(),
        'performance_summary': {
            'accuracy_percent': f"{accuracy*100:.2f}%",
            'precision_percent': f"{precision*100:.2f}%",
            'recall_percent': f"{recall*100:.2f}%",
            'f1_score_percent': f"{f1*100:.2f}%",
            'roc_auc_percent': f"{roc_auc*100:.2f}%" if roc_auc else "N/A"
        },
        'confusion_matrix_details': {
            'true_negatives': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            'false_positives': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'true_positives': int(cm[1, 1]) if cm.shape[0] > 1 else 0
        },
        'dataset_info': {
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'n_features': int(X_train.shape[1]),
            'training_date': datetime.now().isoformat(),
            'dataset': 'Breast Cancer Wisconsin (CSV)'
        }
    }
    
    # Print performance report
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE REPORT")
    print("="*70)
    print(f"\n🎯 Metrics:")
    print(f"  ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  ✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  ✅ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    if roc_auc:
        print(f"  ✅ ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"  Actual Neg  │  {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"  Actual Pos  │  {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    print(f"\n📊 Detailed Results:")
    print(f"  True Negatives:  {cm[0,0]} (Correctly predicted negative)")
    print(f"  False Positives: {cm[0,1]} (Incorrectly predicted positive)")
    print(f"  False Negatives: {cm[1,0]} (Incorrectly predicted negative)")
    print(f"  True Positives:  {cm[1,1]} (Correctly predicted positive)")
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Malignant (0)', 'Benign (1)'],
                              digits=4))
    
    # Save model and metrics
    model_path = 'model/breast_cancer_model.pkl'
    metrics_path = 'model/metrics.json'
    
    print("\n💾 Saving model and metrics...")
    joblib.dump(model, model_path)
    print(f"  ✅ Model saved to: {model_path}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  ✅ Metrics saved to: {metrics_path}")
    
    # Save feature importance if available
    if hasattr(model, 'feature_importances_'):
        try:
            from data import get_feature_names
            feature_names = get_feature_names()
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Sort features by importance
            feature_importance_pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Save to file
            importance_data = {
                'features': [f for f, _ in feature_importance_pairs],
                'importance': [float(i) for _, i in feature_importance_pairs],
                'top_10': [
                    {'feature': f, 'importance': float(i)}
                    for f, i in feature_importance_pairs[:10]
                ]
            }
            
            with open('model/feature_importance.json', 'w') as f:
                json.dump(importance_data, f, indent=4)
            print(f"  ✅ Feature importance saved to: model/feature_importance.json")
            
            # Print top features
            print(f"\n🌟 Top 5 Most Important Features:")
            for i, (feature, importance) in enumerate(feature_importance_pairs[:5], 1):
                print(f"  {i}. {feature}: {importance:.4f}")
                
        except Exception as e:
            print(f"  ⚠️ Could not save feature importance: {e}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    return model, metrics

# Main execution
if __name__ == "__main__":
    # Import data functions
    from data import load_data, split_data, preprocess_data
    
    print("="*70)
    print("🏥 BREAST CANCER MODEL TRAINING SCRIPT")
    print("="*70)
    
    try:
        # Load data
        print("\n📂 Loading data from CSV...")
        X, y, feature_names, target_names = load_data()
        
        # Split data
        print("\n✂️ Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Preprocess data
        print("\n🔧 Preprocessing and scaling data...")
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        
        # Train and compare multiple models
        print("\n🤖 Stage 1: Comparing different models...")
        best_model, best_name, all_results = train_multiple_models(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Save comparison results
        ensure_model_directory()
        with open('model/model_comparison.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n💾 Model comparison saved to: model/model_comparison.json")
        
        # Train the best model with full metrics
        if best_model:
            print(f"\n🎯 Stage 2: Training final model ({best_name}) with full metrics...")
            final_model, final_metrics = train_best_model(
                X_train_scaled, y_train, X_test_scaled, y_test, best_name
            )
            
            print("\n" + "="*70)
            print("🎉 SUCCESS! Your model is ready for deployment!")
            print("="*70)
            print("\nNext steps:")
            print("1. Run the API: uvicorn src.main:app --reload")
            print("2. Visit: http://localhost:8000/docs")
            print("3. Test your model with the /predict endpoint")
        else:
            print("\n❌ No model could be trained successfully.")
            print("Please check your data and try again.")
            
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TRAINING FAILED!")
        print("="*70)
        print(f"Error: {e}")
        print("\nDebug information:")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("1. Ensure Breast_cancer_dataset.csv is in the assets folder")
        print("2. Check that all required packages are installed")
        print("3. Verify your virtual environment is activated")