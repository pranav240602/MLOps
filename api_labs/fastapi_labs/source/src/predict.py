import numpy as np
import joblib
from typing import List, Dict, Optional
import os

def load_model():
    """
    Load the trained model and scaler
    """
    model_path = 'model/breast_cancer_model.pkl'
    scaler_path = 'model/scaler.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load imputer if it exists (for handling missing values)
    imputer = None
    imputer_path = 'model/imputer.pkl'
    if os.path.exists(imputer_path):
        imputer = joblib.load(imputer_path)
    
    return model, scaler, imputer

def make_prediction(features: List[float]) -> Dict:
    """
    Make a prediction for a single sample
    """
    model, scaler, imputer = load_model()
    
    # Reshape features
    features_array = np.array(features).reshape(1, -1)
    
    # Handle missing values if imputer exists
    if imputer is not None and np.isnan(features_array).any():
        features_array = imputer.transform(features_array)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features_scaled)[0]
        prob_malignant = float(probability[0])
        prob_benign = float(probability[1])
        confidence = float(max(probability))
    else:
        # If model doesn't support probabilities
        prob_malignant = 1.0 if prediction == 0 else 0.0
        prob_benign = 1.0 if prediction == 1 else 0.0
        confidence = 1.0
    
    # Determine diagnosis
    diagnosis = "Benign (Not Cancerous)" if prediction == 1 else "Malignant (Cancerous)"
    
    # Create detailed response
    result = {
        'prediction': int(prediction),
        'diagnosis': diagnosis,
        'confidence': confidence,
        'probability_benign': prob_benign,
        'probability_malignant': prob_malignant
    }
    
    # Add risk assessment
    if prob_malignant > 0.75:
        result['risk_level'] = '⚠️ High Risk'
        result['recommendation'] = 'Immediate medical consultation strongly recommended'
    elif prob_malignant > 0.5:
        result['risk_level'] = '⚠️ Medium Risk'
        result['recommendation'] = 'Medical consultation recommended for further evaluation'
    elif prob_malignant > 0.25:
        result['risk_level'] = 'Low Risk'
        result['recommendation'] = 'Regular monitoring recommended'
    else:
        result['risk_level'] = '✅ Very Low Risk'
        result['recommendation'] = 'Continue regular health checkups'
    
    # ADD THE MISSING INTERPRETATION FIELD
    result['interpretation'] = {
        'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
        'medical_note': 'This is a machine learning prediction and should not replace professional medical diagnosis',
        'accuracy_info': f'Model confidence: {confidence:.1%}',
        'next_steps': 'Always consult with healthcare professionals for proper diagnosis and treatment'
    }
    
    return result

def get_feature_importance():
    """
    Get feature importance from the model
    """
    model, _, _ = load_model()
    
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Try to load from saved file first
    importance_file = 'model/feature_importance.json'
    if os.path.exists(importance_file):
        import json
        with open(importance_file, 'r') as f:
            data = json.load(f)
            return data.get('top_10', None)
    
    # Otherwise calculate it
    from data import get_feature_names
    feature_names = get_feature_names()
    importance = model.feature_importances_
    
    # Sort features by importance
    feature_importance = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        {'feature': f, 'importance': float(i)}
        for f, i in feature_importance[:10]
    ]

def batch_predict(features_list: List[List[float]]) -> List[Dict]:
    """
    Make predictions for multiple samples
    """
    model, scaler, imputer = load_model()
    
    # Convert to numpy array
    features_array = np.array(features_list)
    
    # Handle missing values if imputer exists
    if imputer is not None and np.isnan(features_array).any():
        features_array = imputer.transform(features_array)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    results = []
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)
        
        for pred, prob in zip(predictions, probabilities):
            diagnosis = "Benign" if pred == 1 else "Malignant"
            confidence = float(max(prob))
            results.append({
                'prediction': int(pred),
                'diagnosis': diagnosis,
                'confidence': confidence,
                'probability_benign': float(prob[1]),
                'probability_malignant': float(prob[0]),
                'interpretation': {
                    'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
                    'medical_note': 'This is a machine learning prediction and should not replace professional medical diagnosis'
                }
            })
    else:
        for pred in predictions:
            diagnosis = "Benign" if pred == 1 else "Malignant"
            results.append({
                'prediction': int(pred),
                'diagnosis': diagnosis,
                'confidence': 1.0,
                'probability_benign': 1.0 if pred == 1 else 0.0,
                'probability_malignant': 1.0 if pred == 0 else 0.0,
                'interpretation': {
                    'confidence_level': 'High',
                    'medical_note': 'This is a machine learning prediction and should not replace professional medical diagnosis'
                }
            })
    
    return results

def get_model_info():
    """
    Get information about the loaded model
    """
    try:
        model, _, _ = load_model()
        
        # Load metrics if available
        metrics = {}
        metrics_file = 'model/metrics.json'
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        return {
            'model_type': type(model).__name__,
            'model_loaded': True,
            'metrics': metrics.get('performance_summary', {}),
            'training_date': metrics.get('dataset_info', {}).get('training_date', 'Unknown'),
            'accuracy': metrics.get('accuracy', 'Unknown')
        }
    except:
        return {
            'model_loaded': False,
            'message': 'No model found. Please train a model first.'
        }

# Test function
if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING PREDICTION MODULE")
    print("="*70)
    
    try:
        # Test loading model
        print("\n📂 Loading model...")
        model_info = get_model_info()
        
        if model_info['model_loaded']:
            print("✅ Model loaded successfully!")
            print(f"  Type: {model_info['model_type']}")
            print(f"  Accuracy: {model_info.get('accuracy', 'Unknown')}")
            
            # Test with sample features (you may need to adjust based on your feature count)
            print("\n🔮 Making a test prediction...")
            
            # Create sample features (adjust the number based on your dataset)
            # This is just an example with 30 features
            sample_features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 
                             0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                             8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
                             0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                             0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            
            result = make_prediction(sample_features)
            
            print("\n📊 Prediction Result:")
            print(f"  Diagnosis: {result['diagnosis']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Recommendation: {result['recommendation']}")
            print(f"  Interpretation: {result['interpretation']}")
            
            # Test feature importance
            print("\n🌟 Testing feature importance...")
            importance = get_feature_importance()
            if importance:
                print("Top 3 important features:")
                for i, item in enumerate(importance[:3], 1):
                    print(f"  {i}. {item['feature']}: {item['importance']:.4f}")
            
            print("\n✅ All tests passed!")
            
        else:
            print("❌ No model found. Please train a model first.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()