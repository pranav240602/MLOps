"""
Lab 2: Advanced Logging for MLOps
Student: Pranav Rangbulla
Date: October 2025

This lab demonstrates logging best practices for MLOps applications
including structured logging, multiple handlers, and real-world use cases.
"""

import logging
import logging.handlers
import json
from datetime import datetime
import time

# ============================================
# 1. BASIC LOGGING SETUP WITH CUSTOM FORMAT
# ============================================

def setup_basic_logging():
    """Configure basic logging with custom format"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.debug("Debug: System initialized")
    logging.info("Info: Application started")
    logging.warning("Warning: This is a warning")
    logging.error("Error: An error occurred")
    logging.critical("Critical: Critical system failure")


# ============================================
# 2. CUSTOM LOGGER FOR ML PIPELINE
# ============================================

def create_ml_pipeline_logger():
    """Create a custom logger for ML pipeline operations"""
    
    # Create logger
    ml_logger = logging.getLogger("ml_pipeline")
    ml_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'ml_pipeline.log',
        maxBytes=1024*1024,  # 1MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    ml_logger.addHandler(console_handler)
    ml_logger.addHandler(file_handler)
    
    return ml_logger


# ============================================
# 3. STRUCTURED LOGGING WITH JSON
# ============================================

class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


def create_json_logger():
    """Create logger with JSON formatting"""
    json_logger = logging.getLogger("json_logger")
    json_logger.setLevel(logging.DEBUG)
    
    json_handler = logging.FileHandler('app_json.log')
    json_handler.setFormatter(JSONFormatter())
    
    json_logger.addHandler(json_handler)
    
    return json_logger


# ============================================
# 4. PRACTICAL ML USE CASE: MODEL TRAINING LOGGER
# ============================================

def simulate_model_training():
    """Simulate model training with logging"""
    
    ml_logger = create_ml_pipeline_logger()
    
    ml_logger.info("Starting model training pipeline")
    
    # Simulate data loading
    ml_logger.debug("Loading training data...")
    time.sleep(0.5)
    ml_logger.info("Data loaded successfully: 10000 samples")
    
    # Simulate training epochs
    for epoch in range(1, 4):
        ml_logger.info(f"Epoch {epoch}/3 started")
        time.sleep(0.3)
        
        # Simulate metrics
        loss = 0.5 / epoch
        accuracy = 0.7 + (epoch * 0.1)
        
        ml_logger.info(f"Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    ml_logger.info("Model training completed successfully")
    
    # Simulate saving model
    try:
        ml_logger.debug("Saving model to disk...")
        # Simulate potential error
        if False:  # Change to True to test error logging
            raise ValueError("Failed to save model")
        ml_logger.info("Model saved to: ./models/model_v1.pkl")
    except Exception as e:
        ml_logger.exception("Error occurred during model saving")


# ============================================
# 5. API REQUEST LOGGING (MLOPS CONTEXT)
# ============================================

def log_api_request(endpoint, method, status_code, response_time):
    """Log API requests for model serving"""
    
    api_logger = logging.getLogger("api_logger")
    api_logger.setLevel(logging.INFO)
    
    if not api_logger.handlers:
        handler = logging.FileHandler('api_requests.log')
        formatter = logging.Formatter(
            '%(asctime)s - API - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        api_logger.addHandler(handler)
    
    log_message = f"{method} {endpoint} - Status: {status_code} - Time: {response_time}ms"
    
    if status_code >= 500:
        api_logger.error(log_message)
    elif status_code >= 400:
        api_logger.warning(log_message)
    else:
        api_logger.info(log_message)


# ============================================
# 6. EXCEPTION HANDLING WITH LOGGING
# ============================================

def demonstrate_exception_logging():
    """Show how to properly log exceptions"""
    
    logger = logging.getLogger("exception_demo")
    logger.setLevel(logging.DEBUG)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Division by zero
    try:
        result = 10 / 0
    except ZeroDivisionError:
        logger.exception("Division by zero error occurred")
    
    # File not found
    try:
        with open('nonexistent_file.txt', 'r') as f:
            data = f.read()
    except FileNotFoundError:
        logger.error("File not found error", exc_info=True)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("Lab 2: Advanced Logging Demo")
    print("=" * 50)
    
    # Run demonstrations
    print("\n1. Basic Logging Setup:")
    setup_basic_logging()
    
    print("\n2. ML Pipeline Training with Logging:")
    simulate_model_training()
    
    print("\n3. API Request Logging:")
    log_api_request("/predict", "POST", 200, 45)
    log_api_request("/predict", "POST", 404, 12)
    log_api_request("/predict", "POST", 500, 150)
    
    print("\n4. Exception Logging:")
    demonstrate_exception_logging()
    
    print("\n5. JSON Logging:")
    json_logger = create_json_logger()
    json_logger.info("This is a JSON formatted log entry")
    json_logger.error("This is a JSON formatted error")
    
    print("\n" + "=" * 50)
    print("Check the generated log files:")
    print("- ml_pipeline.log")
    print("- api_requests.log")
    print("- app_json.log")
    print("=" * 50)