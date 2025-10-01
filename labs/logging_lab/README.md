# Lab 2: Advanced Logging for MLOps

**Student:** Pranav Rangbulla  
**Course:** IE 7374 MLOps  
**Date:** October 2025

## Overview
This lab demonstrates advanced logging practices for MLOps applications, including:
- Custom loggers for different components
- Rotating file handlers to manage log file sizes
- JSON structured logging for better parsing
- ML pipeline logging with training metrics
- API request logging for model serving endpoints
- Exception handling with proper logging

## Files
- `logging_demo.py` - Main demonstration script with all logging examples
- `ml_pipeline.log` - ML training logs (generated when you run the script)
- `api_requests.log` - API request logs (generated when you run the script)
- `app_json.log` - JSON formatted logs (generated when you run the script)

## How to Run
```bash
cd labs/logging_lab
python logging_demo.py