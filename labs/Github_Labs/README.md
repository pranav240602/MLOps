# Real Estate Analytics Module - MLOps Lab 1
**Submitted by: Pranav Rangbulla (pranav240602)**  
**Course: IE 7374 - MLOps**  
**Unique Implementation: Real Estate Analytics Module replacing basic calculator**
## Author
**Name:** Pranav  
**Course:** IE 7374 - MLOps  
**Professor:** Ramin Mohammadi  
**Semester:** Fall 2024  

## Project Description
This lab implements a comprehensive Real Estate Analytics module with MLOps best practices, including automated testing, CI/CD pipelines, and proper project structure. The module provides various real estate calculations and investment metrics useful for property analysis.

## Key Differentiators from Original Lab
- **Domain Focus**: Instead of a basic calculator, this implements real estate-specific analytics
- **Advanced Features**: Includes ROI calculation, mortgage calculations, property appreciation, and comprehensive property analysis
- **Enhanced Testing**: Extensive test coverage with both pytest and unittest frameworks
- **Real-World Application**: Aligned with actual use cases in property investment and analysis

## Features
- Property value calculations based on square footage
- Monthly mortgage payment calculator
- Return on Investment (ROI) analysis
- Capitalization rate calculations
- Debt-to-income ratio analysis
- Property appreciation projections
- Break-even analysis for rental properties
- Comprehensive property analysis function

## Project Structure
```
Github_Labs/Lab1/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── realestate_analytics.py
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
├── data/
│   └── __init__.py
└── .github/
    └── workflows/
        ├── pytest_action.yml
        └── unittest_action.yml
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Github_Labs/Lab1
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv realestate_env

# Activate virtual environment
# On Windows:
realestate_env\Scripts\activate
# On Mac/Linux:
source realestate_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Code

### Using the Module
```python
from src.realestate_analytics import RealEstateAnalytics

# Create instance
analytics = RealEstateAnalytics()

# Calculate property value
value = analytics.calculate_property_value(2000, 350)  # 2000 sqft at $350/sqft

# Calculate monthly mortgage
payment = analytics.calculate_monthly_mortgage(400000, 4.5, 30)  # $400k at 4.5% for 30 years

# Calculate ROI
roi = analytics.calculate_roi(36000, 6000, 300000)  # Returns percentage
```

## Running Tests

### Run Pytest
```bash
# Run all pytest tests with coverage
pytest test/test_pytest.py -v --cov=src --cov-report=html

# Run specific test
pytest test/test_pytest.py::TestRealEstateAnalytics::test_calculate_property_value_valid -v
```

### Run Unittest
```bash
# Run all unittest tests
python -m unittest test.test_unittest -v

# Run specific test class
python -m unittest test.test_unittest.TestRealEstateAnalyticsUnit -v
```

## CI/CD Pipeline
The project includes GitHub Actions workflows for automated testing:

1. **pytest_action.yml**: Runs pytest with coverage reporting
2. **unittest_action.yml**: Runs unittest framework tests

Both workflows:
- Test against Python 3.8, 3.9, and 3.10
- Cache dependencies for faster builds
- Upload test results as artifacts
- Trigger on push to main/develop and pull requests

## Functions Documentation

### Core Analytics Functions

#### `calculate_property_value(square_feet, price_per_sqft)`
Calculate total property value based on square footage.

#### `calculate_monthly_mortgage(principal, annual_rate, years)`
Calculate monthly mortgage payment using standard amortization formula.

#### `calculate_roi(rental_income, expenses, investment)`
Calculate Return on Investment as a percentage.

#### `calculate_cap_rate(net_operating_income, property_value)`
Calculate Capitalization Rate for investment properties.

#### `estimate_rental_income(bedrooms, location_multiplier)`
Estimate monthly rental income based on property characteristics.

#### `property_appreciation(initial_value, appreciation_rate, years)`
Project future property value based on appreciation rate.

#### `break_even_analysis(purchase_price, monthly_rental, monthly_expenses)`
Calculate months to break even on investment.

#### `analyze_property(property_data)`
Perform comprehensive analysis on property data dictionary.

## Test Coverage
- **Unit Tests**: 30+ test cases covering all functions
- **Edge Cases**: Tests for invalid inputs and boundary conditions
- **Parametrized Tests**: Multiple scenarios tested efficiently
- **Coverage Goal**: >90% code coverage

## Future Enhancements
- Add data visualization capabilities
- Integrate with real estate APIs
- Add machine learning models for price prediction
- Create Flask/FastAPI endpoints for web service
- Add database integration for property data storage

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is created for educational purposes as part of the MLOps course at Northeastern University.

## Acknowledgments
- Professor Ramin Mohammadi for the MLOps course structure
- Northeastern University IE 7374 MLOps course
- Original lab structure inspiration from the course repository
