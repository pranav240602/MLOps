"""
Pytest test cases for Real Estate Analytics Module
Author: Pranav
Date: November 2024
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.realestate_analytics import RealEstateAnalytics


class TestRealEstateAnalytics:
    """Test suite for RealEstateAnalytics class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analytics = RealEstateAnalytics()
    
    # Property Value Tests
    def test_calculate_property_value_valid(self):
        """Test property value calculation with valid inputs"""
        assert self.analytics.calculate_property_value(1500, 300) == 450000.0
        assert self.analytics.calculate_property_value(2000.5, 250.75) == 501625.38
    
    def test_calculate_property_value_invalid(self):
        """Test property value calculation with invalid inputs"""
        with pytest.raises(ValueError):
            self.analytics.calculate_property_value(-1000, 300)
        with pytest.raises(ValueError):
            self.analytics.calculate_property_value(1000, 0)
    
    # Mortgage Tests
    def test_monthly_mortgage_valid(self):
        """Test monthly mortgage calculation"""
        payment = self.analytics.calculate_monthly_mortgage(300000, 4.5, 30)
        assert 1500 < payment < 1600  # Expected range for this mortgage
        
    def test_monthly_mortgage_zero_interest(self):
        """Test mortgage with zero interest"""
        payment = self.analytics.calculate_monthly_mortgage(120000, 0, 10)
        assert payment == 1000.0  # 120000 / (10 * 12)
    
    def test_monthly_mortgage_invalid(self):
        """Test mortgage with invalid parameters"""
        with pytest.raises(ValueError):
            self.analytics.calculate_monthly_mortgage(-100000, 4.5, 30)
        with pytest.raises(ValueError):
            self.analytics.calculate_monthly_mortgage(100000, 4.5, 0)
    
    # ROI Tests
    def test_calculate_roi_positive(self):
        """Test ROI calculation with profit"""
        roi = self.analytics.calculate_roi(24000, 4000, 200000)
        assert roi == 10.0  # (20000 / 200000) * 100
    
    def test_calculate_roi_negative(self):
        """Test ROI calculation with loss"""
        roi = self.analytics.calculate_roi(10000, 15000, 200000)
        assert roi == -2.5  # (-5000 / 200000) * 100
    
    def test_calculate_roi_invalid(self):
        """Test ROI with invalid investment"""
        with pytest.raises(ValueError):
            self.analytics.calculate_roi(24000, 4000, 0)
    
    # Cap Rate Tests
    def test_calculate_cap_rate_valid(self):
        """Test cap rate calculation"""
        cap_rate = self.analytics.calculate_cap_rate(50000, 500000)
        assert cap_rate == 10.0
    
    def test_calculate_cap_rate_invalid(self):
        """Test cap rate with invalid property value"""
        with pytest.raises(ValueError):
            self.analytics.calculate_cap_rate(50000, 0)
    
    # Price per Square Foot Tests
    def test_price_per_sqft_valid(self):
        """Test price per square foot calculation"""
        price_sqft = self.analytics.calculate_price_per_sqft(500000, 2000)
        assert price_sqft == 250.0
    
    def test_price_per_sqft_invalid(self):
        """Test price per sqft with invalid square feet"""
        with pytest.raises(ValueError):
            self.analytics.calculate_price_per_sqft(500000, 0)
    
    # Rental Income Tests
    def test_estimate_rental_income(self):
        """Test rental income estimation"""
        assert self.analytics.estimate_rental_income(0) == 1500.0  # Studio
        assert self.analytics.estimate_rental_income(2) == 2500.0  # 2-bedroom
        assert self.analytics.estimate_rental_income(3, 1200) == 3600.0  # 3-bed in higher market
    
    def test_estimate_rental_income_invalid(self):
        """Test rental income with invalid bedrooms"""
        with pytest.raises(ValueError):
            self.analytics.estimate_rental_income(-1)
    
    # DTI Tests
    def test_debt_to_income_valid(self):
        """Test debt-to-income ratio calculation"""
        dti = self.analytics.calculate_debt_to_income(2000, 8000)
        assert dti == 25.0
    
    def test_debt_to_income_invalid(self):
        """Test DTI with invalid income"""
        with pytest.raises(ValueError):
            self.analytics.calculate_debt_to_income(2000, 0)
    
    # Appreciation Tests
    def test_property_appreciation(self):
        """Test property appreciation calculation"""
        future_value = self.analytics.property_appreciation(500000, 3, 5)
        assert future_value == 579637.04  # Approximately
    
    def test_property_appreciation_invalid(self):
        """Test appreciation with invalid parameters"""
        with pytest.raises(ValueError):
            self.analytics.property_appreciation(0, 3, 5)
        with pytest.raises(ValueError):
            self.analytics.property_appreciation(500000, 3, -1)
    
    # Break Even Tests
    def test_break_even_analysis_valid(self):
        """Test break-even analysis"""
        months = self.analytics.break_even_analysis(240000, 2500, 500)
        assert months == 120  # 240000 / 2000
    
    def test_break_even_analysis_no_profit(self):
        """Test break-even with no positive cash flow"""
        with pytest.raises(ValueError):
            self.analytics.break_even_analysis(240000, 1000, 1500)
    
    # Comprehensive Analysis Test
    def test_analyze_property_valid(self):
        """Test comprehensive property analysis"""
        property_data = {
            'price': 500000,
            'square_feet': 2000,
            'bedrooms': 3,
            'annual_rent': 36000,
            'annual_expenses': 6000
        }
        
        analysis = self.analytics.analyze_property(property_data)
        
        assert analysis['price_per_sqft'] == 250.0
        assert analysis['estimated_roi'] == 6.0
        assert analysis['cap_rate'] == 6.0
    
    def test_analyze_property_missing_keys(self):
        """Test analysis with missing required keys"""
        property_data = {
            'price': 500000,
            'square_feet': 2000
        }
        
        with pytest.raises(ValueError, match="Missing required key"):
            self.analytics.analyze_property(property_data)
    
    @pytest.mark.parametrize("sqft,price,expected", [
        (1000, 200, 200000),
        (1500, 300, 450000),
        (2500, 400, 1000000)
    ])
    def test_property_value_parametrized(self, sqft, price, expected):
        """Parametrized test for property value calculation"""
        assert self.analytics.calculate_property_value(sqft, price) == expected
