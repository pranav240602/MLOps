"""
Unittest test cases for Real Estate Analytics Module
Author: Pranav
Date: November 2024
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.realestate_analytics import RealEstateAnalytics


class TestRealEstateAnalyticsUnit(unittest.TestCase):
    """Unittest suite for RealEstateAnalytics class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analytics = RealEstateAnalytics()
    
    # Property Value Tests
    def test_calculate_property_value_basic(self):
        """Test basic property value calculation"""
        result = self.analytics.calculate_property_value(2000, 350)
        self.assertEqual(result, 700000.0)
    
    def test_calculate_property_value_decimal(self):
        """Test property value with decimal inputs"""
        result = self.analytics.calculate_property_value(1750.5, 299.99)
        self.assertAlmostEqual(result, 525134.50, places=2)
    
    def test_calculate_property_value_raises_error(self):
        """Test property value with negative inputs"""
        with self.assertRaises(ValueError):
            self.analytics.calculate_property_value(-500, 300)
    
    # Mortgage Calculation Tests
    def test_monthly_mortgage_standard(self):
        """Test standard mortgage calculation"""
        result = self.analytics.calculate_monthly_mortgage(400000, 5.0, 30)
        self.assertGreater(result, 2000)
        self.assertLess(result, 2200)
    
    def test_monthly_mortgage_short_term(self):
        """Test mortgage with shorter term"""
        result = self.analytics.calculate_monthly_mortgage(200000, 4.0, 15)
        self.assertGreater(result, 1400)
        self.assertLess(result, 1600)
    
    def test_monthly_mortgage_zero_interest(self):
        """Test mortgage with no interest"""
        result = self.analytics.calculate_monthly_mortgage(240000, 0, 20)
        self.assertEqual(result, 1000.0)
    
    # ROI Calculation Tests  
    def test_roi_profitable_investment(self):
        """Test ROI for profitable investment"""
        result = self.analytics.calculate_roi(30000, 5000, 250000)
        self.assertEqual(result, 10.0)
    
    def test_roi_break_even(self):
        """Test ROI at break-even point"""
        result = self.analytics.calculate_roi(10000, 10000, 200000)
        self.assertEqual(result, 0.0)
    
    def test_roi_loss_investment(self):
        """Test ROI for loss-making investment"""
        result = self.analytics.calculate_roi(8000, 12000, 150000)
        self.assertLess(result, 0)
    
    # Cap Rate Tests
    def test_cap_rate_standard(self):
        """Test standard cap rate calculation"""
        result = self.analytics.calculate_cap_rate(40000, 500000)
        self.assertEqual(result, 8.0)
    
    def test_cap_rate_high_return(self):
        """Test cap rate for high return property"""
        result = self.analytics.calculate_cap_rate(75000, 500000)
        self.assertEqual(result, 15.0)
    
    # Price Per Square Foot Tests
    def test_price_per_sqft_whole_numbers(self):
        """Test price per sqft with whole numbers"""
        result = self.analytics.calculate_price_per_sqft(600000, 2400)
        self.assertEqual(result, 250.0)
    
    def test_price_per_sqft_decimals(self):
        """Test price per sqft with decimals"""
        result = self.analytics.calculate_price_per_sqft(525000, 1750)
        self.assertEqual(result, 300.0)
    
    # Rental Income Estimation Tests
    def test_rental_income_studio(self):
        """Test rental income for studio"""
        result = self.analytics.estimate_rental_income(0)
        self.assertEqual(result, 1500.0)
    
    def test_rental_income_multi_bedroom(self):
        """Test rental income for multi-bedroom"""
        result = self.analytics.estimate_rental_income(4)
        self.assertEqual(result, 3500.0)
    
    def test_rental_income_with_multiplier(self):
        """Test rental income with location multiplier"""
        result = self.analytics.estimate_rental_income(2, 800)
        self.assertEqual(result, 2000.0)
    
    # Debt-to-Income Tests
    def test_dti_healthy_ratio(self):
        """Test healthy DTI ratio"""
        result = self.analytics.calculate_debt_to_income(1500, 6000)
        self.assertEqual(result, 25.0)
    
    def test_dti_high_ratio(self):
        """Test high DTI ratio"""
        result = self.analytics.calculate_debt_to_income(4000, 8000)
        self.assertEqual(result, 50.0)
    
    # Appreciation Tests
    def test_appreciation_standard(self):
        """Test standard property appreciation"""
        result = self.analytics.property_appreciation(400000, 4, 10)
        self.assertGreater(result, 590000)
        self.assertLess(result, 600000)
    
    def test_appreciation_zero_years(self):
        """Test appreciation with zero years"""
        result = self.analytics.property_appreciation(500000, 5, 0)
        self.assertEqual(result, 500000.0)
    
    # Break-Even Analysis Tests
    def test_break_even_standard(self):
        """Test standard break-even analysis"""
        result = self.analytics.break_even_analysis(300000, 3000, 1000)
        self.assertEqual(result, 150)
    
    def test_break_even_quick_return(self):
        """Test break-even with quick return"""
        result = self.analytics.break_even_analysis(100000, 5000, 0)
        self.assertEqual(result, 20)
    
    # Comprehensive Analysis Test
    def test_analyze_property_complete(self):
        """Test complete property analysis"""
        property_data = {
            'price': 600000,
            'square_feet': 2400,
            'bedrooms': 4,
            'annual_rent': 48000,
            'annual_expenses': 8000
        }
        
        result = self.analytics.analyze_property(property_data)
        
        self.assertEqual(result['price_per_sqft'], 250.0)
        self.assertAlmostEqual(result['estimated_roi'], 6.67, places=2)
        self.assertAlmostEqual(result['cap_rate'], 6.67, places=2)
    
    def test_analyze_property_minimal(self):
        """Test property analysis with minimal data"""
        property_data = {
            'price': 400000,
            'square_feet': 1600,
            'bedrooms': 2,
            'annual_rent': 30000
        }
        
        result = self.analytics.analyze_property(property_data)
        
        self.assertEqual(result['price_per_sqft'], 250.0)
        self.assertEqual(result['estimated_roi'], 7.5)
    
    def test_analyze_property_missing_required(self):
        """Test analysis with missing required fields"""
        property_data = {
            'price': 400000,
            'bedrooms': 2
        }
        
        with self.assertRaises(ValueError):
            self.analytics.analyze_property(property_data)


if __name__ == '__main__':
    unittest.main()
