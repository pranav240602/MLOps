"""
Real Estate Analytics Module for MLOps Lab
Author: Pranav
Date: November 2024
Description: Real estate calculations and analytics for property analysis
"""

import math
from typing import Union, List, Dict, Optional
from datetime import datetime


class RealEstateAnalytics:
    """Real Estate Analytics for property calculations and investment metrics"""
    
    @staticmethod
    def calculate_property_value(square_feet: float, price_per_sqft: float) -> float:
        """
        Calculate total property value based on square footage
        
        Args:
            square_feet: Total square footage of property
            price_per_sqft: Price per square foot in the area
            
        Returns:
            Estimated property value
        """
        if square_feet <= 0 or price_per_sqft <= 0:
            raise ValueError("Square feet and price must be positive")
        return round(square_feet * price_per_sqft, 2)
    
    @staticmethod
    def calculate_monthly_mortgage(principal: float, annual_rate: float, years: int) -> float:
        """
        Calculate monthly mortgage payment
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate (as percentage, e.g., 4.5 for 4.5%)
            years: Loan term in years
            
        Returns:
            Monthly payment amount
        """
        if principal <= 0 or annual_rate < 0 or years <= 0:
            raise ValueError("Invalid mortgage parameters")
        
        if annual_rate == 0:
            return round(principal / (years * 12), 2)
        
        monthly_rate = annual_rate / 100 / 12
        num_payments = years * 12
        
        payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                 ((1 + monthly_rate)**num_payments - 1)
        
        return round(payment, 2)
    
    @staticmethod
    def calculate_roi(rental_income: float, expenses: float, investment: float) -> float:
        """
        Calculate Return on Investment (ROI) percentage
        
        Args:
            rental_income: Annual rental income
            expenses: Annual expenses (maintenance, taxes, etc.)
            investment: Initial investment amount
            
        Returns:
            ROI percentage
        """
        if investment <= 0:
            raise ValueError("Investment must be positive")
        
        net_income = rental_income - expenses
        roi = (net_income / investment) * 100
        return round(roi, 2)
    
    @staticmethod
    def calculate_cap_rate(net_operating_income: float, property_value: float) -> float:
        """
        Calculate Capitalization Rate
        
        Args:
            net_operating_income: Annual NOI
            property_value: Current market value of property
            
        Returns:
            Cap rate percentage
        """
        if property_value <= 0:
            raise ValueError("Property value must be positive")
        
        cap_rate = (net_operating_income / property_value) * 100
        return round(cap_rate, 2)
    
    @staticmethod
    def calculate_price_per_sqft(property_price: float, square_feet: float) -> float:
        """
        Calculate price per square foot
        
        Args:
            property_price: Total property price
            square_feet: Total square footage
            
        Returns:
            Price per square foot
        """
        if square_feet <= 0:
            raise ValueError("Square feet must be positive")
        
        return round(property_price / square_feet, 2)
    
    @staticmethod
    def estimate_rental_income(bedrooms: int, location_multiplier: float = 1000) -> float:
        """
        Estimate monthly rental income based on bedrooms and location
        
        Args:
            bedrooms: Number of bedrooms
            location_multiplier: Base rent multiplier for location (default Boston area)
            
        Returns:
            Estimated monthly rental income
        """
        if bedrooms < 0:
            raise ValueError("Bedrooms cannot be negative")
        
        base_rent = 1500  # Base rent for studio
        bedroom_premium = 500  # Additional rent per bedroom
        
        estimated_rent = base_rent + (bedrooms * bedroom_premium)
        return round(estimated_rent * (location_multiplier / 1000), 2)
    
    @staticmethod
    def calculate_debt_to_income(monthly_debt: float, monthly_income: float) -> float:
        """
        Calculate debt-to-income ratio
        
        Args:
            monthly_debt: Total monthly debt payments
            monthly_income: Total monthly income
            
        Returns:
            DTI ratio as percentage
        """
        if monthly_income <= 0:
            raise ValueError("Monthly income must be positive")
        
        dti = (monthly_debt / monthly_income) * 100
        return round(dti, 2)
    
    @staticmethod
    def property_appreciation(initial_value: float, appreciation_rate: float, years: int) -> float:
        """
        Calculate property value after appreciation
        
        Args:
            initial_value: Initial property value
            appreciation_rate: Annual appreciation rate (as percentage)
            years: Number of years
            
        Returns:
            Future property value
        """
        if initial_value <= 0 or years < 0:
            raise ValueError("Invalid appreciation parameters")
        
        rate = appreciation_rate / 100
        future_value = initial_value * ((1 + rate) ** years)
        return round(future_value, 2)
    
    @staticmethod
    def break_even_analysis(purchase_price: float, monthly_rental: float, 
                          monthly_expenses: float) -> int:
        """
        Calculate break-even point in months
        
        Args:
            purchase_price: Total purchase price
            monthly_rental: Monthly rental income
            monthly_expenses: Monthly expenses
            
        Returns:
            Number of months to break even
        """
        monthly_profit = monthly_rental - monthly_expenses
        
        if monthly_profit <= 0:
            raise ValueError("No positive cash flow - break even not possible")
        
        months_to_break_even = math.ceil(purchase_price / monthly_profit)
        return months_to_break_even
    
    @staticmethod
    def analyze_property(property_data: Dict) -> Dict:
        """
        Comprehensive property analysis
        
        Args:
            property_data: Dictionary with property details
            
        Returns:
            Analysis results dictionary
        """
        required_keys = ['price', 'square_feet', 'bedrooms', 'annual_rent']
        
        for key in required_keys:
            if key not in property_data:
                raise ValueError(f"Missing required key: {key}")
        
        analysis = {
            'price_per_sqft': RealEstateAnalytics.calculate_price_per_sqft(
                property_data['price'], 
                property_data['square_feet']
            ),
            'estimated_roi': RealEstateAnalytics.calculate_roi(
                property_data.get('annual_rent', 0),
                property_data.get('annual_expenses', 0),
                property_data['price']
            ),
            'cap_rate': RealEstateAnalytics.calculate_cap_rate(
                property_data.get('annual_rent', 0) - property_data.get('annual_expenses', 0),
                property_data['price']
            )
        }
        
        return analysis
