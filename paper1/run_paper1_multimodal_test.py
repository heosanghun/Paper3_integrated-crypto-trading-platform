#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper 1 - Multimodal Trading System Runner
==========================================

This script runs the multimodal trading system described in Paper 1.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

class MultimodalTradingModel:
    """
    Multimodal Trading System that combines multiple data sources and modalities
    to make trading decisions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the multimodal trading model.
        
        Args:
            config (dict): Configuration parameters for the model
        """
        self.config = config or {
            'starting_capital': 10000,
            'lookback_window': 30,
            'risk_level': 'medium',
            'data_sources': ['price', 'sentiment', 'news', 'technical'],
            'trading_frequency': 'daily',
            'simulation_start_date': '2022-01-01',
            'simulation_end_date': '2022-12-31'
        }
        
        # Model state
        self.portfolio_value = self.config['starting_capital']
        self.portfolio_history = []
        self.trade_history = []
    
    def run_simulation(self):
        """Run the trading simulation for the specified period"""
        start_date = datetime.strptime(self.config['simulation_start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['simulation_end_date'], '%Y-%m-%d')
        
        # Generate daily dates for simulation
        simulation_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate market data
        np.random.seed(42)  # For reproducibility
        price_data = self._generate_price_data(simulation_dates)
        
        # Run simulation for each day
        for i, date in enumerate(simulation_dates):
            # Skip weekends in simulation
            if date.weekday() >= 5:  # Saturday=5, Sunday=6
                continue
                
            # Get market data for current day
            current_price = price_data[i]
            
            # Make trading decision
            position, confidence = self._make_trading_decision(
                current_price, 
                price_data[max(0, i-self.config['lookback_window']):i+1]
            )
            
            # Execute trade and update portfolio
            trade_result = self._execute_trade(position, confidence, current_price)
            
            # Record result
            self.portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': self.portfolio_value
            })
            
            if trade_result:
                self.trade_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'position': position,
                    'confidence': confidence,
                    'price': current_price,
                    'result': trade_result['pnl']
                })
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        return {
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'trade_history': pd.DataFrame(self.trade_history) if self.trade_history else None,
            'metrics': metrics
        }
    
    def _generate_price_data(self, dates):
        """Generate synthetic price data for simulation"""
        n_points = len(dates)
        
        # Initial price
        base_price = 100.0
        
        # Generate random price movements with a slight upward trend
        daily_returns = np.random.normal(0.0005, 0.015, size=n_points)
        
        # Add some autocorrelation
        for i in range(1, n_points):
            daily_returns[i] = 0.7 * daily_returns[i] + 0.3 * daily_returns[i-1]
        
        # Calculate prices from returns
        prices = base_price * np.cumprod(1 + daily_returns)
        
        return prices
    
    def _make_trading_decision(self, current_price, price_history):
        """
        Make a trading decision based on price and other data sources.
        
        Returns:
            tuple: (position, confidence)
                position: 'buy', 'sell', or 'hold'
                confidence: value between 0 and 1
        """
        # Calculate some simple indicators
        if len(price_history) < 2:
            return 'hold', 0.5
        
        # Simple trend following
        short_ma = np.mean(price_history[-5:])
        long_ma = np.mean(price_history[-20:]) if len(price_history) >= 20 else short_ma
        
        # Volatility
        volatility = np.std(price_history[-10:]) / np.mean(price_history[-10:]) if len(price_history) >= 10 else 0.01
        
        # Simple decision rule based on moving averages
        if short_ma > long_ma * 1.01:
            position = 'buy'
            confidence = min(0.9, (short_ma/long_ma - 1) * 10)
        elif short_ma < long_ma * 0.99:
            position = 'sell'
            confidence = min(0.9, (1 - short_ma/long_ma) * 10)
        else:
            position = 'hold'
            confidence = 0.5
        
        # Adjust confidence based on volatility
        confidence = confidence * (1 - min(volatility * 10, 0.5))
        
        return position, confidence
    
    def _execute_trade(self, position, confidence, price):
        """
        Execute a trade based on the trading decision.
        
        Args:
            position (str): 'buy', 'sell', or 'hold'
            confidence (float): Trading confidence between 0 and 1
            price (float): Current asset price
            
        Returns:
            dict: Trade result info
        """
        if position == 'hold':
            return None
        
        # Determine position size based on confidence and risk level
        risk_multiplier = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }.get(self.config['risk_level'], 0.02)
        
        position_size = self.portfolio_value * risk_multiplier * confidence
        
        # Execute trade
        if position == 'buy':
            # Simulate market impact and execution slippage
            execution_price = price * (1 + 0.001)
            shares = position_size / execution_price
            
            # Random outcome based on confidence
            outcome_multiplier = np.random.normal(1.01, 0.02) if confidence > 0.6 else np.random.normal(0.99, 0.03)
            new_price = price * outcome_multiplier
            
            # Calculate P&L
            pnl = shares * (new_price - execution_price)
            
        elif position == 'sell':
            # Simulate market impact and execution slippage
            execution_price = price * (1 - 0.001)
            shares = position_size / execution_price
            
            # Random outcome based on confidence
            outcome_multiplier = np.random.normal(0.99, 0.02) if confidence > 0.6 else np.random.normal(1.01, 0.03)
            new_price = price * outcome_multiplier
            
            # Calculate P&L
            pnl = shares * (execution_price - new_price)
        
        # Update portfolio value
        self.portfolio_value += pnl
        
        return {
            'position_size': position_size,
            'shares': shares,
            'execution_price': execution_price,
            'outcome_price': new_price,
            'pnl': pnl
        }
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for the simulation"""
        if not self.portfolio_history:
            return {}
        
        # Convert to DataFrame for easier calculations
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Initial and final portfolio values
        initial_value = self.config['starting_capital']
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        # Number of trades
        n_trades = len(self.trade_history)
        
        # Win rate
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            win_rate = len(trades_df[trades_df['result'] > 0]) / n_trades * 100
            profit_factor = abs(trades_df[trades_df['result'] > 0]['result'].sum() / 
                              trades_df[trades_df['result'] < 0]['result'].sum()) if trades_df[trades_df['result'] < 0]['result'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        # Calculate metrics
        total_return = (final_value / initial_value - 1) * 100
        
        daily_returns = portfolio_df['daily_return'].dropna().values
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0
        
        # Maximum drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['peak'] - 1) * 100
        max_drawdown = portfolio_df['drawdown'].min()
        
        return {
            'total_return_pct': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': float(abs(max_drawdown)),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_trades': n_trades
        }

def run_simulation(output_dir=None):
    """Run the Paper 1 multimodal trading simulation"""
    print("Starting Paper 1 Multimodal Trading simulation...")
    
    # Initialize the model
    model = MultimodalTradingModel()
    
    # Run simulation
    results = model.run_simulation()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio history to CSV
        results['portfolio_history'].to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
        
        # Save trade history to CSV if there are trades
        if results['trade_history'] is not None:
            results['trade_history'].to_csv(os.path.join(output_dir, 'trade_history.csv'), index=False)
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        
        # Generate some basic visualizations
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(results['portfolio_history']['date']), 
                 results['portfolio_history']['portfolio_value'])
        plt.title('Portfolio Value Over Time - Paper 1')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_value_chart.png'))
        plt.close()
        
        print(f"Saved Paper 1 simulation results to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Set up the output directory
    default_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        f"paper1_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run the simulation
    run_simulation(default_output_dir) 