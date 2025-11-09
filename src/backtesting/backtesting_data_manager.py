# src/backtesting/backtest_data_manager.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

class BacktestDataManager:
    def __init__(self, data_directory: str = "data/backtest"):
        self.data_directory = data_directory
        self.logger = logging.getLogger('BacktestDataManager')
        os.makedirs(data_directory, exist_ok=True)
    
    def load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                           timeframe: str = "M15") -> Optional[pd.DataFrame]:
        """Load historical data for backtesting"""
        try:
            # First try to load from saved CSV
            csv_path = f"{self.data_directory}/{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            
            if os.path.exists(csv_path):
                self.logger.info(f"Loading historical data from {csv_path}")
                data = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
                return data
            
            # If no saved data, fetch from MT5
            self.logger.info("No saved backtest data found, fetching from MT5...")
            data = self._fetch_from_mt5(symbol, start_date, end_date, timeframe)
            
            if data is not None:
                # Save for future use
                data.to_csv(csv_path)
                self.logger.info(f"Saved historical data to {csv_path}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            return None
    
    def _fetch_from_mt5(self, symbol: str, start_date: datetime, end_date: datetime, 
                        timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from MT5"""
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return None
            
            # Convert timeframe string to MT5 constant
            tf_mapping = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_mapping.get(timeframe, mt5.TIMEFRAME_M15)
            
            # Fetch rates
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None:
                self.logger.error(f"Failed to fetch rates for {symbol}")
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame(rates)
            data['time'] = pd.to_datetime(data['time'], unit='s')
            data.set_index('time', inplace=True)
            
            mt5.shutdown()
            
            self.logger.info(f"Fetched {len(data)} bars for {symbol} from {start_date} to {end_date}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching from MT5: {e}")
            return None
    
    def generate_walk_forward_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, period_days: int = 30) -> List[Tuple]:
        """Generate walk-forward testing periods"""
        periods = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = current_start + timedelta(days=period_days)
            if current_end > end_date:
                current_end = end_date
            
            periods.append((current_start, current_end))
            current_start = current_end
        
        return periods