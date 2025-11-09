import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class RobustDataFetcher:
    """Robust data fetcher that handles MT5 quirks and limitations"""
    
    def __init__(self):
        self.logger = logging.getLogger('RobustDataFetcher')
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to MT5 with retry logic"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
            
            self.connected = True
            self.logger.info("Successfully connected to MT5")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def get_robust_data(self, symbol: str, timeframe: str, 
                       required_bars: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get data with multiple fallback strategies"""
        try:
            if not self.connected and not self.connect():
                return None
            
            # Strategy 1: Try direct method first
            data = self._try_direct_method(symbol, timeframe, required_bars)
            if data is not None and len(data) >= required_bars * 0.8:  # 80% of required
                return self._filter_data_by_date(data, end_date)
            
            # Strategy 2: Try with date range
            self.logger.info(f"Direct method failed, trying date range for {symbol} {timeframe}")
            data = self._try_date_range_method(symbol, timeframe, required_bars, end_date)
            if data is not None:
                return data
            
            # Strategy 3: Try with smaller chunks
            self.logger.info(f"Date range method failed, trying chunked method for {symbol} {timeframe}")
            data = self._try_chunked_method(symbol, timeframe, required_bars, end_date)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in robust data fetch: {str(e)}")
            return None
    
    def _try_direct_method(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Try the direct copy_rates_from_pos method"""
        try:
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_tf = tf_map.get(timeframe)
            if mt5_tf is None:
                return None
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
            if rates is None:
                return None
            
            return self._convert_to_dataframe(rates)
            
        except Exception as e:
            self.logger.debug(f"Direct method failed: {str(e)}")
            return None
    
    def _try_date_range_method(self, symbol: str, timeframe: str, 
                             bars: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Try getting data using date range"""
        try:
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_tf = tf_map.get(timeframe)
            if mt5_tf is None:
                return None
            
            # Calculate start date based on timeframe
            if timeframe == 'M1':
                start_date = end_date - timedelta(days=2)
            elif timeframe == 'M15':
                start_date = end_date - timedelta(days=10)
            elif timeframe == 'H1':
                start_date = end_date - timedelta(days=30)
            elif timeframe == 'H4':
                start_date = end_date - timedelta(days=60)
            elif timeframe == 'D1':
                start_date = end_date - timedelta(days=180)
            else:
                start_date = end_date - timedelta(days=30)
            
            rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
            if rates is None:
                return None
            
            df = self._convert_to_dataframe(rates)
            return self._filter_data_by_date(df, end_date)
            
        except Exception as e:
            self.logger.debug(f"Date range method failed: {str(e)}")
            return None
    
    def _try_chunked_method(self, symbol: str, timeframe: str,
                          required_bars: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Try getting data in smaller chunks"""
        try:
            chunk_size = min(1000, required_bars)  # MT5 often limits to 1000 bars
            all_data = []
            
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_tf = tf_map.get(timeframe)
            if mt5_tf is None:
                return None
            
            current_end = end_date
            
            for i in range(5):  # Try up to 5 chunks
                rates = mt5.copy_rates_from(symbol, mt5_tf, current_end, chunk_size)
                if rates is None or len(rates) == 0:
                    break
                
                chunk_df = self._convert_to_dataframe(rates)
                all_data.append(chunk_df)
                
                if len(all_data) * chunk_size >= required_bars:
                    break
                
                # Move to next chunk
                current_end = chunk_df.index[0] - timedelta(hours=1)
            
            if not all_data:
                return None
            
            # Combine all chunks
            combined_df = pd.concat(all_data).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            
            return self._filter_data_by_date(combined_df, end_date)
            
        except Exception as e:
            self.logger.debug(f"Chunked method failed: {str(e)}")
            return None
    
    def _convert_to_dataframe(self, rates) -> pd.DataFrame:
        """Convert MT5 rates to DataFrame"""
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    def _filter_data_by_date(self, data: pd.DataFrame, end_date: datetime) -> pd.DataFrame:
        """Filter data to only include bars up to end_date"""
        return data[data.index <= end_date]
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False