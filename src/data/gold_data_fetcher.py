# src/data/gold_data_fetcher.py
import MetaTrader5 as mt5
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

class GoldDataFetcher:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M15
        self.logger = logging.getLogger('GoldDataFetcher')
        load_dotenv()  # Load environment variables
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 with credentials from environment variables"""
        try:
            # Get credentials from environment
            login = int(os.getenv('MT5_LOGIN', 0))
            password = os.getenv('MT5_PASSWORD', '')
            server = os.getenv('MT5_SERVER', '')
            mt5_path = os.getenv('MT5_PATH', '')
            
            self.logger.info(f"Attempting MT5 connection - Login: {login}, Server: {server}")
            
            if not login or not password or not server:
                self.logger.error("MT5 credentials not found in environment variables")
                self.logger.error("Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file")
                return False
            
            # Initialize MT5
            if not mt5.initialize(path=mt5_path if mt5_path else None):
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to account
            authorized = mt5.login(login=login, password=password, server=server)
            
            if not authorized:
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                mt5.shutdown()
                return False
            
            self.logger.info(f"Connected to MT5 - Account: {account_info.login}, Balance: ${account_info.balance:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.logger.info("Disconnected from MT5")
    
    def get_current_data(self, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get current gold data"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
            if rates is None:
                self.logger.error("Failed to get rates from MT5")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            self.logger.debug(f"Retrieved {len(df)} current bars for {self.symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching current data: {e}")
            return None
    
    def get_historical_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data for training"""
        try:
            from_date = datetime.now() - timedelta(days=days)
            rates = mt5.copy_rates_range(self.symbol, self.timeframe, from_date, datetime.now())
            
            if rates is None:
                self.logger.error("Failed to get historical rates")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} historical bars for {self.symbol} over {days} days")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None
    
    def get_symbol_info(self, symbol: str):
        """Get symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            return info
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None