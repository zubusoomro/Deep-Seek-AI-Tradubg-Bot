import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time

class MT5DataManager:
    def __init__(self):
        self.logger = logging.getLogger('MT5DataManager')
        self.connected = False
        self.symbol_info = {}
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
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
    
    def disconnect(self):
        """Disconnect from MT5 terminal"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
    
    def get_symbol_data(self, symbol: str, timeframe: str, 
                   bars: int = 1000, from_date: datetime = None) -> Optional[pd.DataFrame]:
     """Get historical data for symbol - IMPROVED VERSION"""
     try:
        if not self.connected:
            if not self.connect():
                self.logger.error("MT5 not connected")
                return None
        
        # Enhanced timeframe mapping with better error handling
        tf_map = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
        }
        
        mt5_timeframe = tf_map.get(timeframe)
        if mt5_timeframe is None:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return None
        
        # Check if symbol exists and is available
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found in MT5")
            return None
        
        if not symbol_info.visible:
            self.logger.warning(f"Symbol {symbol} not visible, attempting to select...")
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol {symbol}")
                return None
        
        # Try different methods to get data
        rates = None
        
        if from_date:
            # Method 1: Copy rates from specific date
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, from_date, bars)
        else:
            # Method 2: Copy rates from current position
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            
            # If that fails, try with a specific recent date
            if rates is None:
                recent_date = datetime.now() - timedelta(days=30)
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, recent_date, bars)
        
        if rates is None:
            self.logger.error(f"MT5 returned no rates for {symbol} {timeframe}")
            
            # Debug: Check what's available
            available_bars = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)
            if available_bars is None:
                self.logger.error(f"No data available at all for {symbol} {timeframe}")
            else:
                self.logger.info(f"Single bar available, but not {bars} bars")
            
            return None
        
        if len(rates) == 0:
            self.logger.warning(f"Empty rates array for {symbol} {timeframe}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {symbol} {timeframe}")
            return None
        
        # Convert time and set index
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Calculate additional columns
        df['spread'] = (df['high'] - df['low']).rolling(5).mean()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        self.logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
        return df
        
     except Exception as e:
        self.logger.error(f"Error getting data for {symbol} {timeframe}: {str(e)}")
        return None
    
    def get_multiple_timeframes(self, symbol: str, timeframes: List[str], 
                               bars: int = 1000) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        data = {}
        
        for tf in timeframes:
            tf_data = self.get_symbol_data(symbol, tf, bars)
            if tf_data is not None:
                data[tf] = tf_data
            else:
                self.logger.warning(f"Failed to get {tf} data for {symbol}")
        
        return data
    
    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """Get current tick data for symbol"""
        try:
            if not self.connected:
                if not self.connect():
                    return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tick for {symbol}: {str(e)}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed symbol information"""
        try:
            if symbol not in self.symbol_info:
                info = mt5.symbol_info(symbol)
                if info is None:
                    return None
                
                self.symbol_info[symbol] = {
                    'name': symbol,
                    'point': info.point,
                    'digits': info.digits,
                    'spread': info.spread,
                    'trade_contract_size': info.trade_contract_size,
                    'trade_tick_size': info.trade_tick_size,
                    'trade_tick_value': info.trade_tick_value,
                    'trade_mode': info.trade_mode,
                    'swap_mode': info.swap_mode,
                    'margin_initial': info.margin_initial,
                    'margin_maintenance': info.margin_maintenance
                }
            
            return self.symbol_info[symbol]
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get current account information"""
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                'login': account.login,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'leverage': account.leverage,
                'currency': account.currency,
                'profit': account.profit
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def calculate_volatility(self, symbol: str, timeframe: str = 'H1', 
                           period: int = 20) -> Optional[float]:
        """Calculate current volatility for symbol"""
        try:
            data = self.get_symbol_data(symbol, timeframe, bars=period + 10)
            if data is None or len(data) < period:
                return None
            
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(period).std().iloc[-1]
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return None
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return False
            
            # Check if symbol is selected and visible
            if not info.visible:
                return False
            
            # Check trading hours (simplified)
            current_time = datetime.now()
            # Add your market hours logic here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market open for {symbol}: {str(e)}")
            return False