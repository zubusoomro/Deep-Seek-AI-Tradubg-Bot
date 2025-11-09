import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import time
import os
import json
import hashlib
import socket
import psutil
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

class TradingHelpers:
    """Comprehensive utility functions for trading operations"""
    
    def __init__(self):
        self.logger = logging.getLogger('TradingHelpers')
        self._cache = {}
        self._cache_lock = Lock()
    
    # MT5 Utility Functions
    def symbol_exists(self, symbol: str) -> bool:
        """Check if symbol exists in MT5"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            return symbol_info is not None
        except Exception as e:
            self.logger.error(f"Error checking symbol {symbol}: {str(e)}")
            return False
    
    def get_symbol_precision(self, symbol: str) -> int:
        """Get decimal precision for symbol"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return symbol_info.digits
            return 5  # Default precision
        except Exception as e:
            self.logger.warning(f"Error getting precision for {symbol}: {str(e)}")
            return 5
    
    def normalize_price(self, price: float, symbol: str) -> float:
        """Normalize price to symbol's precision"""
        precision = self.get_symbol_precision(symbol)
        return round(price, precision)
    
    def calculate_pip_value(self, symbol: str, lot_size: float = 1.0) -> float:
        """Calculate pip value for symbol"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            # For forex pairs, pip value is usually $10 for standard lot
            if "USD" in symbol:
                return 10.0 * lot_size
            elif "XAU" in symbol:
                return 1.0 * lot_size  # $1 per pip for gold
            else:
                return 8.0 * lot_size  # Default estimate
            
        except Exception as e:
            self.logger.error(f"Error calculating pip value for {symbol}: {str(e)}")
            return 0.0
    
    # Data Processing Functions
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLC data to different timeframe"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            timeframe_map = {
                'M1': '1T', 'M5': '5T', 'M15': '15T', 'M30': '30T',
                'H1': '1H', 'H4': '4H', 'D1': '1D', 'W1': '1W'
            }
            
            if timeframe not in timeframe_map:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            resampled = data.resample(timeframe_map[timeframe]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'spread': 'mean'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {str(e)}")
            return data
    
    def detect_gaps(self, data: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Detect price gaps in data"""
        try:
            gap_up = data['open'] > data['close'].shift(1) * (1 + threshold)
            gap_down = data['open'] < data['close'].shift(1) * (1 - threshold)
            
            gaps = pd.Series(0, index=data.index)
            gaps[gap_up] = 1    # Gap up
            gaps[gap_down] = -1 # Gap down
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error detecting gaps: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def calculate_rolling_correlations(self, data1: pd.Series, data2: pd.Series, 
                                    window: int = 20) -> pd.Series:
        """Calculate rolling correlation between two series"""
        try:
            return data1.rolling(window).corr(data2)
        except Exception as e:
            self.logger.error(f"Error calculating rolling correlation: {str(e)}")
            return pd.Series(index=data1.index)
    
    # Mathematical Functions
    def calculate_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Z-score for series"""
        try:
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            zscore = (series - rolling_mean) / rolling_std
            return zscore
        except Exception as e:
            self.logger.error(f"Error calculating Z-score: {str(e)}")
            return pd.Series(index=series.index)
    
    def calculate_rolling_percentile(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate rolling percentile rank"""
        try:
            def percentile_func(x):
                if len(x) < window:
                    return np.nan
                return (x[-1] > x[:-1]).sum() / (len(x) - 1)
            
            return series.rolling(window).apply(percentile_func, raw=True)
        except Exception as e:
            self.logger.error(f"Error calculating rolling percentile: {str(e)}")
            return pd.Series(index=series.index)
    
    def exponential_smoothing(self, series: pd.Series, alpha: float = 0.3) -> pd.Series:
        """Apply exponential smoothing to series"""
        try:
            return series.ewm(alpha=alpha).mean()
        except Exception as e:
            self.logger.error(f"Error applying exponential smoothing: {str(e)}")
            return series
    
    # Risk Management Functions
    def calculate_kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            if win_loss_ratio <= 0:
                return 0.0
            kelly = win_rate - (1 - win_rate) / win_loss_ratio
            return max(0.0, min(kelly, 0.25))  # Cap at 25% for safety
        except Exception as e:
            self.logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return 0.0
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns, (1 - confidence) * 100)
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        try:
            var = self.calculate_var(returns, confidence)
            es = returns[returns <= var].mean()
            return es if not np.isnan(es) else var
        except Exception as e:
            self.logger.error(f"Error calculating expected shortfall: {str(e)}")
            return 0.0
    
    # Time and Date Functions
    def is_market_hours(self, symbol: str, current_time: datetime = None) -> bool:
        """Check if current time is within market hours for symbol"""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            # Forex markets (simplified - open 24/5)
            if any(currency in symbol for currency in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']):
                # Monday 00:00 to Friday 23:59
                return current_time.weekday() < 5
            
            # Gold (XAU) - more limited hours
            elif 'XAU' in symbol:
                # Assume 24/5 for simplicity
                return current_time.weekday() < 5
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {str(e)}")
            return True
    
    def time_until_next_bar(self, timeframe: str) -> timedelta:
        """Calculate time until next bar closes"""
        try:
            now = datetime.now()
            
            if timeframe == 'M1':
                next_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            elif timeframe == 'M5':
                next_time = (now + timedelta(minutes=5)).replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
            elif timeframe == 'M15':
                next_time = (now + timedelta(minutes=15)).replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            elif timeframe == 'H1':
                next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            elif timeframe == 'H4':
                next_time = (now + timedelta(hours=4)).replace(hour=(now.hour // 4) * 4, minute=0, second=0, microsecond=0)
            elif timeframe == 'D1':
                next_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                next_time = now + timedelta(minutes=1)
            
            return next_time - now
            
        except Exception as e:
            self.logger.error(f"Error calculating time until next bar: {str(e)}")
            return timedelta(minutes=1)
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of trading days between dates"""
        try:
            all_days = pd.date_range(start=start_date, end=end_date, freq='D')
            trading_days = [day for day in all_days if day.weekday() < 5]  # Monday-Friday
            return trading_days
        except Exception as e:
            self.logger.error(f"Error getting trading days: {str(e)}")
            return []
    
    # File and Cache Functions
    def cache_result(self, key: str, value: Any, ttl: int = 300):
        """Cache result with time-to-live"""
        try:
            with self._cache_lock:
                self._cache[key] = {
                    'value': value,
                    'expires': time.time() + ttl
                }
        except Exception as e:
            self.logger.error(f"Error caching result: {str(e)}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        try:
            with self._cache_lock:
                if key in self._cache:
                    cached = self._cache[key]
                    if time.time() < cached['expires']:
                        return cached['value']
                    else:
                        del self._cache[key]
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached result: {str(e)}")
            return None
    
    def save_data_to_file(self, data: Any, filename: str, directory: str = "data"):
        """Save data to JSON file"""
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'w') as f:
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    data.to_json(f, orient='split', date_format='iso')
                else:
                    json.dump(data, f, indent=2, default=str)
            
            self.logger.debug(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to file: {str(e)}")
            return False
    
    def load_data_from_file(self, filename: str, directory: str = "data") -> Optional[Any]:
        """Load data from JSON file"""
        try:
            filepath = os.path.join(directory, filename)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Try to convert back to DataFrame if it has the right structure
            if isinstance(data, dict) and 'columns' in data and 'data' in data:
                return pd.DataFrame(**data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from file: {str(e)}")
            return None
    
    # System Monitoring Functions
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            }
        except Exception as e:
            self.logger.error(f"Error getting system resources: {str(e)}")
            return {}
    
    def check_internet_connection(self, host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
        """Check internet connection"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except Exception:
            return False
    
    def get_mt5_connection_status(self) -> Dict[str, Any]:
        """Get MT5 connection status and statistics"""
        try:
            account_info = mt5.account_info()
            terminal_info = mt5.terminal_info()
            
            return {
                'connected': mt5.initialize(),
                'account_number': account_info.login if account_info else None,
                'balance': account_info.balance if account_info else 0,
                'equity': account_info.equity if account_info else 0,
                'leverage': account_info.leverage if account_info else 0,
                'terminal_build': terminal_info.build if terminal_info else 0,
                'connected': terminal_info.connected if terminal_info else False
            }
        except Exception as e:
            self.logger.error(f"Error getting MT5 connection status: {str(e)}")
            return {'connected': False}
    
    # Validation Functions
    def validate_trade_signal(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate trade signal before execution"""
        try:
            required_fields = ['symbol', 'action', 'confidence', 'lot_size', 'stop_loss', 'take_profit']
            
            # Check required fields
            for field in required_fields:
                if field not in signal:
                    return False, f"Missing required field: {field}"
            
            # Validate action
            if signal['action'] not in ['BUY', 'SELL', 'HOLD']:
                return False, f"Invalid action: {signal['action']}"
            
            # Validate confidence
            if not (0 <= signal['confidence'] <= 1):
                return False, f"Invalid confidence: {signal['confidence']}"
            
            # Validate lot size
            if signal['lot_size'] <= 0:
                return False, f"Invalid lot size: {signal['lot_size']}"
            
            # Validate stop loss and take profit
            if signal['action'] == 'BUY':
                if signal['stop_loss'] >= signal.get('current_price', float('inf')):
                    return False, "Stop loss must be below current price for BUY"
                if signal['take_profit'] <= signal.get('current_price', 0):
                    return False, "Take profit must be above current price for BUY"
            elif signal['action'] == 'SELL':
                if signal['stop_loss'] <= signal.get('current_price', 0):
                    return False, "Stop loss must be above current price for SELL"
                if signal['take_profit'] >= signal.get('current_price', float('inf')):
                    return False, "Take profit must be below current price for SELL"
            
            return True, "Valid signal"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration settings"""
        errors = []
        
        try:
            # Check required sections
            required_sections = ['strategies', 'symbols', 'risk_config']
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing configuration section: {section}")
            
            # Validate risk settings
            risk_config = config.get('risk_config', {})
            if risk_config.get('max_daily_loss_percent', 100) > 50:
                errors.append("Max daily loss percent too high")
            
            if risk_config.get('max_position_size_percent', 100) > 20:
                errors.append("Max position size percent too high")
            
            # Validate symbols
            symbols = config.get('symbols', {})
            if not symbols:
                errors.append("No symbols configured")
            
            # Validate strategies
            strategies = config.get('strategies', {}).get('active_strategies', [])
            if not strategies:
                errors.append("No active strategies configured")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
            return False, errors
    
    # Performance Optimization Functions
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            # Convert object types to category where possible
            for col in df.select_dtypes(include=['object']):
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
            
            # Downcast numeric types
            for col in df.select_dtypes(include=[np.number]):
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            optimized_memory = df.memory_usage(deep=True).sum()
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            self.logger.debug(f"DataFrame memory reduced by {reduction:.1f}%")
            return df
            
        except Exception as e:
            self.logger.error(f"Error optimizing DataFrame memory: {str(e)}")
            return df
    
    def batch_process_data(self, data: pd.DataFrame, batch_size: int = 1000, 
                         process_func: callable = None) -> pd.DataFrame:
        """Process large DataFrame in batches"""
        try:
            if process_func is None:
                return data
            
            results = []
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                processed_batch = process_func(batch)
                results.append(processed_batch)
                
                if (i // batch_size) % 10 == 0:  # Log every 10 batches
                    self.logger.debug(f"Processed {i + len(batch)}/{len(data)} rows")
            
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return data
    
    # Miscellaneous Utilities
    def generate_trade_id(self, symbol: str, strategy: str) -> str:
        """Generate unique trade ID"""
        try:
            timestamp = int(time.time() * 1000)
            unique_str = f"{symbol}_{strategy}_{timestamp}"
            return hashlib.md5(unique_str.encode()).hexdigest()[:12].upper()
        except Exception as e:
            self.logger.error(f"Error generating trade ID: {str(e)}")
            return f"TRADE_{int(time.time())}"
    
    def format_currency(self, amount: float, currency: str = "USD") -> str:
        """Format currency amount"""
        try:
            if currency == "USD":
                return f"${amount:,.2f}"
            elif currency == "EUR":
                return f"€{amount:,.2f}"
            elif currency == "GBP":
                return f"£{amount:,.2f}"
            elif currency == "JPY":
                return f"¥{amount:,.0f}"
            else:
                return f"{amount:,.2f} {currency}"
        except Exception as e:
            self.logger.error(f"Error formatting currency: {str(e)}")
            return f"{amount:.2f}"
    
    def safe_division(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value on zero denominator"""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except Exception as e:
            self.logger.error(f"Error in safe division: {str(e)}")
            return default
    
    def retry_operation(self, operation: callable, max_retries: int = 3, 
                       delay: float = 1.0, backoff: float = 2.0, 
                       exceptions: tuple = (Exception,)) -> Any:
        """Retry operation with exponential backoff"""
        try:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return operation()
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff ** attempt)
                        self.logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries}), retrying in {sleep_time:.1f}s: {str(e)}")
                        time.sleep(sleep_time)
                    else:
                        self.logger.error(f"Operation failed after {max_retries} attempts: {str(e)}")
            
            raise last_exception
            
        except Exception as e:
            self.logger.error(f"Error in retry operation: {str(e)}")
            raise

# Global helper instance
_global_helpers = None

def get_helpers() -> TradingHelpers:
    """Get global TradingHelpers instance"""
    global _global_helpers
    if _global_helpers is None:
        _global_helpers = TradingHelpers()
    return _global_helpers

# Common utility functions (module-level)
def timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_valid_symbol(symbol: str) -> bool:
    """Check if symbol format is valid"""
    return bool(symbol and len(symbol) <= 12 and symbol.isalnum())

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def normalize_probability(probabilities: np.ndarray) -> np.ndarray:
    """Normalize probabilities to sum to 1"""
    total = np.sum(probabilities)
    if total == 0:
        return probabilities
    return probabilities / total

def create_directory_if_not_exists(directory: str):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes"""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return 0.0