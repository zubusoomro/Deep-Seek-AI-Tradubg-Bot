import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import json

class TradingLogger:
    """Advanced logging system for trading bot"""
    
    def __init__(self, name: str = "TradingBot", log_level: str = "INFO", 
                 log_dir: str = "logs", max_bytes: int = 10*1024*1024, 
                 backup_count: int = 5):
        self.name = name
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        formatter = self._create_formatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # File handler - all messages
        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'trading_bot.log'),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler - errors only
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'errors.log'),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Trade file handler - trade events only
        trade_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'trades.log'),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        trade_handler.setFormatter(self._create_trade_formatter())
        trade_handler.setLevel(logging.INFO)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(trade_handler)
    
    def _create_formatter(self) -> logging.Formatter:
        """Create standard formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _create_trade_formatter(self) -> logging.Formatter:
        """Create formatter for trade-specific logs"""
        return logging.Formatter(
            '%(asctime)s - TRADE - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def log_trade(self, action: str, symbol: str, quantity: float, 
                 price: float, confidence: float, strategy: str, 
                 pnl: float = 0.0, additional_info: dict = None):
        """Log trade execution with structured data"""
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'confidence': confidence,
            'strategy': strategy,
            'pnl': pnl,
            'additional_info': additional_info or {}
        }
        
        trade_message = f"{action} | {symbol} | Qty: {quantity} | Price: {price:.5f} | "
        trade_message += f"Conf: {confidence:.2f} | Strategy: {strategy} | PnL: {pnl:.2f}"
        
        if additional_info:
            trade_message += f" | Info: {json.dumps(additional_info)}"
        
        self.logger.info(trade_message)
        
        # Also write to trades log file
        trade_handler = next((h for h in self.logger.handlers 
                            if getattr(h, 'baseFilename', '').endswith('trades.log')), None)
        if trade_handler:
            trade_handler.acquire()
            try:
                with open(trade_handler.baseFilename, 'a') as f:
                    f.write(json.dumps(trade_data) + '\n')
            finally:
                trade_handler.release()
    
    def log_signal(self, symbol: str, signal_type: str, confidence: float, 
                  strategy: str, rationale: str):
        """Log trading signal"""
        message = f"SIGNAL | {symbol} | {signal_type} | Conf: {confidence:.2f} | "
        message += f"Strategy: {strategy} | {rationale}"
        self.logger.info(message)
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        message = "PERFORMANCE | "
        message += " | ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                              for k, v in metrics.items()])
        self.logger.info(message)
    
    def log_ml_update(self, model_name: str, accuracy: float, 
                     features_used: int, training_samples: int):
        """Log ML model updates"""
        message = f"ML_UPDATE | {model_name} | Accuracy: {accuracy:.4f} | "
        message += f"Features: {features_used} | Samples: {training_samples}"
        self.logger.info(message)
    
    def log_market_regime(self, regime: str, confidence: float, indicators: dict):
        """Log market regime detection"""
        message = f"REGIME | {regime} | Conf: {confidence:.2f} | "
        message += f"Indicators: {indicators}"
        self.logger.info(message)
    
    def debug(self, message: str):
        """Debug level log"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Info level log"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Warning level log"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Error level log"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str):
        """Critical level log"""
        self.logger.critical(message)
    
    def get_log_file_paths(self) -> dict:
        """Get paths to all log files"""
        return {
            'main': os.path.join(self.log_dir, 'trading_bot.log'),
            'errors': os.path.join(self.log_dir, 'errors.log'),
            'trades': os.path.join(self.log_dir, 'trades.log')
        }
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days"""
        try:
            current_time = datetime.now().timestamp()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            for filename in os.listdir(self.log_dir):
                filepath = os.path.join(self.log_dir, filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        self.logger.info(f"Removed old log file: {filename}")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {str(e)}")

def setup_logger(name: str = "TradingBot", **kwargs) -> TradingLogger:
    """Convenience function to setup logger"""
    return TradingLogger(name, **kwargs)

# Global logger instance
global_logger = None

def get_logger(name: str = "TradingBot") -> TradingLogger:
    """Get or create global logger instance"""
    global global_logger
    if global_logger is None:
        global_logger = setup_logger(name)
    return global_logger