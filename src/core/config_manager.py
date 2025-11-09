# src/core/config_manager.py
import yaml
import json
import os
import logging
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = None):
        # Use absolute path to config directory
        if config_path is None:
            self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        else:
            self.config_path = config_path
            
        self.logger = logging.getLogger('ConfigManager')
        
        # Create config directory if it doesn't exist
        self._ensure_config_directory()
        
        # Create default config files if they don't exist
        self._create_default_configs()
        
        self.gold_config = self._load_gold_config()
        self.risk_config = self._load_risk_config()
        
        self.logger.info(f"ConfigManager initialized with config path: {self.config_path}")
    
    def _ensure_config_directory(self):
        """Ensure config directory exists"""
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)
            self.logger.info(f"Created config directory: {self.config_path}")
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist"""
        # Default gold_scalping.yaml
        gold_config_path = os.path.join(self.config_path, "gold_scalping.yaml")
        if not os.path.exists(gold_config_path):
            self.logger.info("Creating default gold_scalping.yaml")
            default_gold_config = """symbol: "XAUUSD"
timeframe: "M15"
trading_mode: "scalping"

risk_config:
  risk_per_trade: 0.005
  max_daily_trades: 8
  max_daily_loss: 0.02
  min_confidence: 0.60

trading_sessions:
  london_open: "08:00"
  london_close: "16:00"
  ny_open: "13:00"
  ny_close: "21:00"
  avoid_fomo: ["14:30-15:00", "19:00-20:00"]

position_management:
  breakeven_trigger: 1.0
  partial_profit_levels:
    - { profit_ratio: 0.5, close_percent: 50 }
    - { profit_ratio: 1.5, close_percent: 25 }
    - { profit_ratio: 2.0, close_percent: 25 }
  trailing_stop: true
  trailing_activation: 1.5

ml_config:
  features: 8
  primary_model: "lightgbm"
  retrain_interval: 100
  min_confidence_threshold: 0.60
"""
            with open(gold_config_path, 'w') as f:
                f.write(default_gold_config)
        
        # Default risk_config.json
        risk_config_path = os.path.join(self.config_path, "risk_config.json")
        if not os.path.exists(risk_config_path):
            self.logger.info("Creating default risk_config.json")
            default_risk_config = {
                "risk_per_trade": 0.005,
                "max_daily_trades": 8,
                "max_daily_loss": 0.02,
                "max_drawdown": 0.1,
                "emergency_stop_loss": 0.05
            }
            with open(risk_config_path, 'w') as f:
                json.dump(default_risk_config, f, indent=2)
    
    def _load_gold_config(self) -> Dict[str, Any]:
        """Load gold-specific configuration"""
        try:
            config_file = os.path.join(self.config_path, "gold_scalping.yaml")
            self.logger.info(f"Loading gold config from: {config_file}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info("Gold config loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading gold config: {e}")
            # Return default config as fallback
            return self._get_default_gold_config()
    
    def _load_risk_config(self) -> Dict[str, Any]:
        """Load risk configuration"""
        try:
            config_file = os.path.join(self.config_path, "risk_config.json")
            self.logger.info(f"Loading risk config from: {config_file}")
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.info("Risk config loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading risk config: {e}")
            # Return default config as fallback
            return self._get_default_risk_config()
    
    def _get_default_gold_config(self) -> Dict[str, Any]:
        """Get default gold configuration"""
        return {
            'symbol': 'XAUUSD',
            'timeframe': 'M15',
            'trading_mode': 'scalping',
            'risk_config': {
                'risk_per_trade': 0.005,
                'max_daily_trades': 8,
                'max_daily_loss': 0.02,
                'min_confidence': 0.60
            },
            'trading_sessions': {
                'london_open': '08:00',
                'london_close': '16:00',
                'ny_open': '13:00',
                'ny_close': '21:00'
            },
            'position_management': {
                'breakeven_trigger': 1.0,
                'partial_profit_levels': [
                    {'profit_ratio': 0.5, 'close_percent': 50},
                    {'profit_ratio': 1.5, 'close_percent': 25},
                    {'profit_ratio': 2.0, 'close_percent': 25}
                ],
                'trailing_stop': True,
                'trailing_activation': 1.5
            },
            'ml_config': {
                'features': 8,
                'primary_model': 'lightgbm',
                'retrain_interval': 100,
                'min_confidence_threshold': 0.60
            }
        }
    
    def _get_default_risk_config(self) -> Dict[str, Any]:
        """Get default risk configuration"""
        return {
            'risk_per_trade': 0.005,
            'max_daily_trades': 8,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.1,
            'emergency_stop_loss': 0.05
        }
    
    def get_gold_config(self) -> Dict[str, Any]:
        return self.gold_config
    
    def get_risk_config(self) -> Dict[str, Any]:
        return self.risk_config
    
    def get_position_management_config(self) -> Dict[str, Any]:
        return self.gold_config.get('position_management', {})
    
    def get_trading_sessions(self) -> Dict[str, Any]:
        return self.gold_config.get('trading_sessions', {})