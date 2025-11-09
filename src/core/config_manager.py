import yaml
import json
from typing import Dict, Any, List
import os
import logging

class ConfigManager:
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self.logger = logging.getLogger('ConfigManager')
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        try:
            # Load strategy configurations
            with open(os.path.join(self.config_path, "strategies.yaml"), 'r') as f:
                self.strategies = yaml.safe_load(f)
            
            # Load symbol configurations
            with open(os.path.join(self.config_path, "symbols.json"), 'r') as f:
                self.symbols = json.load(f)
            
            # Load risk configurations
            with open(os.path.join(self.config_path, "risk_config.json"), 'r') as f:
                self.risk_config = json.load(f)
            
            # Load ML configurations
            with open(os.path.join(self.config_path, "ml_config.json"), 'r') as f:
                self.ml_config = json.load(f)
                
            self.logger.info("All configurations loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")
            raise
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for specific strategy"""
        return self.strategies['strategy_parameters'].get(strategy_name, {})
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for specific symbol"""
        return self.symbols.get(symbol, {})
    
    def get_trading_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get configuration for trading mode"""
        return self.strategies['trading_modes'].get(mode, {})
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategies"""
        return self.strategies['active_strategies']
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols"""
        return list(self.symbols.keys())
    
    def update_strategy_config(self, strategy_name: str, updates: Dict[str, Any]):
        """Update strategy configuration"""
        if strategy_name in self.strategies['strategy_parameters']:
            self.strategies['strategy_parameters'][strategy_name].update(updates)
            self._save_config("strategies.yaml", self.strategies)
    
    def _save_config(self, filename: str, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            filepath = os.path.join(self.config_path, filename)
            if filename.endswith('.yaml'):
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif filename.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config {filename}: {str(e)}")