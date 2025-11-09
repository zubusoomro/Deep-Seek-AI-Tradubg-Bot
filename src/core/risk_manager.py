import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import MetaTrader5 as mt5

class RiskManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('RiskManager')
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        
    def calculate_position_size(self, symbol: str, confidence: float, 
                              trading_mode: str, current_price: float) -> float:
        """Calculate position size based on risk parameters and confidence"""
        try:
            # Get account information
            account_info = mt5.account_info()
            if not account_info:
                self.logger.error("Failed to get account info")
                return 0.0
            
            balance = account_info.balance
            equity = account_info.equity
            
            # Get risk parameters
            risk_config = self.config.risk_config
            symbol_config = self.config.get_symbol_config(symbol)
            mode_config = self.config.get_trading_mode_config(trading_mode)
            
            # Base risk per trade
            base_risk_percent = risk_config['position_sizing']['base_risk_per_trade']
            max_position_size_percent = risk_config['account_risk']['max_position_size_percent']
            
            # Adjust risk based on confidence
            confidence_multipliers = risk_config['position_sizing']['confidence_multiplier']
            if confidence >= 0.8:
                confidence_multiplier = confidence_multipliers['high']
            elif confidence >= 0.6:
                confidence_multiplier = confidence_multipliers['medium']
            else:
                confidence_multiplier = confidence_multipliers['low']
            
            # Calculate position size
            risk_amount = balance * (base_risk_percent / 100) * confidence_multiplier
            risk_amount = min(risk_amount, balance * (max_position_size_percent / 100))
            
            # Adjust for volatility if enabled
            if risk_config['position_sizing']['volatility_adjustment']:
                volatility_factor = symbol_config.get('volatility_adjustment', 1.0)
                risk_amount *= volatility_factor
            
            # Convert to lot size
            tick_value = symbol_config.get('tick_value', 1.0)
            lot_size = symbol_config.get('lot_size', 100000)
            
            # Calculate lot size based on risk
            if tick_value > 0:
                position_size = risk_amount / (tick_value * lot_size * 0.01)  # Assuming 1% move
            else:
                position_size = risk_amount / (current_price * lot_size * 0.01)
            
            # Normalize lot size
            position_size = self._normalize_lot_size(symbol, position_size)
            
            self.logger.info(f"Position size for {symbol}: {position_size:.2f} lots "
                           f"(confidence: {confidence:.2f}, risk: ${risk_amount:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def calculate_stop_loss_take_profit(self, symbol: str, action: str, 
                                      data: pd.DataFrame, confidence: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        try:
            current_price = data['close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            risk_config = self.config.risk_config
            atr_multiplier = risk_config['stop_loss']['atr_multiplier']
            risk_reward_ratio = risk_config['take_profit']['risk_reward_ratio']
            
            # Calculate stop loss based on ATR
            if action == "BUY":
                stop_loss = current_price - (atr * atr_multiplier)
                take_profit = current_price + (atr * atr_multiplier * risk_reward_ratio)
            else:  # SELL
                stop_loss = current_price + (atr * atr_multiplier)
                take_profit = current_price - (atr * atr_multiplier * risk_reward_ratio)
            
            # Adjust based on support/resistance if enabled
            if risk_config['stop_loss']['use_support_resistance']:
                stop_loss = self._adjust_stop_to_support_resistance(
                    symbol, action, current_price, stop_loss, data
                )
            
            # Validate levels
            stop_loss, take_profit = self._validate_sl_tp_levels(
                symbol, action, current_price, stop_loss, take_profit
            )
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {str(e)}")
            # Return conservative defaults
            if action == "BUY":
                return current_price * 0.99, current_price * 1.015
            else:
                return current_price * 1.01, current_price * 0.985
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _adjust_stop_to_support_resistance(self, symbol: str, action: str, 
                                         current_price: float, stop_loss: float,
                                         data: pd.DataFrame) -> float:
        """Adjust stop loss to nearest support/resistance level"""
        try:
            # Calculate recent support and resistance
            support_level = data['low'].rolling(20).min().iloc[-1]
            resistance_level = data['high'].rolling(20).max().iloc[-1]
            
            if action == "BUY":
                # Place stop loss below recent support
                adjusted_stop = min(stop_loss, support_level * 0.995)
            else:  # SELL
                # Place stop loss above recent resistance
                adjusted_stop = max(stop_loss, resistance_level * 1.005)
            
            return adjusted_stop
            
        except Exception as e:
            self.logger.warning(f"Error adjusting stop to S/R: {str(e)}")
            return stop_loss
    
    def _validate_sl_tp_levels(self, symbol: str, action: str, current_price: float,
                             stop_loss: float, take_profit: float) -> Tuple[float, float]:
        """Validate and adjust SL/TP levels to ensure they're reasonable"""
        # Calculate minimum distance (10 pips for forex, $2 for gold)
        if "XAU" in symbol:
            min_distance = 2.0
        else:
            min_distance = current_price * 0.001  # 0.1%
        
        # Ensure minimum distance from current price
        if action == "BUY":
            if stop_loss >= current_price - min_distance:
                stop_loss = current_price - min_distance
            if take_profit <= current_price + min_distance:
                take_profit = current_price + min_distance
        else:  # SELL
            if stop_loss <= current_price + min_distance:
                stop_loss = current_price + min_distance
            if take_profit >= current_price - min_distance:
                take_profit = current_price - min_distance
        
        # Ensure positive risk-reward ratio
        if action == "BUY":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        else:
            risk = stop_loss - current_price
            reward = current_price - take_profit
        
        if reward / risk < 1.0:
            # Adjust take profit to maintain minimum 1:1 risk-reward
            min_reward_ratio = 1.2
            if action == "BUY":
                take_profit = current_price + (risk * min_reward_ratio)
            else:
                take_profit = current_price - (risk * min_reward_ratio)
        
        return stop_loss, take_profit
    
    def _normalize_lot_size(self, symbol: str, lot_size: float) -> float:
        """Normalize lot size to broker requirements"""
        # MT5 typically allows 0.01 lot increments
        normalized_lots = round(lot_size / 0.01) * 0.01
        
        # Ensure minimum lot size
        min_lots = 0.01
        normalized_lots = max(normalized_lots, min_lots)
        
        return normalized_lots
    
    def can_open_new_position(self, symbol: str) -> bool:
        """Check if new position can be opened based on risk rules"""
        try:
            risk_config = self.config.risk_config
            account_info = mt5.account_info()
            
            if not account_info:
                return False
            
            # Check daily loss limit
            daily_loss_limit = account_info.balance * (risk_config['account_risk']['max_daily_loss_percent'] / 100)
            if self.daily_pnl < -daily_loss_limit:
                self.logger.warning("Daily loss limit reached")
                return False
            
            # Check total loss limit
            total_loss_limit = account_info.balance * (risk_config['account_risk']['max_total_loss_percent'] / 100)
            if self.total_pnl < -total_loss_limit:
                self.logger.warning("Total loss limit reached")
                return False
            
            # Check maximum daily trades
            max_daily_trades = risk_config.get('max_daily_trades', 50)
            if self.daily_trades >= max_daily_trades:
                self.logger.warning("Daily trade limit reached")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position eligibility: {str(e)}")
            return False
    
    def update_pnl(self, pnl: float):
        """Update PnL tracking"""
        self.daily_pnl += pnl
        self.total_pnl += pnl
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.daily_trades = 0