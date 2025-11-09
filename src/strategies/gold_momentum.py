# src/strategies/gold_momentum.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, time
from enum import Enum

class TradeSignal(Enum):
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"

class GoldMomentumStrategy:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('GoldMomentumStrategy')
        self.min_confidence = 0.60
        
    def generate_signal(self, data: pd.DataFrame, ml_prediction: str, ml_confidence: float) -> Dict[str, Any]:
        """Generate trading signal with Gold-specific filters"""
        try:
            # Basic ML signal
            if ml_prediction == "HOLD" or ml_confidence < self.min_confidence:
                return {"signal": TradeSignal.HOLD, "confidence": ml_confidence, "reason": "Low confidence"}
            
            # Apply Gold-specific filters
            filter_result = self._apply_gold_filters(data, ml_prediction)
            if not filter_result["allowed"]:
                return {"signal": TradeSignal.HOLD, "confidence": ml_confidence, "reason": filter_result["reason"]}
            
            # Calculate position size
            position_size = self._calculate_position_size(data, ml_confidence)
            
            # Calculate stop loss and take profit
            sl, tp = self._calculate_risk_levels(data, ml_prediction)
            
            return {
                "signal": TradeSignal.BUY if ml_prediction == "BUY" else TradeSignal.SELL,
                "confidence": ml_confidence,
                "position_size": position_size,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": "Momentum breakout with confirmation",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {"signal": TradeSignal.HOLD, "confidence": 0.0, "reason": f"Error: {str(e)}"}
    
    def _apply_gold_filters(self, data: pd.DataFrame, direction: str) -> Dict[str, Any]:
        """Apply Gold-specific trading filters"""
        current_time = datetime.now().time()
        current_price = data['close'].iloc[-1]
        
        # 1. Trading Session Filter
        if not self._is_trading_session(current_time):
            return {"allowed": False, "reason": "Outside trading session"}
        
        # 2. Spread Filter (simplified - using volatility as proxy)
        if self._is_high_spread_period(data):
            return {"allowed": False, "reason": "High spread period"}
        
        # 3. Volatility Filter
        if self._is_excessive_volatility(data):
            return {"allowed": False, "reason": "Excessive volatility"}
        
        # 4. News Filter (avoid FOMC, NFP times - simplified)
        if self._is_news_period(current_time):
            return {"allowed": False, "reason": "High impact news period"}
        
        # 5. Trend Consistency Filter
        if not self._has_trend_confirmation(data, direction):
            return {"allowed": False, "reason": "Lacking trend confirmation"}
        
        return {"allowed": True, "reason": "All filters passed"}
    
    def _is_trading_session(self, current_time: time) -> bool:
        """Check if current time is within trading sessions"""
        sessions = self.config.get_trading_sessions()
        
        london_start = time(8, 0)   # 08:00
        london_end = time(16, 0)    # 16:00
        ny_start = time(13, 0)      # 13:00
        ny_end = time(21, 0)        # 21:00
        
        # Check London session
        if london_start <= current_time <= london_end:
            return True
        
        # Check NY session  
        if ny_start <= current_time <= ny_end:
            return True
            
        return False
    
    def _is_high_spread_period(self, data: pd.DataFrame) -> bool:
        """Check if current period has high spread (using volatility proxy)"""
        recent_volatility = data['close'].pct_change().rolling(5).std().iloc[-1]
        avg_volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # If current volatility is 50% higher than average, consider it high spread
        return recent_volatility > (avg_volatility * 1.5)
    
    def _is_excessive_volatility(self, data: pd.DataFrame) -> bool:
        """Check for excessive volatility (avoid trading)"""
        atr = self._calculate_atr(data).iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # If ATR is more than 1% of price, consider it excessive for scalping
        return (atr / current_price) > 0.01
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high, low, close = data['high'], data['low'], data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _is_news_period(self, current_time: time) -> bool:
        """Check if current time is during high-impact news (simplified)"""
        # Avoid first 30 minutes of US open (13:30-14:00)
        if time(13, 30) <= current_time <= time(14, 0):
            return True
        
        # Avoid Asian session lunch time (low liquidity)
        if time(4, 0) <= current_time <= time(6, 0):
            return True
            
        return False
    
    def _has_trend_confirmation(self, data: pd.DataFrame, direction: str) -> bool:
        """Check if higher timeframe confirms the direction"""
        # Simple trend confirmation using EMA
        ema_8 = data['close'].ewm(span=8).mean().iloc[-1]
        ema_21 = data['close'].ewm(span=21).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if direction == "BUY":
            # For BUY: Price above EMA8, EMA8 above EMA21
            return current_price > ema_8 and ema_8 > ema_21
        else:  # SELL
            # For SELL: Price below EMA8, EMA8 below EMA21  
            return current_price < ema_8 and ema_8 < ema_21
    
    def _calculate_position_size(self, data: pd.DataFrame, confidence: float) -> float:
        """Calculate position size based on risk and confidence"""
        risk_config = self.config.get_risk_config()
        risk_per_trade = risk_config.get('risk_per_trade', 0.005)  # 0.5%
        
        # Adjust risk based on confidence
        confidence_multiplier = min(1.0, confidence / 0.8)  # Max 1.0 at 80% confidence
        adjusted_risk = risk_per_trade * confidence_multiplier
        
        # Calculate position size (simplified - in lots)
        # For Gold, 1 lot = 100 oz, 1 pip = $1 for micro lot
        account_balance = 10000  # This should come from MT5 account info
        risk_amount = account_balance * adjusted_risk
        
        # Use ATR for stop loss distance
        atr = self._calculate_atr(data).iloc[-1]
        stop_distance_pips = atr * 0.5  # Conservative stop at 0.5 ATR
        
        # Position size in lots (simplified calculation)
        pip_value = 1.0  # For micro lot
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Normalize to standard lot sizes
        position_size = max(0.01, min(1.0, position_size))  # Between 0.01 and 1.0 lots
        
        self.logger.info(f"Position size: {position_size:.2f} lots (risk: {adjusted_risk:.3%})")
        return position_size
    
    def _calculate_risk_levels(self, data: pd.DataFrame, direction: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        current_price = data['close'].iloc[-1]
        atr = self._calculate_atr(data).iloc[-1]
        
        # Use ATR-based stops
        stop_distance = atr * 0.8  # Stop at 0.8 ATR
        
        if direction == "BUY":
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * 1.5)  # 1.5R reward
        else:  # SELL
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * 1.5)  # 1.5R reward
        
        return stop_loss, take_profit