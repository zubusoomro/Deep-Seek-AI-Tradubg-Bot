import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

class MarketRegimeDetector:
    def __init__(self):
        self.logger = logging.getLogger('MarketRegimeDetector')
        self.regime_history = []
    
    def detect_regime(self, data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """Detect current market regime with multiple methods"""
        try:
            if len(data) < 50:
                return {"regime": "unknown", "confidence": 0.0}
            
            # Multiple regime detection methods
            volatility_regime = self._volatility_based_regime(data)
            trend_regime = self._trend_based_regime(data)
            volume_regime = self._volume_based_regime(data)
            
            # Combine results
            combined_regime = self._combine_regime_signals(
                volatility_regime, trend_regime, volume_regime
            )
            
            # Calculate regime confidence
            confidence = self._calculate_regime_confidence(
                volatility_regime, trend_regime, volume_regime
            )
            
            regime_info = {
                "regime": combined_regime,
                "confidence": confidence,
                "volatility": volatility_regime,
                "trend": trend_regime,
                "volume": volume_regime,
                "timestamp": datetime.now()
            }
            
            self.regime_history.append(regime_info)
            
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {"regime": "unknown", "confidence": 0.0}
    
    def _volatility_based_regime(self, data: pd.DataFrame) -> str:
        """Detect regime based on volatility"""
        returns = data['close'].pct_change().dropna()
        
        # Calculate multiple volatility measures
        short_term_vol = returns.rolling(10).std().iloc[-1]
        long_term_vol = returns.rolling(50).std().iloc[-1]
        vol_ratio = short_term_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Regime classification
        if vol_ratio > 1.5:
            return "high_volatility"
        elif vol_ratio < 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"
    
    def _trend_based_regime(self, data: pd.DataFrame) -> str:
        """Detect regime based on trend strength"""
        # Calculate multiple trend indicators
        price = data['close']
        
        # ADX for trend strength
        adx = self._calculate_adx(data)
        
        # Moving average alignment
        ma_short = price.rolling(20).mean()
        ma_medium = price.rolling(50).mean()
        ma_long = price.rolling(100).mean()
        
        # Trend direction
        if ma_short.iloc[-1] > ma_medium.iloc[-1] > ma_long.iloc[-1]:
            trend_direction = "bullish"
        elif ma_short.iloc[-1] < ma_medium.iloc[-1] < ma_long.iloc[-1]:
            trend_direction = "bearish"
        else:
            trend_direction = "sideways"
        
        # Trend strength classification
        if adx > 40:
            return f"strong_{trend_direction}"
        elif adx > 25:
            return f"moderate_{trend_direction}"
        else:
            return f"weak_{trend_direction}"
    
    def _volume_based_regime(self, data: pd.DataFrame) -> str:
        """Detect regime based on volume patterns"""
        if 'volume' not in data.columns:
            return "unknown"
        
        volume = data['volume']
        price = data['close']
        
        # Volume trends
        volume_ma_short = volume.rolling(10).mean()
        volume_ma_long = volume.rolling(50).mean()
        
        # Price-Volume correlation
        price_change = price.pct_change()
        volume_change = volume.pct_change()
        pv_correlation = price_change.rolling(20).corr(volume_change).iloc[-1]
        
        # Volume regime classification
        volume_ratio = volume_ma_short.iloc[-1] / volume_ma_long.iloc[-1]
        
        if volume_ratio > 1.3 and pv_correlation > 0.3:
            return "high_volume_breakout"
        elif volume_ratio < 0.7:
            return "low_volume_consolidation"
        else:
            return "normal_volume"
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not adx.empty else 0.0
    
    def _combine_regime_signals(self, volatility: str, trend: str, volume: str) -> str:
        """Combine multiple regime signals into final regime"""
        # Simple rule-based combination
        if "high_volatility" in volatility and "strong" in trend:
            return "trending_high_vol"
        elif "high_volatility" in volatility and "weak" in trend:
            return "ranging_high_vol"
        elif "low_volatility" in volatility and "strong" in trend:
            return "trending_low_vol"
        elif "low_volatility" in volatility and "weak" in trend:
            return "ranging_low_vol"
        elif "high_volume_breakout" in volume:
            return "breakout"
        else:
            return "normal"
    
    def _calculate_regime_confidence(self, volatility: str, trend: str, volume: str) -> float:
        """Calculate confidence in regime detection"""
        confidence_factors = []
        
        # Volatility confidence
        if "high_volatility" in volatility or "low_volatility" in volatility:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Trend confidence
        if "strong" in trend:
            confidence_factors.append(0.9)
        elif "moderate" in trend:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Volume confidence
        if volume != "unknown":
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def get_regime_history(self, lookback: int = 100) -> List[Dict]:
        """Get regime history for analysis"""
        return self.regime_history[-lookback:]