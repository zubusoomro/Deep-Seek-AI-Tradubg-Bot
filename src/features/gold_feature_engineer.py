# src/features/gold_feature_engineer.py
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
import logging

class GoldFeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger('GoldFeatureEngineer')
        self.feature_count = 8
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate 8 essential features for Gold scalping"""
        try:
            features = {}
            
            # 1. Price Momentum (ROC 5 & 15)
            features['momentum_5'] = self._calculate_roc(data['close'], 5)
            features['momentum_15'] = self._calculate_roc(data['close'], 15)
            
            # 2. Volatility (ATR normalized)
            features['volatility'] = self._calculate_atr(data, 14)
            
            # 3. Trend Strength (EMA crossover)
            features['trend_strength'] = self._calculate_ema_crossover(data, 8, 21)
            
            # 4. Market Regime (RSI regime)
            features['market_regime'] = self._calculate_rsi_regime(data['close'], 14)
            
            # 5. Support/Resistance (Pivot levels)
            features['pivot_strength'] = self._calculate_pivot_strength(data)
            
            # 6. Session Volume (London/NY sessions)
            features['session_volume'] = self._calculate_session_volume(data)
            
            # 7. Liquidity Zones (Recent highs/lows)
            features['liquidity_zones'] = self._calculate_liquidity_zones(data, 20)
            
            # 8. Spread Conditions
            features['spread_condition'] = self._calculate_spread_condition(data)
            
            features_df = pd.DataFrame(features)
            features_df = features_df.dropna()
            
            self.logger.info(f"Generated {len(features_df)} feature samples")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return pd.DataFrame()
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Rate of Change"""
        return ((prices / prices.shift(period)) - 1) * 100
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range normalized by price"""
        high, low, close = data['high'], data['low'], data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr / close  # Normalize
    
    def _calculate_ema_crossover(self, data: pd.DataFrame, fast: int, slow: int) -> pd.Series:
        """EMA crossover strength"""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        return (ema_fast - ema_slow) / data['close']
    
    def _calculate_rsi_regime(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI-based market regime"""
        rsi = talib.RSI(prices, timeperiod=period)
        # Normalize to -1 (oversold) to +1 (overbought)
        rsi_normalized = (rsi - 50) / 50
        return rsi_normalized
    
    def _calculate_pivot_strength(self, data: pd.DataFrame) -> pd.Series:
        """Pivot point strength"""
        pivot = (data['high'] + data['low'] + data['close']) / 3
        r1 = 2 * pivot - data['low']
        s1 = 2 * pivot - data['high']
        # Normalize position between S1 and R1
        return (data['close'] - s1) / (r1 - s1)
    
    def _calculate_session_volume(self, data: pd.DataFrame) -> pd.Series:
        """Session-based volume profile (simplified)"""
        # For MT5, we might not have volume for forex, so use price movement as proxy
        london_session = data.between_time('08:00', '16:00')
        ny_session = data.between_time('13:00', '21:00')
        
        # Use range as proxy for activity
        daily_range = data['high'] - data['low']
        avg_range = daily_range.rolling(5).mean()
        session_strength = daily_range / avg_range
        
        return session_strength
    
    def _calculate_liquidity_zones(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Recent highs/lows liquidity zones"""
        recent_high = data['high'].rolling(period).max()
        recent_low = data['low'].rolling(period).min()
        current_price = data['close']
        
        # Normalize position between recent low and high
        return (current_price - recent_low) / (recent_high - recent_low)
    
    def _calculate_spread_condition(self, data: pd.DataFrame) -> pd.Series:
        """Spread condition (simplified - using volatility as proxy)"""
        volatility = data['close'].pct_change().rolling(5).std()
        avg_volatility = volatility.rolling(20).mean()
        return volatility / avg_volatility
    
    def get_feature_count(self) -> int:
        return self.feature_count