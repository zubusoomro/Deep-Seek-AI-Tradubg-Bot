import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class TrendAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('TrendAnalyzer')
    
    def analyze_trend(self, data: pd.DataFrame, 
                     multiple_timeframes: Dict[str, pd.DataFrame] = None) -> Dict[str, any]:
        """Comprehensive trend analysis"""
        try:
            trend_info = {}
            
            # Single timeframe analysis
            trend_info['primary'] = self._single_timeframe_analysis(data)
            
            # Multi-timeframe analysis if provided
            if multiple_timeframes:
                trend_info['multiple'] = self._multi_timeframe_analysis(multiple_timeframes)
                trend_info['combined_score'] = self._combine_trend_scores(trend_info)
            else:
                trend_info['combined_score'] = trend_info['primary']['score']
            
            return trend_info
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {'primary': {'direction': 'neutral', 'score': 0.5}, 'combined_score': 0.5}
    
    def _single_timeframe_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze trend for a single timeframe"""
        if len(data) < 100:
            return {'direction': 'neutral', 'score': 0.5, 'strength': 'weak'}
        
        price = data['close']
        
        # Multiple moving averages
        ma_20 = price.rolling(20).mean()
        ma_50 = price.rolling(50).mean()
        ma_100 = price.rolling(100).mean()
        ma_200 = price.rolling(200).mean()
        
        # ADX for trend strength
        adx = self._calculate_adx(data)
        
        # Ichimoku Cloud analysis
        ichimoku_signal = self._ichimoku_analysis(data)
        
        # Price position relative to MAs
        above_20 = price.iloc[-1] > ma_20.iloc[-1]
        above_50 = price.iloc[-1] > ma_50.iloc[-1]
        above_100 = price.iloc[-1] > ma_100.iloc[-1]
        above_200 = price.iloc[-1] > ma_200.iloc[-1]
        
        # MA alignment (bullish if all aligned up)
        ma_alignment = above_20 and above_50 and above_100 and above_200
        
        # Trend direction based on MA configuration
        bullish_count = sum([above_20, above_50, above_100, above_200])
        
        if bullish_count >= 3:
            direction = 'bullish'
        elif bullish_count <= 1:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Trend strength based on ADX and MA slope
        ma_slope = self._calculate_ma_slope(ma_20)
        
        if adx > 40 and ma_slope > 0.001:
            strength = 'strong'
            score = 0.8 + (adx / 100) * 0.2
        elif adx > 25:
            strength = 'moderate'
            score = 0.6 + (adx / 100) * 0.2
        else:
            strength = 'weak'
            score = 0.4 + (adx / 100) * 0.2
        
        # Adjust score based on MA alignment
        if ma_alignment and direction == 'bullish':
            score = min(score + 0.1, 1.0)
        elif not any([above_20, above_50, above_100, above_200]) and direction == 'bearish':
            score = min(score + 0.1, 1.0)
        
        return {
            'direction': direction,
            'strength': strength,
            'score': score,
            'adx': adx,
            'ma_alignment': ma_alignment,
            'ichimoku_signal': ichimoku_signal,
            'ma_20': ma_20.iloc[-1],
            'ma_50': ma_50.iloc[-1],
            'ma_100': ma_100.iloc[-1],
            'ma_200': ma_200.iloc[-1]
        }
    
    def _multi_timeframe_analysis(self, multiple_timeframes: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Analyze trend across multiple timeframes"""
        timeframe_scores = {}
        directions = []
        
        for tf_name, tf_data in multiple_timeframes.items():
            analysis = self._single_timeframe_analysis(tf_data)
            timeframe_scores[tf_name] = analysis
            directions.append(analysis['direction'])
        
        # Count trend directions
        bullish_count = directions.count('bullish')
        bearish_count = directions.count('bearish')
        neutral_count = directions.count('neutral')
        
        # Overall direction
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_direction = 'bullish'
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_direction = 'bearish'
        else:
            overall_direction = 'neutral'
        
        # Average score
        avg_score = np.mean([tf['score'] for tf in timeframe_scores.values()])
        
        # Confluence score (how many timeframes agree)
        total_frames = len(multiple_timeframes)
        confluence = max(bullish_count, bearish_count, neutral_count) / total_frames
        
        return {
            'overall_direction': overall_direction,
            'avg_score': avg_score,
            'confluence': confluence,
            'timeframe_analysis': timeframe_scores,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count
        }
    
    def _combine_trend_scores(self, trend_info: Dict) -> float:
        """Combine trend scores from multiple analyses"""
        primary_score = trend_info['primary']['score']
        
        if 'multiple' in trend_info:
            multi_score = trend_info['multiple']['avg_score']
            confluence = trend_info['multiple']['confluence']
            
            # Weighted average favoring higher confluence
            combined_score = (primary_score * 0.4 + multi_score * 0.6) * confluence
        else:
            combined_score = primary_score
        
        return min(combined_score, 1.0)
    
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
    
    def _calculate_ma_slope(self, ma_series: pd.Series, period: int = 5) -> float:
        """Calculate the slope of moving average"""
        if len(ma_series) < period:
            return 0.0
        
        recent_values = ma_series.iloc[-period:].values
        if np.isnan(recent_values).any():
            return 0.0
        
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        return slope / ma_series.iloc[-1]  # Normalize by current value
    
    def _ichimoku_analysis(self, data: pd.DataFrame) -> str:
        """Ichimoku Cloud analysis"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_period = 9
        tenkan_high = high.rolling(tenkan_period).max()
        tenkan_low = low.rolling(tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_period = 26
        kijun_high = high.rolling(kijun_period).max()
        kijun_low = low.rolling(kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_period = 52
        senkou_high = high.rolling(senkou_period).max()
        senkou_low = low.rolling(senkou_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
        
        # Current values
        current_close = close.iloc[-1]
        current_tenkan = tenkan_sen.iloc[-1]
        current_kijun = kijun_sen.iloc[-1]
        current_senkou_a = senkou_span_a.iloc[-1]
        current_senkou_b = senkou_span_b.iloc[-1]
        
        # Cloud analysis
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        
        if current_close > cloud_top and current_tenkan > current_kijun:
            return "strong_bullish"
        elif current_close > cloud_top:
            return "bullish"
        elif current_close < cloud_bottom and current_tenkan < current_kijun:
            return "strong_bearish"
        elif current_close < cloud_bottom:
            return "bearish"
        elif cloud_top > current_close > cloud_bottom:
            return "neutral_inside_cloud"
        else:
            return "neutral"
    
    def get_trend_strength_category(self, score: float) -> str:
        """Convert trend score to category"""
        if score >= 0.8:
            return "very_strong"
        elif score >= 0.7:
            return "strong"
        elif score >= 0.6:
            return "moderate"
        elif score >= 0.5:
            return "weak"
        else:
            return "very_weak"
    
    def is_trend_aligned(self, higher_tf_direction: str, lower_tf_direction: str) -> bool:
        """Check if trends are aligned across timeframes"""
        return higher_tf_direction == lower_tf_direction
    
    def get_recommended_action(self, trend_info: Dict) -> str:
        """Get recommended trading action based on trend analysis"""
        score = trend_info['combined_score']
        direction = trend_info.get('multiple', {}).get('overall_direction', 
                     trend_info['primary']['direction'])
        
        if score < 0.6:
            return "HOLD"
        
        if direction == 'bullish':
            return "BUY"
        elif direction == 'bearish':
            return "SELL"
        else:
            return "HOLD"