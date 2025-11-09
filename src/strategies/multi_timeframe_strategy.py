import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from ..core.trading_bot import TradeSignal

@dataclass
class TimeframeAnalysis:
    trend: str
    momentum: float
    volatility: float
    support: float
    resistance: float
    rsi: float
    macd_signal: str

class MultiTimeframeStrategy:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('MultiTimeframeStrategy')
        self.strategy_name = "multi_timeframe_momentum"
    
    def generate_signal(self, symbol: str, multi_timeframe_data: Dict, 
                       ml_confidence: float) -> Optional[TradeSignal]:
        """Generate signals based on multi-timeframe confluence"""
        try:
            required_tfs = ['H4', 'H1', 'M15']
            if not all(tf in multi_timeframe_data for tf in required_tfs):
                return None
            
            # Analyze each timeframe
            h4_analysis = self._analyze_timeframe(multi_timeframe_data['H4'], 'H4')
            h1_analysis = self._analyze_timeframe(multi_timeframe_data['H1'], 'H1') 
            m15_analysis = self._analyze_timeframe(multi_timeframe_data['M15'], 'M15')
            
            # Get confluence score
            confluence = self._calculate_confluence(h4_analysis, h1_analysis, m15_analysis)
            
            if confluence['score'] < 0.65:
                return None
            
            # Generate signal
            return self._create_signal(symbol, confluence, ml_confidence)
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe strategy error for {symbol}: {str(e)}")
            return None
    
    def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> TimeframeAnalysis:
        """Comprehensive timeframe analysis"""
        if len(data) < 50:
            return TimeframeAnalysis('neutral', 0, 0, 0, 0, 50, 'neutral')
        
        # Trend analysis
        trend = self._determine_trend(data)
        
        # Momentum analysis
        momentum = self._calculate_momentum(data)
        
        # Volatility
        volatility = self._calculate_volatility(data)
        
        # Support/Resistance
        support, resistance = self._calculate_support_resistance(data)
        
        # RSI
        rsi = self._calculate_rsi(data['close']).iloc[-1]
        
        # MACD signal
        macd_signal = self._get_macd_signal(data)
        
        return TimeframeAnalysis(
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            support=support,
            resistance=resistance,
            rsi=rsi,
            macd_signal=macd_signal
        )
    
    def _calculate_confluence(self, h4: TimeframeAnalysis, h1: TimeframeAnalysis, 
                            m15: TimeframeAnalysis) -> Dict:
        """Calculate confluence across timeframes"""
        # Trend alignment (40% weight)
        trend_score = self._calculate_trend_alignment(h4.trend, h1.trend, m15.trend)
        
        # Momentum alignment (30% weight)
        momentum_score = self._calculate_momentum_alignment(h4.momentum, h1.momentum, m15.momentum)
        
        # RSI alignment (20% weight)
        rsi_score = self._calculate_rsi_alignment(h4.rsi, h1.rsi, m15.rsi)
        
        # MACD alignment (10% weight)
        macd_score = self._calculate_macd_alignment(h4.macd_signal, h1.macd_signal, m15.macd_signal)
        
        total_score = (trend_score * 0.4 + momentum_score * 0.3 + 
                      rsi_score * 0.2 + macd_score * 0.1)
        
        # Determine direction
        direction = self._determine_direction(h4, h1, m15, total_score)
        
        return {
            'score': total_score,
            'direction': direction,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'rationale': f"MTF Confluence: H4({h4.trend}), H1({h1.trend}), M15({m15.trend})"
        }
    
    def _calculate_trend_alignment(self, h4_trend: str, h1_trend: str, m15_trend: str) -> float:
        """Calculate trend alignment score"""
        trends = [h4_trend, h1_trend, m15_trend]
        
        if all(t == 'bullish' for t in trends):
            return 1.0
        elif all(t == 'bearish' for t in trends):
            return 1.0
        elif trends.count('bullish') >= 2:
            return 0.8
        elif trends.count('bearish') >= 2:
            return 0.8
        else:
            return 0.3
    
    def _calculate_momentum_alignment(self, h4_mom: float, h1_mom: float, m15_mom: float) -> float:
        """Calculate momentum alignment score"""
        if h4_mom > 0 and h1_mom > 0 and m15_mom > 0:
            return 1.0
        elif h4_mom < 0 and h1_mom < 0 and m15_mom < 0:
            return 1.0
        elif (h4_mom > 0 and h1_mom > 0) or (h4_mom > 0 and m15_mom > 0):
            return 0.7
        elif (h4_mom < 0 and h1_mom < 0) or (h4_mom < 0 and m15_mom < 0):
            return 0.7
        else:
            return 0.4
    
    def _calculate_rsi_alignment(self, h4_rsi: float, h1_rsi: float, m15_rsi: float) -> float:
        """Calculate RSI alignment score"""
        # Check for oversold conditions (bullish)
        if h4_rsi < 30 and h1_rsi < 30 and m15_rsi < 30:
            return 1.0
        # Check for overbought conditions (bearish)
        elif h4_rsi > 70 and h1_rsi > 70 and m15_rsi > 70:
            return 1.0
        # Partial alignment
        elif (h4_rsi < 30 and h1_rsi < 30) or (h4_rsi < 30 and m15_rsi < 30):
            return 0.7
        elif (h4_rsi > 70 and h1_rsi > 70) or (h4_rsi > 70 and m15_rsi > 70):
            return 0.7
        else:
            return 0.5
    
    def _calculate_macd_alignment(self, h4_macd: str, h1_macd: str, m15_macd: str) -> float:
        """Calculate MACD alignment score"""
        if h4_macd == h1_macd == m15_macd == 'bullish':
            return 1.0
        elif h4_macd == h1_macd == m15_macd == 'bearish':
            return 1.0
        elif (h4_macd == h1_macd == 'bullish') or (h4_macd == m15_macd == 'bullish'):
            return 0.8
        elif (h4_macd == h1_macd == 'bearish') or (h4_macd == m15_macd == 'bearish'):
            return 0.8
        else:
            return 0.4
    
    def _determine_direction(self, h4: TimeframeAnalysis, h1: TimeframeAnalysis, 
                           m15: TimeframeAnalysis, score: float) -> str:
        """Determine trade direction based on analysis"""
        if score < 0.6:
            return "HOLD"
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Count bullish/bearish signals
        for analysis in [h4, h1, m15]:
            if analysis.trend == 'bullish':
                bullish_signals += 1
            elif analysis.trend == 'bearish':
                bearish_signals += 1
            
            if analysis.momentum > 0:
                bullish_signals += 0.5
            elif analysis.momentum < 0:
                bearish_signals += 0.5
            
            if analysis.macd_signal == 'bullish':
                bullish_signals += 0.5
            elif analysis.macd_signal == 'bearish':
                bearish_signals += 0.5
        
        if bullish_signals > bearish_signals:
            return "BUY"
        elif bearish_signals > bullish_signals:
            return "SELL"
        else:
            return "HOLD"
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine trend direction"""
        price = data['close']
        
        # Multiple moving averages
        sma_20 = price.rolling(20).mean()
        sma_50 = price.rolling(50).mean()
        
        # Price position
        above_20 = price.iloc[-1] > sma_20.iloc[-1]
        above_50 = price.iloc[-1] > sma_50.iloc[-1]
        
        # Slope
        sma_20_slope = sma_20.iloc[-1] - sma_20.iloc[-5]
        sma_50_slope = sma_50.iloc[-1] - sma_50.iloc[-10]
        
        if above_20 and above_50 and sma_20_slope > 0 and sma_50_slope > 0:
            return "bullish"
        elif not above_20 and not above_50 and sma_20_slope < 0 and sma_50_slope < 0:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        price = data['close']
        
        # Rate of Change
        roc_5 = (price.iloc[-1] / price.iloc[-5] - 1) * 100
        roc_10 = (price.iloc[-1] / price.iloc[-10] - 1) * 100
        
        # RSI momentum
        rsi = self._calculate_rsi(price)
        rsi_momentum = (rsi.iloc[-1] - 50) / 50
        
        return (roc_5 + roc_10) / 20 + rsi_momentum
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate normalized volatility"""
        returns = data['close'].pct_change()
        return returns.rolling(20).std().iloc[-1]
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        high = data['high']
        low = data['low']
        
        support = low.rolling(20).min().iloc[-1]
        resistance = high.rolling(20).max().iloc[-1]
        
        return support, resistance
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_macd_signal(self, data: pd.DataFrame) -> str:
        """Get MACD signal"""
        price = data['close']
        
        exp1 = price.ewm(span=12).mean()
        exp2 = price.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return "bullish"
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return "bearish"
        else:
            return "neutral"
    
    def _create_signal(self, symbol: str, confluence: Dict, ml_confidence: float) -> TradeSignal:
        """Create trade signal"""
        confidence = confluence['score'] * 0.7 + ml_confidence * 0.3
        
        # These would be calculated by risk manager
        lot_size = 0.1
        stop_loss = 0.0
        take_profit = 0.0
        
        return TradeSignal(
            symbol=symbol,
            action=confluence['direction'],
            confidence=confidence,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.strategy_name,
            timeframe="H4/H1/M15",
            rationale=confluence['rationale']
        )