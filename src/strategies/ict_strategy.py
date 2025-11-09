import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from dataclasses import dataclass
from ..core.trading_bot import TradeSignal

@dataclass
class ICTLevels:
    order_blocks: list
    fair_value_gaps: list
    liquidity_levels: list
    market_structure: str

class ICTStrategy:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('ICTStrategy')
        self.strategy_name = "ict_smart_money"
    
    def generate_signal(self, symbol: str, multi_timeframe_data: Dict, 
                       ml_confidence: float) -> Optional[TradeSignal]:
        """Generate ICT-based trading signal"""
        try:
            # Analyze multiple timeframes for confluence
            h1_data = multi_timeframe_data.get('H1')
            h4_data = multi_timeframe_data.get('H4')
            m15_data = multi_timeframe_data.get('M15')
            
            if h1_data is None or h4_data is None:
                return None
            
            # Calculate ICT levels across timeframes
            h4_levels = self._calculate_ict_levels(h4_data)
            h1_levels = self._calculate_ict_levels(h1_data)
            m15_levels = self._calculate_ict_levels(m15_data) if m15_data is not None else None
            
            # Determine market structure
            market_structure = self._analyze_market_structure(h4_data, h1_data)
            
            # Find trading opportunities
            signal = self._find_ict_opportunity(
                symbol, h4_levels, h1_levels, m15_levels, market_structure, ml_confidence
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in ICT strategy for {symbol}: {str(e)}")
            return None
    
    def _calculate_ict_levels(self, data: pd.DataFrame) -> ICTLevels:
        """Calculate ICT key levels"""
        order_blocks = self._find_order_blocks(data)
        fair_value_gaps = self._find_fair_value_gaps(data)
        liquidity_levels = self._find_liquidity_levels(data)
        market_structure = self._determine_market_structure_single(data)
        
        return ICTLevels(
            order_blocks=order_blocks,
            fair_value_gaps=fair_value_gaps,
            liquidity_levels=liquidity_levels,
            market_structure=market_structure
        )
    
    def _find_order_blocks(self, data: pd.DataFrame, lookback: int = 20) -> list:
        """Find order blocks (significant swing points)"""
        order_blocks = []
        
        # Find swing highs and lows
        highs = data['high']
        lows = data['low']
        
        for i in range(2, len(data) - 2):
            # Swing high
            if (highs.iloc[i] > highs.iloc[i-1] and 
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i+2]):
                order_blocks.append(('resistance', highs.iloc[i], data.index[i]))
            
            # Swing low
            if (lows.iloc[i] < lows.iloc[i-1] and 
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i+2]):
                order_blocks.append(('support', lows.iloc[i], data.index[i]))
        
        # Return recent order blocks (last 10)
        return order_blocks[-10:] if order_blocks else []
    
    def _find_fair_value_gaps(self, data: pd.DataFrame) -> list:
        """Find Fair Value Gaps (FVGs)"""
        fvgs = []
        
        for i in range(1, len(data) - 1):
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            prev_high = data['high'].iloc[i-1]
            prev_low = data['low'].iloc[i-1]
            next_high = data['high'].iloc[i+1]
            next_low = data['low'].iloc[i+1]
            
            # Bullish FVG (current low > previous high)
            if current_low > prev_high and next_low > prev_high:
                fvgs.append(('bullish', prev_high, current_low, data.index[i]))
            
            # Bearish FVG (current high < previous low)
            if current_high < prev_low and next_high < prev_low:
                fvgs.append(('bearish', current_high, prev_low, data.index[i]))
        
        return fvgs[-5:] if fvgs else []
    
    def _find_liquidity_levels(self, data: pd.DataFrame) -> list:
        """Find liquidity levels (recent highs and lows)"""
        # Recent swing highs (liquidity above)
        recent_highs = data['high'].rolling(10).max().dropna().tolist()[-3:]
        
        # Recent swing lows (liquidity below)
        recent_lows = data['low'].rolling(10).min().dropna().tolist()[-3:]
        
        return recent_highs + recent_lows
    
    def _determine_market_structure_single(self, data: pd.DataFrame) -> str:
        """Determine market structure for single timeframe"""
        closes = data['close']
        
        # Check for higher highs/higher lows (uptrend)
        if (closes.iloc[-1] > closes.iloc[-5] and 
            data['low'].iloc[-1] > data['low'].iloc[-5]):
            return "bullish"
        
        # Check for lower highs/lower lows (downtrend)
        elif (closes.iloc[-1] < closes.iloc[-5] and 
              data['high'].iloc[-1] < data['high'].iloc[-5]):
            return "bearish"
        
        else:
            return "ranging"
    
    def _analyze_market_structure(self, higher_tf_data: pd.DataFrame, 
                                lower_tf_data: pd.DataFrame) -> str:
        """Analyze market structure across timeframes"""
        higher_tf_structure = self._determine_market_structure_single(higher_tf_data)
        lower_tf_structure = self._determine_market_structure_single(lower_tf_data)
        
        # Higher timeframe dominates
        if higher_tf_structure in ["bullish", "bearish"]:
            return higher_tf_structure
        else:
            return lower_tf_structure
    
    def _find_ict_opportunity(self, symbol: str, h4_levels: ICTLevels, 
                            h1_levels: ICTLevels, m15_levels: ICTLevels,
                            market_structure: str, ml_confidence: float) -> Optional[TradeSignal]:
        """Find ICT trading opportunity with confluence"""
        try:
            current_data = h1_levels  # Use H1 for current price
            if not current_data:
                return None
            
            # Get current price from the latest data point
            current_price = h1_levels.market_structure  # This needs to be actual price
            
            # Look for bullish setup
            bullish_signal = self._check_bullish_setup(
                current_price, h4_levels, h1_levels, m15_levels, market_structure
            )
            
            if bullish_signal:
                return self._create_signal(
                    symbol, "BUY", bullish_signal['confidence'] * ml_confidence,
                    bullish_signal['rationale']
                )
            
            # Look for bearish setup
            bearish_signal = self._check_bearish_setup(
                current_price, h4_levels, h1_levels, m15_levels, market_structure
            )
            
            if bearish_signal:
                return self._create_signal(
                    symbol, "SELL", bearish_signal['confidence'] * ml_confidence,
                    bearish_signal['rationale']
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ICT opportunity: {str(e)}")
            return None
    
    def _check_bullish_setup(self, current_price: float, h4_levels: ICTLevels,
                           h1_levels: ICTLevels, m15_levels: ICTLevels,
                           market_structure: str) -> Optional[Dict]:
        """Check for bullish ICT setup"""
        # Simplified bullish logic
        if market_structure != "bullish":
            return None
        
        # Check for order block confluence
        h4_bullish_blocks = [ob for ob in h4_levels.order_blocks if ob[0] == 'support']
        h1_bullish_blocks = [ob for ob in h1_levels.order_blocks if ob[0] == 'support']
        
        if not h4_bullish_blocks or not h1_bullish_blocks:
            return None
        
        # Check if price is near support level
        nearest_support = h1_bullish_blocks[-1][1]  # price level
        price_distance = abs(current_price - nearest_support) / current_price
        
        if price_distance < 0.002:  # Within 0.2% of support
            confidence = 0.7 - (price_distance * 100)  # Closer = higher confidence
            return {
                'confidence': max(confidence, 0.5),
                'rationale': f"Bullish ICT - Price at support level, market structure bullish"
            }
        
        return None
    
    def _check_bearish_setup(self, current_price: float, h4_levels: ICTLevels,
                           h1_levels: ICTLevels, m15_levels: ICTLevels,
                           market_structure: str) -> Optional[Dict]:
        """Check for bearish ICT setup"""
        if market_structure != "bearish":
            return None
        
        # Check for order block confluence
        h4_bearish_blocks = [ob for ob in h4_levels.order_blocks if ob[0] == 'resistance']
        h1_bearish_blocks = [ob for ob in h1_levels.order_blocks if ob[0] == 'resistance']
        
        if not h4_bearish_blocks or not h1_bearish_blocks:
            return None
        
        # Check if price is near resistance level
        nearest_resistance = h1_bearish_blocks[-1][1]  # price level
        price_distance = abs(current_price - nearest_resistance) / current_price
        
        if price_distance < 0.002:  # Within 0.2% of resistance
            confidence = 0.7 - (price_distance * 100)
            return {
                'confidence': max(confidence, 0.5),
                'rationale': f"Bearish ICT - Price at resistance level, market structure bearish"
            }
        
        return None
    
    def _create_signal(self, symbol: str, action: str, confidence: float, 
                      rationale: str) -> TradeSignal:
        """Create trade signal"""
        # These would be calculated based on current market conditions
        lot_size = 0.1  # Would be calculated by risk manager
        stop_loss = 0.0  # Would be calculated
        take_profit = 0.0  # Would be calculated
        
        return TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.strategy_name,
            timeframe="H1",
            rationale=rationale
        )