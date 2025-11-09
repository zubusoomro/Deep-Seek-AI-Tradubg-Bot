import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from ..core.trading_bot import TradeSignal

@dataclass
class SMCLevels:
    supply_zones: List[Tuple[float, float]]
    demand_zones: List[Tuple[float, float]] 
    breaker_blocks: List[float]
    order_blocks: List[float]
    liquidity_levels: List[float]

class SMCStrategy:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('SMCStrategy')
        self.strategy_name = "smc_advanced"
    
    def generate_signal(self, symbol: str, multi_timeframe_data: Dict,
                       ml_confidence: float) -> Optional[TradeSignal]:
        """Generate SMC-based trading signals"""
        try:
            h4_data = multi_timeframe_data.get('H4')
            h1_data = multi_timeframe_data.get('H1')
            
            if h4_data is None or h1_data is None:
                return None
            
            # Calculate SMC levels
            h4_levels = self._calculate_smc_levels(h4_data)
            h1_levels = self._calculate_smc_levels(h1_data)
            
            # Get current price from H1 data
            current_price = h1_data['close'].iloc[-1]
            
            # Find trading opportunities
            signal = self._find_smc_opportunity(
                symbol, current_price, h4_levels, h1_levels, ml_confidence
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"SMC strategy error for {symbol}: {str(e)}")
            return None
    
    def _calculate_smc_levels(self, data: pd.DataFrame) -> SMCLevels:
        """Calculate Smart Money Concepts levels"""
        supply_zones = self._find_supply_zones(data)
        demand_zones = self._find_demand_zones(data)
        breaker_blocks = self._find_breaker_blocks(data)
        order_blocks = self._find_order_blocks(data)
        liquidity_levels = self._find_liquidity_levels(data)
        
        return SMCLevels(
            supply_zones=supply_zones,
            demand_zones=demand_zones,
            breaker_blocks=breaker_blocks,
            order_blocks=order_blocks,
            liquidity_levels=liquidity_levels
        )
    
    def _find_supply_zones(self, data: pd.DataFrame, lookback: int = 100) -> List[Tuple[float, float]]:
        """Find supply zones (where price rejected and moved down)"""
        supply_zones = []
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        for i in range(2, len(data) - 10):
            # Look for swing highs followed by significant downward moves
            if (highs.iloc[i] > highs.iloc[i-1] and 
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i+2]):
                
                # Check if price moved down significantly after this high
                future_lows = lows.iloc[i+1:i+11]
                if len(future_lows) > 0:
                    min_future_low = future_lows.min()
                    if min_future_low < highs.iloc[i] * 0.98:  # At least 2% drop
                        strength = (highs.iloc[i] - min_future_low) / highs.iloc[i]
                        supply_zones.append((highs.iloc[i], strength))
        
        # Return strongest zones
        supply_zones.sort(key=lambda x: x[1], reverse=True)
        return supply_zones[:10]
    
    def _find_demand_zones(self, data: pd.DataFrame, lookback: int = 100) -> List[Tuple[float, float]]:
        """Find demand zones (where price rejected and moved up)"""
        demand_zones = []
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        for i in range(2, len(data) - 10):
            # Look for swing lows followed by significant upward moves
            if (lows.iloc[i] < lows.iloc[i-1] and 
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i+2]):
                
                # Check if price moved up significantly after this low
                future_highs = highs.iloc[i+1:i+11]
                if len(future_highs) > 0:
                    max_future_high = future_highs.max()
                    if max_future_high > lows.iloc[i] * 1.02:  # At least 2% rise
                        strength = (max_future_high - lows.iloc[i]) / lows.iloc[i]
                        demand_zones.append((lows.iloc[i], strength))
        
        # Return strongest zones
        demand_zones.sort(key=lambda x: x[1], reverse=True)
        return demand_zones[:10]
    
    def _find_breaker_blocks(self, data: pd.DataFrame) -> List[float]:
        """Find breaker blocks (significant break of structure)"""
        breaker_blocks = []
        highs = data['high']
        lows = data['low']
        
        for i in range(10, len(data) - 5):
            # Bullish breaker: price breaks above previous resistance
            if (highs.iloc[i] > highs.iloc[i-5:i].max() and 
                lows.iloc[i+1] > lows.iloc[i-5:i].max()):
                breaker_blocks.append(highs.iloc[i])
            
            # Bearish breaker: price breaks below previous support
            if (lows.iloc[i] < lows.iloc[i-5:i].min() and 
                highs.iloc[i+1] < highs.iloc[i-5:i].min()):
                breaker_blocks.append(lows.iloc[i])
        
        return breaker_blocks[-5:]
    
    def _find_order_blocks(self, data: pd.DataFrame) -> List[float]:
        """Find order blocks (institutional accumulation/distribution)"""
        order_blocks = []
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        for i in range(5, len(data) - 3):
            # Look for candles with small bodies and large wicks (indecision)
            body_size = abs(closes.iloc[i] - data['open'].iloc[i])
            total_range = highs.iloc[i] - lows.iloc[i]
            
            if total_range > 0 and body_size / total_range < 0.3:
                # This is an indecision candle, potential order block
                if closes.iloc[i] > data['open'].iloc[i]:
                    # Bullish order block
                    order_blocks.append(lows.iloc[i])
                else:
                    # Bearish order block  
                    order_blocks.append(highs.iloc[i])
        
        return order_blocks[-10:]
    
    def _find_liquidity_levels(self, data: pd.DataFrame) -> List[float]:
        """Find liquidity levels (recent highs and lows)"""
        highs = data['high']
        lows = data['low']
        
        # Recent swing highs (above market liquidity)
        recent_highs = highs.rolling(10).max().dropna().tolist()[-5:]
        
        # Recent swing lows (below market liquidity)
        recent_lows = lows.rolling(10).min().dropna().tolist()[-5:]
        
        return recent_highs + recent_lows
    
    def _find_smc_opportunity(self, symbol: str, current_price: float,
                            h4_levels: SMCLevels, h1_levels: SMCLevels,
                            ml_confidence: float) -> Optional[TradeSignal]:
        """Find SMC trading opportunity"""
        # Look for bullish setup
        bullish_signal = self._check_bullish_setup(current_price, h4_levels, h1_levels)
        if bullish_signal:
            return self._create_signal(
                symbol, "BUY", bullish_signal['confidence'] * ml_confidence,
                bullish_signal['rationale']
            )
        
        # Look for bearish setup
        bearish_signal = self._check_bearish_setup(current_price, h4_levels, h1_levels)
        if bearish_signal:
            return self._create_signal(
                symbol, "SELL", bearish_signal['confidence'] * ml_confidence, 
                bearish_signal['rationale']
            )
        
        return None
    
    def _check_bullish_setup(self, current_price: float, h4_levels: SMCLevels,
                           h1_levels: SMCLevels) -> Optional[Dict]:
        """Check for bullish SMC setup"""
        # Check if price is near demand zone
        nearest_demand = self._find_nearest_level(current_price, 
                                                h4_levels.demand_zones + h1_levels.demand_zones)
        if nearest_demand is None:
            return None
        
        demand_price, strength = nearest_demand
        price_distance = abs(current_price - demand_price) / current_price
        
        # Check if we're close to demand zone and it's strong
        if price_distance < 0.002 and strength > 0.03:  # Within 0.2% and strong zone
            confidence = 0.7 - (price_distance * 100)
            confidence *= strength * 10  # Adjust for zone strength
            
            # Check for additional confluence
            if self._has_liquidity_above(current_price, h4_levels.liquidity_levels):
                confidence *= 1.1
            
            return {
                'confidence': min(confidence, 0.9),
                'rationale': f"Bullish SMC - Price at demand zone (strength: {strength:.3f})"
            }
        
        return None
    
    def _check_bearish_setup(self, current_price: float, h4_levels: SMCLevels,
                           h1_levels: SMCLevels) -> Optional[Dict]:
        """Check for bearish SMC setup"""
        # Check if price is near supply zone
        nearest_supply = self._find_nearest_level(current_price,
                                                h4_levels.supply_zones + h1_levels.supply_zones)
        if nearest_supply is None:
            return None
        
        supply_price, strength = nearest_supply
        price_distance = abs(current_price - supply_price) / current_price
        
        # Check if we're close to supply zone and it's strong
        if price_distance < 0.002 and strength > 0.03:  # Within 0.2% and strong zone
            confidence = 0.7 - (price_distance * 100)
            confidence *= strength * 10  # Adjust for zone strength
            
            # Check for additional confluence
            if self._has_liquidity_below(current_price, h4_levels.liquidity_levels):
                confidence *= 1.1
            
            return {
                'confidence': min(confidence, 0.9),
                'rationale': f"Bearish SMC - Price at supply zone (strength: {strength:.3f})"
            }
        
        return None
    
    def _find_nearest_level(self, current_price: float, 
                          levels: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Find nearest SMC level to current price"""
        if not levels:
            return None
        
        nearest = min(levels, key=lambda x: abs(x[0] - current_price))
        return nearest
    
    def _has_liquidity_above(self, current_price: float, liquidity_levels: List[float]) -> bool:
        """Check if there's liquidity above current price"""
        return any(level > current_price for level in liquidity_levels)
    
    def _has_liquidity_below(self, current_price: float, liquidity_levels: List[float]) -> bool:
        """Check if there's liquidity below current price"""
        return any(level < current_price for level in liquidity_levels)
    
    def _create_signal(self, symbol: str, action: str, confidence: float,
                      rationale: str) -> TradeSignal:
        """Create trade signal from SMC analysis"""
        # These would be calculated by risk manager
        lot_size = 0.1
        stop_loss = 0.0
        take_profit = 0.0
        
        return TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.strategy_name,
            timeframe="H4/H1",
            rationale=rationale
        )