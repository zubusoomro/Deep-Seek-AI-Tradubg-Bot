import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

class PositionManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('PositionManager')
        self.active_positions = {}
    
    def open_position(self, symbol: str, action: str, lot_size: float,
                    stop_loss: float, take_profit: float, comment: str = "") -> bool:
        """Open a new trading position"""
        try:
            # Validate inputs
            if lot_size <= 0:
                self.logger.error(f"Invalid lot size: {lot_size}")
                return False
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 2024001,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
            
            self.logger.info(f"Position opened: {symbol} {action} {lot_size} lots "
                           f"SL: {stop_loss:.5f} TP: {take_profit:.5f}")
            
            # Track position
            self.active_positions[result.order] = {
                'symbol': symbol,
                'type': action,
                'volume': lot_size,
                'open_price': result.price,
                'sl': stop_loss,
                'tp': take_profit,
                'open_time': datetime.now()
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return False
    
    def close_position(self, ticket: int, percentage: float = 100.0) -> bool:
        """Close a position (fully or partially)"""
        try:
            if ticket not in self.active_positions:
                self.logger.error(f"Position not found: {ticket}")
                return False
            
            position = self.active_positions[ticket]
            
            if percentage == 100.0:
                # Close entire position
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": position['symbol'],
                    "volume": position['volume'],
                    "type": mt5.ORDER_TYPE_SELL if position['type'] == "BUY" else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(position['symbol']).bid if position['type'] == "BUY" else mt5.symbol_info_tick(position['symbol']).ask,
                    "deviation": 20,
                    "magic": 2024001,
                    "comment": "Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
            else:
                # Partial close
                close_volume = position['volume'] * (percentage / 100.0)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": position['symbol'],
                    "volume": close_volume,
                    "type": mt5.ORDER_TYPE_SELL if position['type'] == "BUY" else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(position['symbol']).bid if position['type'] == "BUY" else mt5.symbol_info_tick(position['symbol']).ask,
                    "deviation": 20,
                    "magic": 2024001,
                    "comment": f"Partial close {percentage}%",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close position failed: {result.retcode}")
                return False
            
            if percentage == 100.0:
                del self.active_positions[ticket]
                self.logger.info(f"Position closed: {ticket}")
            else:
                self.active_positions[ticket]['volume'] -= close_volume
                self.logger.info(f"Position partially closed: {ticket} - {percentage}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False
    
    def modify_position(self, ticket: int, stop_loss: float = None, 
                       take_profit: float = None) -> bool:
        """Modify position's stop loss or take profit"""
        try:
            if ticket not in self.active_positions:
                self.logger.error(f"Position not found for modification: {ticket}")
                return False
            
            position = self.active_positions[ticket]
            
            # Use current values if not provided
            if stop_loss is None:
                stop_loss = position['sl']
            if take_profit is None:
                take_profit = position['tp']
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": position['symbol'],
                "sl": stop_loss,
                "tp": take_profit,
                "magic": 2024001,
                "comment": "Modified SL/TP",
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Modify position failed: {result.retcode}")
                return False
            
            # Update local tracking
            self.active_positions[ticket]['sl'] = stop_loss
            self.active_positions[ticket]['tp'] = take_profit
            
            self.logger.info(f"Position modified: {ticket} SL: {stop_loss:.5f} TP: {take_profit:.5f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying position: {str(e)}")
            return False
    
    def get_active_positions(self) -> Dict:
        """Get all active positions"""
        return self.active_positions.copy()
    
    def update_trailing_stops(self):
        """Update trailing stops for all active positions"""
        try:
            for ticket, position in self.active_positions.items():
                current_price = mt5.symbol_info_tick(position['symbol']).bid if position['type'] == "BUY" else mt5.symbol_info_tick(position['symbol']).ask
                
                if position['type'] == "BUY":
                    # For long positions, trail stop loss below current price
                    trail_distance = position['open_price'] * 0.005  # 0.5% trail
                    new_sl = current_price - trail_distance
                    
                    if new_sl > position['sl']:
                        self.modify_position(ticket, stop_loss=new_sl)
                
                else:  # SELL
                    # For short positions, trail stop loss above current price
                    trail_distance = position['open_price'] * 0.005  # 0.5% trail
                    new_sl = current_price + trail_distance
                    
                    if new_sl < position['sl']:
                        self.modify_position(ticket, stop_loss=new_sl)
                        
        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {str(e)}")