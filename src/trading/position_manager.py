# src/trading/position_manager.py
import MetaTrader5 as mt5
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class ExitType(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    BREAKEVEN = "BREAKEVEN"
    PARTIAL_PROFIT = "PARTIAL_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"

class PositionManager:
    def __init__(self, config_manager, risk_manager):
        self.config = config_manager
        self.risk_manager = risk_manager
        self.logger = logging.getLogger('PositionManager')
        
        self.active_positions = {}
        self.position_management_config = config_manager.get_position_management_config()
    
    def open_position(self, signal: Dict[str, Any]) -> Optional[int]:
        """Open a new position based on signal"""
        try:
            # Check risk management first
            risk_check = self.risk_manager.can_open_trade(signal)
            if not risk_check["allowed"]:
                self.logger.warning(f"Cannot open position: {risk_check['reason']}")
                return None
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if signal['signal'].value == "BUY" else mt5.ORDER_TYPE_SELL
            symbol = self.config.get_gold_config().get('symbol', 'XAUUSD')
            
            order_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': signal['position_size'],
                'type': order_type,
                'price': mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
                'sl': signal['stop_loss'],
                'tp': signal['take_profit'],
                'deviation': 10,
                'magic': 2023,
                'comment': 'Gold Momentum Bot',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(order_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode}")
                return None
            
            # Record the trade
            trade_data = {
                'ticket': result.order,
                'symbol': symbol,
                'direction': signal['signal'].value,
                'volume': signal['position_size'],
                'open_price': result.price,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'open_time': datetime.now(),
                'magic': 2023
            }
            
            self.risk_manager.record_trade_open(trade_data)
            
            # Track position for management
            self.active_positions[result.order] = {
                **trade_data,
                'initial_stop_loss': signal['stop_loss'],
                'breakeven_triggered': False,
                'partial_profits_taken': 0,
                'max_profit_reached': result.price
            }
            
            self.logger.info(f"Position opened: Ticket {result.order}, {signal['signal'].value} {signal['position_size']} lots")
            return result.order
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return None
    
    def manage_open_positions(self, current_data: pd.DataFrame):
        """Manage all open positions (breakeven, partial profits, trailing stops)"""
        try:
            current_price = current_data['close'].iloc[-1]
            
            for ticket, position in list(self.active_positions.items()):
                if not self._position_still_open(ticket):
                    continue
                
                # Calculate current profit in pips
                current_profit = self._calculate_current_profit(position, current_price)
                
                # Update max profit reached
                if current_profit > position.get('max_profit_reached', 0):
                    self.active_positions[ticket]['max_profit_reached'] = current_profit
                
                # Check for breakeven trigger
                if not position['breakeven_triggered']:
                    self._check_breakeven_trigger(ticket, position, current_profit, current_price)
                
                # Check for partial profit taking
                self._check_partial_profits(ticket, position, current_profit, current_price)
                
                # Check for trailing stop
                self._check_trailing_stop(ticket, position, current_profit, current_price)
                    
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
    
    def _position_still_open(self, ticket: int) -> bool:
        """Check if position is still open"""
        position = mt5.positions_get(ticket=ticket)
        return position is not None and len(position) > 0
    
    def _calculate_current_profit(self, position: Dict[str, Any], current_price: float) -> float:
        """Calculate current profit in account currency"""
        if position['direction'] == 'BUY':
            return current_price - position['open_price']
        else:  # SELL
            return position['open_price'] - current_price
    
    def _check_breakeven_trigger(self, ticket: int, position: Dict[str, Any], current_profit: float, current_price: float):
        """Check and trigger breakeven move"""
        breakeven_config = self.position_management_config.get('breakeven_trigger', 1.0)
        
        # Calculate risk (distance from open to initial stop)
        if position['direction'] == 'BUY':
            initial_risk = position['open_price'] - position['initial_stop_loss']
        else:
            initial_risk = position['initial_stop_loss'] - position['open_price']
        
        # Check if profit reached breakeven trigger (e.g., 1R)
        if current_profit >= (initial_risk * breakeven_config):
            # Move stop loss to breakeven
            new_stop = position['open_price']
            
            if self._modify_position(ticket, new_stop, position['take_profit']):
                self.active_positions[ticket]['breakeven_triggered'] = True
                self.active_positions[ticket]['stop_loss'] = new_stop
                self.logger.info(f"Breakeven triggered for ticket {ticket}")
    
    def _check_partial_profits(self, ticket: int, position: Dict[str, Any], current_profit: float, current_price: float):
        """Check and execute partial profit taking"""
        partial_configs = self.position_management_config.get('partial_profit_levels', [])
        taken_already = position.get('partial_profits_taken', 0)
        
        if taken_already >= len(partial_configs):
            return  # All partial profits already taken
        
        # Calculate risk (distance from open to initial stop)
        if position['direction'] == 'BUY':
            initial_risk = position['open_price'] - position['initial_stop_loss']
        else:
            initial_risk = position['initial_stop_loss'] - position['open_price']
        
        next_level = partial_configs[taken_already]
        profit_ratio = next_level['profit_ratio']
        close_percent = next_level['close_percent']
        
        # Check if we reached the profit level for next partial
        if current_profit >= (initial_risk * profit_ratio):
            # Close partial position
            close_volume = position['volume'] * (close_percent / 100)
            
            if self._close_partial_position(ticket, close_volume, current_price):
                self.active_positions[ticket]['partial_profits_taken'] += 1
                self.active_positions[ticket]['volume'] -= close_volume
                
                # Record partial profit
                partial_pnl = (current_profit / position['open_price']) * close_volume * 100000  # Simplified PnL calc
                self.risk_manager.record_trade_close(ticket, current_price, partial_pnl)
                
                self.logger.info(f"Partial profit taken for ticket {ticket}: {close_percent}% at {profit_ratio}R")
    
    def _check_trailing_stop(self, ticket: int, position: Dict[str, Any], current_profit: float, current_price: float):
        """Check and update trailing stop"""
        if not self.position_management_config.get('trailing_stop', False):
            return
        
        activation_ratio = self.position_management_config.get('trailing_activation', 1.5)
        
        # Calculate risk
        if position['direction'] == 'BUY':
            initial_risk = position['open_price'] - position['initial_stop_loss']
        else:
            initial_risk = position['initial_stop_loss'] - position['open_price']
        
        # Check if trailing stop should be activated
        if current_profit >= (initial_risk * activation_ratio):
            # Calculate new trailing stop
            trail_distance = initial_risk * 0.5  # Trail at 0.5R behind current price
            
            if position['direction'] == 'BUY':
                new_stop = current_price - trail_distance
            else:
                new_stop = current_price + trail_distance
            
            # Only move stop if it's better than current stop
            if (position['direction'] == 'BUY' and new_stop > position['stop_loss']) or \
               (position['direction'] == 'SELL' and new_stop < position['stop_loss']):
                
                if self._modify_position(ticket, new_stop, position['take_profit']):
                    self.active_positions[ticket]['stop_loss'] = new_stop
                    self.logger.info(f"Trailing stop updated for ticket {ticket}: {new_stop:.5f}")
    
    def _modify_position(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify position's stop loss and take profit"""
        try:
            modify_request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'position': ticket,
                'sl': new_sl,
                'tp': new_tp,
                'magic': 2023
            }
            
            result = mt5.order_send(modify_request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {e}")
            return False
    
    def _close_partial_position(self, ticket: int, volume: float, price: float) -> bool:
        """Close partial position"""
        try:
            position = mt5.positions_get(ticket=ticket)[0]
            
            if position.volume <= volume:
                return False  # Can't close more than we have
            
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            close_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': position.symbol,
                'volume': volume,
                'type': close_type,
                'price': price,
                'deviation': 10,
                'magic': 2023,
                'comment': 'Partial close',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            self.logger.error(f"Error closing partial position {ticket}: {e}")
            return False
    
    def close_all_positions(self):
        """Close all open positions (emergency stop)"""
        try:
            positions = mt5.positions_get()
            for position in positions:
                if position.magic == 2023:  # Our bot's magic number
                    self._close_position(position.ticket)
            
            self.logger.info("All positions closed")
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    def _close_position(self, ticket: int) -> bool:
        """Close a single position"""
        try:
            position = mt5.positions_get(ticket=ticket)[0]
            
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            current_price = mt5.symbol_info_tick(position.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask
            
            close_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': position.symbol,
                'volume': position.volume,
                'type': close_type,
                'price': current_price,
                'deviation': 10,
                'magic': 2023,
                'comment': 'Bot close',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Record the close
                pnl = result.profit
                self.risk_manager.record_trade_close(ticket, current_price, pnl)
                
                # Remove from active positions
                if ticket in self.active_positions:
                    del self.active_positions[ticket]
                
                self.logger.info(f"Position {ticket} closed, PnL: {pnl:.2f}")
                return True
            else:
                self.logger.error(f"Failed to close position {ticket}: {result.retcode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return False