# src/risk/risk_manager.py
import pandas as pd
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('RiskManager')
        
        # Trade tracking
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
        
        self._load_risk_limits()
    
    def _load_risk_limits(self):
        """Load risk limits from configuration"""
        risk_config = self.config.get_risk_config()
        self.max_daily_trades = risk_config.get('max_daily_trades', 8)
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.02)  # 2%
        self.risk_per_trade = risk_config.get('risk_per_trade', 0.005)  # 0.5%
    
    def can_open_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Check if we can open a new trade based on risk limits"""
        try:
            # 1. Check daily trade limit
            if len(self.daily_trades) >= self.max_daily_trades:
                return {"allowed": False, "reason": "Daily trade limit reached"}
            
            # 2. Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                return {"allowed": False, "reason": "Daily loss limit reached"}
            
            # 3. Check if market is too volatile
            if self._is_market_too_volatile():
                return {"allowed": False, "reason": "Market too volatile"}
            
            # 4. Check if we have recent similar trades (avoid overexposure)
            if self._has_recent_similar_trades(signal):
                return {"allowed": False, "reason": "Recent similar trades"}
            
            return {"allowed": True, "reason": "Risk checks passed"}
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return {"allowed": False, "reason": f"Risk check error: {str(e)}"}
    
    def _is_market_too_volatile(self) -> bool:
        """Check if market volatility is too high for safe trading"""
        # This would use real volatility data
        # For now, return False (market is safe)
        return False
    
    def _has_recent_similar_trades(self, signal: Dict[str, Any]) -> bool:
        """Check if we have recent similar trades to avoid overexposure"""
        current_time = datetime.now()
        recent_threshold = current_time - timedelta(hours=1)  # Last hour
        
        recent_trades = [
            trade for trade in self.daily_trades 
            if trade['timestamp'] > recent_threshold and trade['direction'] == signal['signal'].value
        ]
        
        # If we have 2+ recent trades in same direction, avoid opening new one
        return len(recent_trades) >= 2
    
    def record_trade_open(self, trade_data: Dict[str, Any]):
        """Record when a trade is opened"""
        trade_record = {
            'ticket': trade_data.get('ticket'),
            'symbol': trade_data.get('symbol', 'XAUUSD'),
            'direction': trade_data.get('direction'),
            'volume': trade_data.get('volume'),
            'open_price': trade_data.get('open_price'),
            'stop_loss': trade_data.get('stop_loss'),
            'take_profit': trade_data.get('take_profit'),
            'timestamp': datetime.now(),
            'status': 'open'
        }
        
        self.daily_trades.append(trade_record)
        self.logger.info(f"Trade opened: {trade_record}")
    
    def record_trade_close(self, ticket: int, close_price: float, pnl: float):
        """Record when a trade is closed"""
        for trade in self.daily_trades:
            if trade.get('ticket') == ticket and trade['status'] == 'open':
                trade['close_price'] = close_price
                trade['pnl'] = pnl
                trade['close_time'] = datetime.now()
                trade['status'] = 'closed'
                
                self.daily_pnl += pnl
                self.logger.info(f"Trade closed: Ticket {ticket}, PnL: {pnl:.2f}")
                break
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics"""
        closed_trades = [t for t in self.daily_trades if t.get('status') == 'closed']
        open_trades = [t for t in self.daily_trades if t.get('status') == 'open']
        
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        return {
            'total_trades': len(self.daily_trades),
            'open_trades': len(open_trades),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'daily_pnl': self.daily_pnl,
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'max_daily_loss_remaining': self.max_daily_loss + min(0, self.daily_pnl)
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (should be called at start of new trading day)"""
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.logger.info("Daily statistics reset")