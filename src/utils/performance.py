import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from enum import Enum

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class TradeRecord:
    id: str
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    swap: float
    profit: float
    strategy: str
    confidence: float
    stop_loss: float
    take_profit: float
    exit_reason: str

class PerformanceTracker:
    """Comprehensive performance tracking and analytics"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('PerformanceTracker')
        
        # Trade records
        self.trades: List[TradeRecord] = []
        self.active_trades: Dict[str, TradeRecord] = {}
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'recovery_factor': 0.0,
            'expectancy': 0.0
        }
        
        # Equity curve
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Strategy performance
        self.strategy_performance = {}
        
        # Load historical data if exists
        self._load_performance_data()
    
    def record_trade_entry(self, trade_id: str, symbol: str, direction: TradeDirection,
                         entry_time: datetime, entry_price: float, quantity: float,
                         strategy: str, confidence: float, stop_loss: float, 
                         take_profit: float, commission: float = 0.0) -> bool:
        """Record a new trade entry"""
        try:
            trade = TradeRecord(
                id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                commission=commission,
                swap=0.0,
                profit=0.0,
                strategy=strategy,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_reason=""
            )
            
            self.active_trades[trade_id] = trade
            self.logger.info(f"Recorded trade entry: {trade_id} - {symbol} {direction.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording trade entry: {str(e)}")
            return False
    
    def record_trade_exit(self, trade_id: str, exit_time: datetime, exit_price: float,
                         profit: float, exit_reason: str, commission: float = 0.0,
                         swap: float = 0.0) -> bool:
        """Record trade exit and calculate performance"""
        try:
            if trade_id not in self.active_trades:
                self.logger.error(f"Trade {trade_id} not found in active trades")
                return False
            
            trade = self.active_trades[trade_id]
            
            # Update trade record
            trade.exit_time = exit_time
            trade.exit_price = exit_price
            trade.profit = profit
            trade.commission += commission
            trade.swap = swap
            trade.exit_reason = exit_reason
            
            # Move to completed trades
            self.trades.append(trade)
            del self.active_trades[trade_id]
            
            # Update performance metrics
            self._update_performance_metrics(trade)
            
            # Update equity curve
            self._update_equity_curve(trade)
            
            # Update strategy performance
            self._update_strategy_performance(trade)
            
            self.logger.info(f"Recorded trade exit: {trade_id} - Profit: ${profit:.2f}")
            
            # Save performance data
            self._save_performance_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording trade exit: {str(e)}")
            return False
    
    def _update_performance_metrics(self, trade: TradeRecord):
        """Update overall performance metrics"""
        self.metrics['total_trades'] += 1
        self.metrics['net_profit'] += trade.profit
        
        if trade.profit > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['total_profit'] += trade.profit
            self.metrics['largest_win'] = max(self.metrics['largest_win'], trade.profit)
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['total_loss'] += abs(trade.profit)
            self.metrics['largest_loss'] = min(self.metrics['largest_loss'], trade.profit)
        
        # Calculate derived metrics
        if self.metrics['winning_trades'] > 0:
            self.metrics['average_win'] = self.metrics['total_profit'] / self.metrics['winning_trades']
        
        if self.metrics['losing_trades'] > 0:
            self.metrics['average_loss'] = self.metrics['total_loss'] / self.metrics['losing_trades']
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        if self.metrics['total_loss'] > 0:
            self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss']
        
        # Calculate expectancy
        if self.metrics['total_trades'] > 0:
            win_prob = self.metrics['win_rate']
            loss_prob = 1 - win_prob
            avg_win = self.metrics['average_win']
            avg_loss = self.metrics['average_loss']
            self.metrics['expectancy'] = (win_prob * avg_win) - (loss_prob * avg_loss)
    
    def _update_equity_curve(self, trade: TradeRecord):
        """Update equity curve and drawdown calculations"""
        equity_point = {
            'timestamp': trade.exit_time,
            'equity': self.metrics['net_profit'],
            'trade_profit': trade.profit
        }
        
        self.equity_curve.append(equity_point)
        
        # Calculate drawdown
        if len(self.equity_curve) > 1:
            peak_equity = max(point['equity'] for point in self.equity_curve)
            current_equity = self.equity_curve[-1]['equity']
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            
            drawdown_point = {
                'timestamp': trade.exit_time,
                'drawdown': drawdown,
                'peak_equity': peak_equity,
                'current_equity': current_equity
            }
            
            self.drawdown_curve.append(drawdown_point)
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
    
    def _update_strategy_performance(self, trade: TradeRecord):
        """Update strategy-specific performance metrics"""
        strategy = trade.strategy
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0,
                'net_profit': 0.0,
                'win_rate': 0.0,
                'average_profit': 0.0,
                'total_confidence': 0.0,
                'average_confidence': 0.0
            }
        
        perf = self.strategy_performance[strategy]
        perf['total_trades'] += 1
        perf['net_profit'] += trade.profit
        perf['total_profit'] += max(0, trade.profit)
        perf['total_confidence'] += trade.confidence
        
        if trade.profit > 0:
            perf['winning_trades'] += 1
        
        # Update derived metrics
        perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
        perf['average_profit'] = perf['net_profit'] / perf['total_trades']
        perf['average_confidence'] = perf['total_confidence'] / perf['total_trades']
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'summary_metrics': self.metrics.copy(),
            'strategy_performance': self.strategy_performance.copy(),
            'recent_trades': self.get_recent_trades(20),
            'time_period_analysis': self.get_time_period_analysis(),
            'symbol_analysis': self.get_symbol_analysis(),
            'confidence_analysis': self.get_confidence_analysis(),
            'risk_metrics': self.calculate_risk_metrics()
        }
        
        return report
    
    def get_recent_trades(self, count: int = 10) -> List[Dict]:
        """Get recent trades with details"""
        recent_trades = sorted(self.trades, key=lambda x: x.exit_time or x.entry_time, 
                              reverse=True)[:count]
        
        return [{
            'id': trade.id,
            'symbol': trade.symbol,
            'direction': trade.direction.value,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'profit': trade.profit,
            'strategy': trade.strategy,
            'confidence': trade.confidence,
            'exit_reason': trade.exit_reason
        } for trade in recent_trades]
    
    def get_time_period_analysis(self) -> Dict:
        """Analyze performance by time periods"""
        if not self.trades:
            return {}
        
        # Group by hour of day
        hourly_performance = {}
        for trade in self.trades:
            if trade.exit_time:
                hour = trade.exit_time.hour
                if hour not in hourly_performance:
                    hourly_performance[hour] = {'trades': 0, 'profit': 0.0}
                hourly_performance[hour]['trades'] += 1
                hourly_performance[hour]['profit'] += trade.profit
        
        # Group by day of week
        daily_performance = {}
        for trade in self.trades:
            if trade.exit_time:
                day = trade.exit_time.strftime('%A')
                if day not in daily_performance:
                    daily_performance[day] = {'trades': 0, 'profit': 0.0}
                daily_performance[day]['trades'] += 1
                daily_performance[day]['profit'] += trade.profit
        
        return {
            'hourly': hourly_performance,
            'daily': daily_performance
        }
    
    def get_symbol_analysis(self) -> Dict:
        """Analyze performance by symbol"""
        symbol_performance = {}
        
        for trade in self.trades:
            symbol = trade.symbol
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0.0,
                    'net_profit': 0.0,
                    'win_rate': 0.0
                }
            
            perf = symbol_performance[symbol]
            perf['total_trades'] += 1
            perf['net_profit'] += trade.profit
            
            if trade.profit > 0:
                perf['winning_trades'] += 1
            
            perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
        
        return symbol_performance
    
    def get_confidence_analysis(self) -> Dict:
        """Analyze performance by confidence levels"""
        confidence_buckets = {
            'low': {'min': 0.0, 'max': 0.6, 'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
            'medium': {'min': 0.6, 'max': 0.8, 'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
            'high': {'min': 0.8, 'max': 1.0, 'trades': 0, 'profit': 0.0, 'win_rate': 0.0}
        }
        
        for trade in self.trades:
            for bucket_name, bucket in confidence_buckets.items():
                if bucket['min'] <= trade.confidence < bucket['max']:
                    bucket['trades'] += 1
                    bucket['profit'] += trade.profit
                    if trade.profit > 0:
                        bucket['win_rate'] = ((bucket.get('winning_trades', 0) + 1) / bucket['trades'])
                    else:
                        bucket['win_rate'] = (bucket.get('winning_trades', 0) / bucket['trades'])
                    break
        
        return confidence_buckets
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate advanced risk metrics"""
        if len(self.trades) < 2:
            return {}
        
        profits = [trade.profit for trade in self.trades]
        
        # Sharpe Ratio (simplified)
        avg_profit = np.mean(profits)
        std_profit = np.std(profits)
        sharpe_ratio = avg_profit / std_profit if std_profit > 0 else 0
        
        # Recovery Factor
        max_dd = self.metrics['max_drawdown']
        recovery_factor = self.metrics['net_profit'] / max_dd if max_dd > 0 else 0
        
        # Kelly Criterion
        win_rate = self.metrics['win_rate']
        avg_win = self.metrics['average_win']
        avg_loss = self.metrics['average_loss']
        
        if avg_loss > 0:
            kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        else:
            kelly = 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'recovery_factor': recovery_factor,
            'kelly_criterion': kelly,
            'profit_std_dev': std_profit,
            'profit_variance': np.var(profits),
            'value_at_risk_95': np.percentile(profits, 5)
        }
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            data = {
                'trades': [{
                    'id': trade.id,
                    'symbol': trade.symbol,
                    'direction': trade.direction.value,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'commission': trade.commission,
                    'swap': trade.swap,
                    'profit': trade.profit,
                    'strategy': trade.strategy,
                    'confidence': trade.confidence,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'exit_reason': trade.exit_reason
                } for trade in self.trades],
                'metrics': self.metrics,
                'equity_curve': self.equity_curve,
                'strategy_performance': self.strategy_performance
            }
            
            with open('performance_data.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")
    
    def _load_performance_data(self):
        """Load performance data from file"""
        try:
            if os.path.exists('performance_data.json'):
                with open('performance_data.json', 'r') as f:
                    data = json.load(f)
                
                # Load trades
                for trade_data in data.get('trades', []):
                    trade = TradeRecord(
                        id=trade_data['id'],
                        symbol=trade_data['symbol'],
                        direction=TradeDirection(trade_data['direction']),
                        entry_time=datetime.fromisoformat(trade_data['entry_time']),
                        exit_time=datetime.fromisoformat(trade_data['exit_time']) if trade_data['exit_time'] else None,
                        entry_price=trade_data['entry_price'],
                        exit_price=trade_data['exit_price'],
                        quantity=trade_data['quantity'],
                        commission=trade_data['commission'],
                        swap=trade_data['swap'],
                        profit=trade_data['profit'],
                        strategy=trade_data['strategy'],
                        confidence=trade_data['confidence'],
                        stop_loss=trade_data['stop_loss'],
                        take_profit=trade_data['take_profit'],
                        exit_reason=trade_data['exit_reason']
                    )
                    self.trades.append(trade)
                
                # Load metrics
                self.metrics.update(data.get('metrics', {}))
                self.equity_curve = data.get('equity_curve', [])
                self.strategy_performance = data.get('strategy_performance', {})
                
                self.logger.info(f"Loaded {len(self.trades)} historical trades")
                
        except Exception as e:
            self.logger.error(f"Error loading performance data: {str(e)}")
    
    def reset_performance_data(self):
        """Reset all performance data"""
        self.trades.clear()
        self.active_trades.clear()
        self.equity_curve.clear()
        self.drawdown_curve.clear()
        self.strategy_performance.clear()
        
        # Reset metrics
        for key in self.metrics:
            if isinstance(self.metrics[key], (int, float)):
                self.metrics[key] = 0
        
        self.logger.info("Performance data reset")