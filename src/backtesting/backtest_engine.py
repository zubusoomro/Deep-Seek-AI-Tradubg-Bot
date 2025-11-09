# src/backtesting/backtest_engine.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.config_manager import ConfigManager
from ..data.gold_data_fetcher import GoldDataFetcher
from ..features.gold_feature_engineer import GoldFeatureEngineer
from ..ml.gold_predictor import GoldPredictor
from ..strategies.gold_momentum import GoldMomentumStrategy, TradeSignal
from ..risk.risk_manager import RiskManager
from ..trading.position_manager import PositionManager

class BacktestEngine:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger('BacktestEngine')
        
        # Initialize components (same as live bot)
        self.data_fetcher = GoldDataFetcher()
        self.feature_engineer = GoldFeatureEngineer()
        self.predictor = GoldPredictor(self.config)
        self.strategy = GoldMomentumStrategy(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Backtest-specific attributes
        self.results = {}
        self.trades = []
        self.equity_curve = []
        self.current_equity = 10000  # Starting equity
        self.initial_equity = 10000
        
        # Performance metrics
        self.metrics = {}
    
    def run_backtest(self, historical_data: pd.DataFrame, 
                    initial_equity: float = 10000) -> Dict[str, Any]:
        """Run complete backtest on historical data"""
        self.logger.info(f"Starting backtest with {len(historical_data)} bars")
        
        self.current_equity = initial_equity
        self.initial_equity = initial_equity
        self.equity_curve = [initial_equity]
        self.trades = []
        
        # Load ML models
        self.predictor.load_models("models")
        
        # Track open positions
        open_positions = {}
        position_id = 1
        
        # Main backtest loop
        for i in range(50, len(historical_data)):  # Start from 50 to have enough data for features
            current_time = historical_data.index[i]
            current_data = historical_data.iloc[:i+1].copy()
            
            try:
                # 1. Manage existing positions
                open_positions = self._manage_positions(open_positions, current_data, current_time)
                
                # 2. Check if we should analyze for new trade (respect trading sessions)
                if not self._should_analyze(current_time):
                    self.equity_curve.append(self.current_equity)
                    continue
                
                # 3. Generate trading decision
                decision = self._get_trading_decision(current_data)
                
                # 4. Check risk management
                if decision['action'] != 'HOLD':
                    risk_check = self.risk_manager.can_open_trade({
                        'signal': TradeSignal(decision['action']),
                        'confidence': decision['confidence']
                    })
                    
                    if risk_check['allowed']:
                        # 5. Execute trade
                        position = self._execute_trade(
                            decision, current_data, current_time, position_id
                        )
                        if position:
                            open_positions[position_id] = position
                            position_id += 1
                
                # Update equity curve
                self.equity_curve.append(self.current_equity)
                
            except Exception as e:
                self.logger.error(f"Error at {current_time}: {e}")
                self.equity_curve.append(self.current_equity)
        
        # Close any remaining positions
        for pos_id, position in open_positions.items():
            self._close_position(pos_id, position, historical_data.iloc[-1], 
                               historical_data.index[-1], "End of backtest")
        
        # Calculate performance metrics
        self._calculate_metrics(historical_data)
        
        self.logger.info("Backtest completed successfully")
        return self._get_results()
    
    def _should_analyze(self, current_time: datetime) -> bool:
        """Check if we should analyze at current time (respect trading sessions)"""
        current_time_only = current_time.time()
        
        # London session: 08:00-16:00
        london_start = datetime.strptime("08:00", "%H:%M").time()
        london_end = datetime.strptime("16:00", "%H:%M").time()
        
        # NY session: 13:00-21:00  
        ny_start = datetime.strptime("13:00", "%H:%M").time()
        ny_end = datetime.strptime("21:00", "%H:%M").time()
        
        in_london = london_start <= current_time_only <= london_end
        in_ny = ny_start <= current_time_only <= ny_end
        
        return in_london or in_ny
    
    def _get_trading_decision(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get trading decision for current data"""
        try:
            # Generate features
            features = self.feature_engineer.generate_features(data)
            if features.empty:
                return {"action": "HOLD", "reason": "No features", "confidence": 0.0}
            
            # Get ML prediction
            ml_prediction, ml_confidence = self.predictor.predict(features)
            
            # Generate trading signal
            signal = self.strategy.generate_signal(data, ml_prediction, ml_confidence)
            
            return {
                "action": signal["signal"].value,
                "confidence": signal["confidence"],
                "position_size": signal.get("position_size", 0.1),
                "stop_loss": signal.get("stop_loss", 0),
                "take_profit": signal.get("take_profit", 0),
                "reason": signal.get("reason", ""),
                "timestamp": data.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {e}")
            return {"action": "HOLD", "reason": f"Error: {str(e)}", "confidence": 0.0}
    
    def _execute_trade(self, decision: Dict, data: pd.DataFrame, 
                      timestamp: datetime, position_id: int) -> Optional[Dict]:
        """Execute a trade in backtest"""
        try:
            current_price = data['close'].iloc[-1]
            spread = 0.0002  # 2 pips spread for Gold
            
            # Calculate entry price with spread
            if decision['action'] == 'BUY':
                entry_price = current_price + spread / 2
            else:  # SELL
                entry_price = current_price - spread / 2
            
            # Calculate position value
            risk_per_trade = self.config.get_risk_config().get('risk_per_trade', 0.005)
            risk_amount = self.current_equity * risk_per_trade * decision['confidence']
            position_size = decision['position_size']
            
            # Create position
            position = {
                'id': position_id,
                'direction': decision['action'],
                'entry_price': entry_price,
                'current_price': entry_price,
                'size': position_size,
                'stop_loss': decision['stop_loss'],
                'take_profit': decision['take_profit'],
                'entry_time': timestamp,
                'status': 'open',
                'max_profit': 0,
                'breakeven_triggered': False,
                'partial_profits_taken': 0
            }
            
            # Record trade
            trade_record = {
                'position_id': position_id,
                'direction': decision['action'],
                'entry_price': entry_price,
                'size': position_size,
                'entry_time': timestamp,
                'stop_loss': decision['stop_loss'],
                'take_profit': decision['take_profit'],
                'confidence': decision['confidence'],
                'reason': decision['reason']
            }
            
            self.trades.append(trade_record)
            self.risk_manager.record_trade_open(trade_record)
            
            self.logger.info(f"Backtest: Opened {decision['action']} position {position_id} at {entry_price:.5f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _manage_positions(self, open_positions: Dict, data: pd.DataFrame, 
                         current_time: datetime) -> Dict:
        """Manage open positions (check for exits, breakeven, partial profits)"""
        current_price = data['close'].iloc[-1]
        positions_to_remove = []
        
        for pos_id, position in open_positions.items():
            if position['status'] != 'open':
                positions_to_remove.append(pos_id)
                continue
            
            # Update current price
            position['current_price'] = current_price
            
            # Calculate current profit
            if position['direction'] == 'BUY':
                profit_pips = (current_price - position['entry_price']) * 10000  # Simplified to pips
                stop_loss_hit = current_price <= position['stop_loss']
                take_profit_hit = current_price >= position['take_profit']
            else:  # SELL
                profit_pips = (position['entry_price'] - current_price) * 10000
                stop_loss_hit = current_price >= position['stop_loss']
                take_profit_hit = current_price <= position['take_profit']
            
            # Update max profit
            position['max_profit'] = max(position['max_profit'], profit_pips)
            
            # Check breakeven
            if not position['breakeven_triggered'] and profit_pips >= 10:  # 10 pips profit
                position['stop_loss'] = position['entry_price']
                position['breakeven_triggered'] = True
                self.logger.debug(f"Position {pos_id} moved to breakeven")
            
            # Check partial profits (simplified)
            partial_levels = [
                (15, 0.5),  # 15 pips, close 50%
                (25, 0.25), # 25 pips, close 25%
                (40, 0.25)  # 40 pips, close 25%
            ]
            
            for level_pips, close_ratio in partial_levels:
                if (profit_pips >= level_pips and 
                    position['partial_profits_taken'] < len(partial_levels)):
                    
                    # Close partial position
                    partial_pnl = profit_pips * close_ratio * position['size'] * 10  # Simplified PnL
                    self.current_equity += partial_pnl
                    
                    position['size'] *= (1 - close_ratio)
                    position['partial_profits_taken'] += 1
                    
                    self.logger.debug(f"Position {pos_id} partial close: {close_ratio*100}% at {level_pips} pips")
            
            # Check for exit conditions
            if stop_loss_hit or take_profit_hit:
                exit_type = "SL" if stop_loss_hit else "TP"
                self._close_position(pos_id, position, data, current_time, exit_type)
                positions_to_remove.append(pos_id)
        
        # Remove closed positions
        for pos_id in positions_to_remove:
            if pos_id in open_positions:
                del open_positions[pos_id]
        
        return open_positions
    
    def _close_position(self, pos_id: int, position: Dict, data: pd.DataFrame,
                       timestamp: datetime, exit_type: str):
        """Close a position"""
        try:
            current_price = data['close'].iloc[-1]
            spread = 0.0002
            
            # Calculate exit price with spread
            if position['direction'] == 'BUY':
                exit_price = current_price - spread / 2
                pnl_pips = (exit_price - position['entry_price']) * 10000
            else:  # SELL
                exit_price = current_price + spread / 2
                pnl_pips = (position['entry_price'] - exit_price) * 10000
            
            # Calculate PnL in account currency
            pnl = pnl_pips * position['size'] * 10  # Simplified calculation
            
            # Update equity
            self.current_equity += pnl
            
            # Record trade close
            for trade in self.trades:
                if trade.get('position_id') == pos_id and 'exit_time' not in trade:
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = timestamp
                    trade['pnl'] = pnl
                    trade['exit_type'] = exit_type
                    break
            
            self.risk_manager.record_trade_close(pos_id, exit_price, pnl)
            
            position['status'] = 'closed'
            
            self.logger.info(f"Backtest: Closed position {pos_id} at {exit_price:.5f}, PnL: {pnl:.2f} ({exit_type})")
            
        except Exception as e:
            self.logger.error(f"Error closing position {pos_id}: {e}")
    
    def _calculate_metrics(self, data: pd.DataFrame):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            self.logger.warning("No trades executed during backtest")
            return
        
        # Filter completed trades
        completed_trades = [t for t in self.trades if 'pnl' in t]
        
        if not completed_trades:
            return
        
        # Basic metrics
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in completed_trades)
        win_rate = len(winning_trades) / total_trades
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        self.metrics = {
            'initial_equity': self.initial_equity,
            'final_equity': self.current_equity,
            'total_return': (self.current_equity - self.initial_equity) / self.initial_equity,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_pnl': total_pnl / total_trades,
            'average_win': gross_profit / len(winning_trades) if winning_trades else 0,
            'average_loss': gross_loss / len(losing_trades) if losing_trades else 0,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': (win_rate * (gross_profit/len(winning_trades)) + 
                          (1-win_rate) * (-gross_loss/len(losing_trades))) if winning_trades and losing_trades else 0
        }
    
    def _get_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        return {
            'metrics': self.metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'initial_equity': self.initial_equity,
            'final_equity': self.current_equity
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.metrics:
            return "No backtest results available"
        
        report = f"""
=== GOLD SCALPING BOT BACKTEST REPORT ===

PERFORMANCE SUMMARY:
-------------------
Initial Equity: ${self.metrics['initial_equity']:,.2f}
Final Equity: ${self.metrics['final_equity']:,.2f}
Total Return: {self.metrics['total_return']:.2%}
Total PnL: ${self.metrics['total_pnl']:,.2f}

TRADING STATISTICS:
------------------
Total Trades: {self.metrics['total_trades']}
Winning Trades: {self.metrics['winning_trades']}
Losing Trades: {self.metrics['losing_trades']}
Win Rate: {self.metrics['win_rate']:.2%}
Profit Factor: {self.metrics['profit_factor']:.2f}

RISK METRICS:
------------
Max Drawdown: {self.metrics['max_drawdown']:.2%}
Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
Average PnL: ${self.metrics['average_pnl']:.2f}
Average Win: ${self.metrics['average_win']:.2f}
Average Loss: ${self.metrics['average_loss']:.2f}
Expectancy: ${self.metrics['expectancy']:.2f}

TRADE DISTRIBUTION:
------------------
"""
        
        # Add trade distribution
        if self.trades:
            pnl_values = [t['pnl'] for t in self.trades if 'pnl' in t]
            report += f"Best Trade: ${max(pnl_values):.2f}\n"
            report += f"Worst Trade: ${min(pnl_values):.2f}\n"
        
        return report
    
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        if not self.equity_curve:
            self.logger.warning("No equity curve data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        ax1.plot(self.equity_curve, label='Equity', linewidth=2)
        ax1.axhline(y=self.initial_equity, color='r', linestyle='--', label='Initial Equity')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        ax2.fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # PnL distribution
        if self.trades:
            pnl_values = [t['pnl'] for t in self.trades if 'pnl' in t]
            ax3.hist(pnl_values, bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--')
            ax3.set_title('PnL Distribution')
            ax3.set_xlabel('PnL ($)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True)
        
        # Monthly returns
        if len(self.equity_curve) > 30:
            # This would require timestamps for proper monthly calculation
            # Simplified version
            monthly_returns = []
            for i in range(30, len(self.equity_curve), 30):
                ret = (self.equity_curve[i] - self.equity_curve[i-30]) / self.equity_curve[i-30]
                monthly_returns.append(ret)
            
            ax4.bar(range(len(monthly_returns)), monthly_returns, alpha=0.7)
            ax4.set_title('Monthly Returns')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Return (%)')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()