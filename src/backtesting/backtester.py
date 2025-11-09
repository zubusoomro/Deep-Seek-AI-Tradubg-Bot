import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import MetaTrader5 as mt5
from ..data.mt5_data import MT5DataManager
from ..data.robust_data_fetcher import RobustDataFetcher

@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    spread: float = 0.0002  # Default spread
    commission: float = 0.0  # Commission per trade
    slippage: float = 0.0    # Slippage per trade

@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: float
    profit: float
    commission: float
    slippage: float
    net_profit: float
    strategy: str
    confidence: float

class Backtester:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config_manager, trading_bot):
        self.config = config_manager
        self.trading_bot = trading_bot
        self.logger = logging.getLogger('Backtester')
        
        # Backtest results
        self.trades: List[BacktestTrade] = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'expectancy': 0.0
        }
    
    def run_backtest(self, symbol: str, timeframe: str, 
                   config: BacktestConfig) -> Dict:
        """Run backtest for a symbol"""
        try:
            self.logger.info(f"Starting backtest for {symbol} from {config.start_date} to {config.end_date}")
            
            # Get historical data
            data = self._get_historical_data(symbol, timeframe, config)
            if data is None or len(data) == 0:
                self.logger.error(f"No historical data for {symbol}")
                return {}
            
            # Initialize tracking
            current_balance = config.initial_balance
            peak_balance = current_balance
            self.equity_curve = []
            self.trades = []
            
            # Simulate trading day by day
            current_date = config.start_date
            active_trades = {}
            
            while current_date <= config.end_date:
                # Skip weekends for forex
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue
                
                # Get data up to current date
                current_data = data[data.index <= current_date].copy()
                if len(current_data) < 100:  # Need sufficient data
                    current_date += timedelta(days=1)
                    continue
                
                # Check for trade exits (next day logic)
                self._check_trade_exits(active_trades, current_data, current_date, config)
                
                # Generate trading signals for current day
                signals = self._generate_signals(symbol, current_data, current_date)
                
                # Execute signals
                for signal in signals:
                    if signal.action in ['BUY', 'SELL']:
                        trade = self._execute_trade(
                            signal, current_data, current_date, config, current_balance
                        )
                        if trade:
                            active_trades[trade.entry_time] = trade
                            current_balance += trade.net_profit
                
                # Update equity curve
                self.equity_curve.append({
                    'date': current_date,
                    'equity': current_balance,
                    'balance': current_balance
                })
                
                # Update peak and drawdown
                if current_balance > peak_balance:
                    peak_balance = current_balance
                
                drawdown = (peak_balance - current_balance) / peak_balance
                self.drawdown_curve.append({
                    'date': current_date,
                    'drawdown': drawdown,
                    'peak_equity': peak_balance
                })
                
                # Move to next day
                current_date += timedelta(days=1)
            
            # Calculate final metrics
            self._calculate_performance_metrics(config.initial_balance)
            
            # Generate report
            report = self._generate_report(symbol, config)
            
            self.logger.info(f"Backtest completed for {symbol}. Net profit: ${self.metrics['net_profit']:.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return {}
    
    def _get_historical_data(self, symbol: str, timeframe: str, 
                       config: BacktestConfig) -> Optional[pd.DataFrame]:
     """Get historical data for backtesting - ULTRA ROBUST VERSION"""
     try:
        # Calculate required bars
        days_diff = (config.end_date - config.start_date).days
        if timeframe == 'M1':
            required_bars = days_diff * 24 * 60
        elif timeframe == 'M15':
            required_bars = days_diff * 24 * 4
        elif timeframe == 'H1':
            required_bars = days_diff * 24
        elif timeframe == 'H4':
            required_bars = days_diff * 6
        elif timeframe == 'D1':
            required_bars = days_diff
        else:
            required_bars = days_diff * 24  # Default to H1
        
        # Add buffer
        required_bars = int(required_bars * 1.5)
        
        self.logger.info(f"Fetching {required_bars} bars for {symbol} {timeframe}")
        
        # Use robust data fetcher
        
        fetcher = RobustDataFetcher()
        
        data = fetcher.get_robust_data(symbol, timeframe, required_bars, config.end_date)
        
        if data is None:
            self.logger.error(f"All data fetch methods failed for {symbol} {timeframe}")
            return None
        
        # Filter to date range
        data = data[(data.index >= config.start_date) & (data.index <= config.end_date)]
        
        if len(data) == 0:
            self.logger.error(f"No data in date range for {symbol} {timeframe}")
            return None
        
        self.logger.info(f"Successfully retrieved {len(data)} bars for {symbol} {timeframe}")
        return data
        
     except Exception as e:
        self.logger.error(f"Error getting historical data: {str(e)}")
        return None
    
    def _generate_signals(self, symbol: str, data: pd.DataFrame, 
                     current_date: datetime) -> List:
     """Generate trading signals with proper data passing"""
     try:
        if hasattr(self, 'trading_bot') and self.trading_bot:
            
            # Get REAL multi-timeframe data
            multi_timeframe_data = self._get_real_multi_timeframe_data(symbol, current_date)
            
            if not multi_timeframe_data:
                self.logger.warning(f"No multi-timeframe data for {symbol} on {current_date}")
                return []
            
            # PASS THE DATA DIRECTLY to trading bot
            signals = self.trading_bot.generate_signals_multi(symbol, multi_timeframe_data)
            
            valid_signals = []
            for signal in signals:
                if signal and signal.action in ['BUY', 'SELL']:
                    valid_signals.append(signal)
            
            if valid_signals:
                self.logger.info(f"Generated {len(valid_signals)} signals for {symbol} on {current_date.date()}")
                for signal in valid_signals:
                    self.logger.info(f"  - {signal.action} | {signal.strategy} | Conf: {signal.confidence:.3f}")
            else:
                self.logger.info(f"No valid signals for {symbol} on {current_date.date()}")
                
            return valid_signals
            
        return []
            
     except Exception as e:
        self.logger.error(f"Error generating signals: {str(e)}")
        return []
        
    def _get_real_multi_timeframe_data(self, symbol: str, current_date: datetime) -> Dict:
     """Get REAL multi-timeframe data for backtesting"""
     try:
        mt5_data = MT5DataManager()
        if not mt5_data.connect():
            self.logger.error("Failed to connect to MT5 for multi-timeframe data")
            return {}
        
        # Define required timeframes based on trading mode
        timeframes = ['M15', 'H1', 'H4']  # Standard for day trading
        
        multi_tf_data = {}
        
        for tf in timeframes:
            # Calculate how many bars we need for this timeframe
            if tf == 'M15':
                bars_needed = 400  # ~4 days of M15 data
            elif tf == 'H1':
                bars_needed = 200  # ~8 days of H1 data  
            elif tf == 'H4':
                bars_needed = 100  # ~16 days of H4 data
            else:
                bars_needed = 100
            
            # Get historical data up to current backtest date
            tf_data = mt5_data.get_symbol_data(symbol, tf, bars=bars_needed)
            
            if tf_data is not None and len(tf_data) > 50:
                # Filter data to only include bars up to current backtest date
                tf_data = tf_data[tf_data.index <= current_date]
                if len(tf_data) > 0:
                    multi_tf_data[tf] = tf_data
                    self.logger.debug(f"Got {len(tf_data)} {tf} bars for {symbol}")
                else:
                    self.logger.warning(f"No {tf} data before {current_date} for {symbol}")
            else:
                self.logger.warning(f"Insufficient {tf} data for {symbol}")
        
        mt5_data.disconnect()
        
        if not multi_tf_data:
            self.logger.error(f"No multi-timeframe data obtained for {symbol}")
            return {}
            
        self.logger.info(f"Obtained {len(multi_tf_data)} timeframes for {symbol}")
        return multi_tf_data
        
     except Exception as e:
        self.logger.error(f"Error getting multi-timeframe data: {str(e)}")
        return {}
    
    def _execute_trade(self, signal, data: pd.DataFrame, current_date: datetime,
                      config: BacktestConfig, current_balance: float) -> Optional[BacktestTrade]:
        """Execute a trade in backtest"""
        try:
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Calculate position size based on risk management
            risk_per_trade = current_balance * 0.01  # 1% risk per trade
            stop_distance = abs(current_price - signal.stop_loss)
            
            if stop_distance == 0:
                return None
            
            # Calculate quantity (simplified)
            quantity = risk_per_trade / stop_distance
            
            # Apply slippage and commission
            entry_price = current_price
            if signal.direction == 'BUY':
                entry_price += config.slippage
            else:
                entry_price -= config.slippage
            
            # For backtesting, we'll use a simple exit on next day
            exit_date = current_date + timedelta(days=1)
            exit_price = self._get_exit_price(signal.symbol, exit_date)
            
            if exit_price is None:
                return None
            
            # Calculate profit
            if signal.direction == 'BUY':
                profit = (exit_price - entry_price) * quantity
            else:
                profit = (entry_price - exit_price) * quantity
            
            # Apply commission
            commission = config.commission
            net_profit = profit - commission
            
            trade = BacktestTrade(
                entry_time=current_date,
                exit_time=exit_date,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                profit=profit,
                commission=commission,
                slippage=config.slippage,
                net_profit=net_profit,
                strategy=signal.strategy,
                confidence=signal.confidence
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return None
    
    def _get_exit_price(self, symbol: str, exit_date: datetime) -> Optional[float]:
        """Get exit price for a given date"""
        # This would query historical data for the exit price
        # Simplified implementation
        return None
    
    def _check_trade_exits(self, active_trades: Dict, data: pd.DataFrame,
                          current_date: datetime, config: BacktestConfig):
        """Check for trade exits based on stop loss/take profit"""
        # Implementation for checking if active trades should be exited
        pass
    
    def _calculate_performance_metrics(self, initial_balance: float):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return
        
        profits = [trade.net_profit for trade in self.trades]
        
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([p for p in profits if p > 0])
        self.metrics['losing_trades'] = len([p for p in profits if p < 0])
        self.metrics['net_profit'] = sum(profits)
        self.metrics['total_profit'] = sum([p for p in profits if p > 0])
        self.metrics['total_loss'] = abs(sum([p for p in profits if p < 0]))
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        if self.metrics['total_loss'] > 0:
            self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss']
        
        # Calculate max drawdown
        if self.drawdown_curve:
            self.metrics['max_drawdown'] = max([d['drawdown'] for d in self.drawdown_curve])
        
        # Calculate Sharpe ratio (simplified)
        if len(profits) > 1:
            returns = np.array(profits) / initial_balance
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            self.metrics['sharpe_ratio'] = sharpe
        
        # Calculate expectancy
        if self.metrics['winning_trades'] > 0 and self.metrics['losing_trades'] > 0:
            avg_win = self.metrics['total_profit'] / self.metrics['winning_trades']
            avg_loss = self.metrics['total_loss'] / self.metrics['losing_trades']
            self.metrics['expectancy'] = (self.metrics['win_rate'] * avg_win - 
                                        (1 - self.metrics['win_rate']) * avg_loss)
    
    def _generate_report(self, symbol: str, config: BacktestConfig) -> Dict:
        """Generate comprehensive backtest report"""
        report = {
            'symbol': symbol,
            'period': f"{config.start_date.date()} to {config.end_date.date()}",
            'initial_balance': config.initial_balance,
            'final_balance': config.initial_balance + self.metrics['net_profit'],
            'metrics': self.metrics.copy(),
            'trades': len(self.trades),
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve
        }
        
        return report

    def run_comparative_backtest(self, symbols: List[str], config: BacktestConfig) -> Dict:
        """Run backtest across multiple symbols for comparison"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Running comparative backtest for {symbol}")
            result = self.run_backtest(symbol, 'H1', config)
            results[symbol] = result
        
        return results