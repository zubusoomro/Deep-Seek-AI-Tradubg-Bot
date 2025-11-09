#!/usr/bin/env python3
"""
Backtesting Script for Advanced Trading Bot
Test strategies on historical data before live trading
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
from src.core.config_manager import ConfigManager
from src.core.trading_bot import AdvancedTradingBot
from src.backtesting.backtester import Backtester, BacktestConfig

def setup_backtest_logging():
    """Setup logging for backtesting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/backtesting.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    print("Advanced Trading Bot - Strategy Backtesting")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--timeframe', default='H1', help='Timeframe for backtesting')
    args = parser.parse_args()
    
    # Setup logging
    setup_backtest_logging()
    logger = logging.getLogger('Backtesting')
    
    # Initialize components
    config_manager = ConfigManager()
    
    # For backtesting, we need a trading bot instance but won't connect to MT5
    from src.ml.ml_engine import EnhancedMLEngine
    from src.core.risk_manager import RiskManager
    from src.core.position_manager import PositionManager
    
    ml_engine = EnhancedMLEngine(config_manager, use_gpu=False)
    risk_manager = RiskManager(config_manager)
    position_manager = PositionManager(config_manager)
    
    trading_bot = AdvancedTradingBot(config_manager, ml_engine, risk_manager, position_manager)
    
    # Create backtester
    backtester = Backtester(config_manager, trading_bot)
    
    # Determine symbols to backtest
    if args.symbols:
        symbols_to_test = args.symbols
    else:
        symbols_to_test = config_manager.get_active_symbols()
    
    # Setup backtest configuration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.initial_balance,
        spread=0.0002,
        commission=0.0,
        slippage=0.0001
    )
    
    logger.info(f"Backtesting {len(symbols_to_test)} symbols from {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial balance: ${args.initial_balance:,.2f}")
    
    # Run backtests
    results = {}
    for symbol in symbols_to_test:
        logger.info(f"Backtesting {symbol}...")
        
        result = backtester.run_backtest(symbol, args.timeframe, backtest_config)
        results[symbol] = result
        
        if result:
            metrics = result.get('metrics', {})
            net_profit = metrics.get('net_profit', 0)
            win_rate = metrics.get('win_rate', 0)
            total_trades = metrics.get('total_trades', 0)
            
            logger.info(f"  {symbol} Results: "
                       f"Trades: {total_trades}, "
                       f"Win Rate: {win_rate:.1%}, "
                       f"Net Profit: ${net_profit:.2f}")
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("BACKTESTING SUMMARY REPORT")
    print("=" * 60)
    
    total_net_profit = 0
    total_trades = 0
    winning_symbols = 0
    
    for symbol, result in results.items():
        if result:
            metrics = result.get('metrics', {})
            net_profit = metrics.get('net_profit', 0)
            trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0)
            
            total_net_profit += net_profit
            total_trades += trades
            
            status = "PROFITABLE" if net_profit > 0 else "LOSS"
            if net_profit > 0:
                winning_symbols += 1
            
            print(f"{symbol:<10} {status:<15} "
                  f"Trades: {trades:>3} | "
                  f"Win Rate: {win_rate:>6.1%} | "
                  f"Net: ${net_profit:>8.2f}")
    
    print("-" * 60)
    print(f"Overall Results:")
    print(f"  Total Net Profit: ${total_net_profit:,.2f}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Profitable Symbols: {winning_symbols}/{len(symbols_to_test)}")
    print(f"  Return on Capital: {(total_net_profit/args.initial_balance)*100:.2f}%")
    
    # Save detailed results
    import json
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Backtesting completed! Results saved to backtest_results.json")

if __name__ == "__main__":
    main()