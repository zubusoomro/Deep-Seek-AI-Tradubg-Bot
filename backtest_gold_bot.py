# backtest_gold_bot.py
#!/usr/bin/env python3
"""
Backtesting Script for Gold Scalping Bot
"""

import sys
import os
import logging
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backtesting.backtest_data_manager import BacktestDataManager
from src.backtesting.backtest_engine import BacktestEngine
from src.core.config_manager import ConfigManager

def setup_backtest_logging():
    """Setup logging for backtesting"""
    log_dir = "logs/backtest"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    print("GOLD SCALPING BOT - BACKTESTING")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Backtest Gold Scalping Bot')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-06-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to backtest')
    parser.add_argument('--timeframe', type=str, default='M15', help='Timeframe for backtest')
    parser.add_argument('--initial-equity', type=float, default=10000, help='Initial equity')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward testing')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_backtest_logging()
    logger = logging.getLogger('BacktestMain')
    
    # Initialize components
    config = ConfigManager()
    data_manager = BacktestDataManager()
    backtest_engine = BacktestEngine(config)
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    logger.info(f"Backtesting period: {start_date} to {end_date}")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    logger.info(f"Initial Equity: ${args.initial_equity:,.2f}")
    
    if args.walk_forward:
        # Walk-forward testing
        logger.info("Running walk-forward backtest...")
        periods = data_manager.generate_walk_forward_data(args.symbol, start_date, end_date, 30)
        
        all_results = []
        for i, (period_start, period_end) in enumerate(periods):
            logger.info(f"Walk-forward period {i+1}: {period_start} to {period_end}")
            
            # Load data for period
            historical_data = data_manager.load_historical_data(
                args.symbol, period_start, period_end, args.timeframe
            )
            
            if historical_data is None or len(historical_data) < 100:
                logger.warning(f"Insufficient data for period {i+1}, skipping")
                continue
            
            # Run backtest for period
            results = backtest_engine.run_backtest(historical_data, args.initial_equity)
            all_results.append(results)
            
            # Print period results
            print(f"\n--- Period {i+1} Results ---")
            print(backtest_engine.generate_report())
        
        # Aggregate walk-forward results
        if all_results:
            logger.info("Walk-forward testing completed")
            # Here you could aggregate results across all periods
        else:
            logger.error("No successful walk-forward periods")
            
    else:
        # Single backtest
        logger.info("Running single backtest...")
        
        # Load historical data
        historical_data = data_manager.load_historical_data(
            args.symbol, start_date, end_date, args.timeframe
        )
        
        if historical_data is None:
            logger.error("Failed to load historical data")
            return
        
        if len(historical_data) < 100:
            logger.error("Insufficient historical data for backtest")
            return
        
        logger.info(f"Loaded {len(historical_data)} historical bars")
        
        # Run backtest
        results = backtest_engine.run_backtest(historical_data, args.initial_equity)
        
        # Generate report
        report = backtest_engine.generate_report()
        print(report)
        
        # Save detailed results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{results_dir}/backtest_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Detailed report saved to {report_file}")
        
        # Generate plots if requested
        if args.plot:
            plot_file = f"{results_dir}/backtest_plot_{timestamp}.png"
            backtest_engine.plot_results(plot_file)
    
    logger.info("Backtesting completed!")

if __name__ == "__main__":
    main()