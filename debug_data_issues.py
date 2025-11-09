#!/usr/bin/env python3
"""
Debug data fetching issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.robust_data_fetcher import RobustDataFetcher
from datetime import datetime, timedelta

def debug_data_fetching():
    print("🔧 Debugging Data Fetching Issues")
    print("=" * 40)
    
    fetcher = RobustDataFetcher()
    
    symbols = ['XAUUSD.s', 'USDJPY.s', 'EURUSD.s']
    timeframes = ['M15', 'H1', 'H4']
    end_date = datetime.now()
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}:")
        for tf in timeframes:
            print(f"  {tf}: ", end='')
            data = fetcher.get_robust_data(symbol, tf, 100, end_date)
            if data is not None and len(data) > 0:
                print(f"✅ SUCCESS - {len(data)} bars, from {data.index[0]} to {data.index[-1]}")
            else:
                print("❌ FAILED")
    
    fetcher.disconnect()

if __name__ == "__main__":
    debug_data_fetching()