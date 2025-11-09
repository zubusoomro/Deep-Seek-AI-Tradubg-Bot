#!/usr/bin/env python3
"""
Simple Test - Check if MT5 connection and data work
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
from datetime import datetime, timedelta
from src.core.config_manager import ConfigManager
from src.data.mt5_data import MT5DataManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def test_mt5_connection():
    print("Testing MT5 Connection and Data")
    print("=" * 40)
    
    setup_logging()
    logger = logging.getLogger('MT5Test')
    
    # Initialize MT5
    mt5_data = MT5DataManager()
    
    if not mt5_data.connect():
        print("FAILED: Could not connect to MT5")
        return False
    
    print("SUCCESS: Connected to MT5")
    
    # Test symbols
    config = ConfigManager()
    symbols = config.get_active_symbols()
    
    print(f"Testing symbols: {symbols}")
    
    for symbol in symbols:
        print(f"\nTesting {symbol}:")
        
        # Try different symbol variations
        symbol_variations = [
            symbol,  # Original (XAUUSD.s)
            symbol.replace('.s', ''),  # Without .s (XAUUSD)
            symbol.split('.')[0],  # Just the base (XAUUSD)
        ]
        
        for sym in symbol_variations:
            print(f"  Trying '{sym}': ", end='')
            try:
                data = mt5_data.get_symbol_data(sym, 'H1', bars=100)
                if data is not None and len(data) > 0:
                    print(f"SUCCESS - Got {len(data)} bars")
                    print(f"    Latest price: {data['close'].iloc[-1]:.5f}")
                    break
                else:
                    print("NO DATA")
            except Exception as e:
                print(f"ERROR: {str(e)}")
        else:
            print(f"  All variations failed for {symbol}")
    
    mt5_data.disconnect()
    print("\nTest completed!")
    return True

if __name__ == "__main__":
    test_mt5_connection()