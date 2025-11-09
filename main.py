#!/usr/bin/env python3
"""
Advanced AI Trading Bot - Complete Production System
Author: AI Trading Bot Developer
Version: 2.0.0
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config_manager import ConfigManager
from src.core.trading_bot import AdvancedTradingBot
from src.core.risk_manager import RiskManager
from src.core.position_manager import PositionManager
from src.ml.ml_engine import EnhancedMLEngine

def setup_environment():
    """Setup logging and environment"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    print("🚀 Advanced AI Trading Bot Starting...")
    print("=" * 50)

def main():
    """Main execution function"""
    try:
        setup_environment()
        
        # Initialize components
        print("📋 Initializing Configuration Manager...")
        config_manager = ConfigManager()
        
        print("🤖 Initializing ML Engine (GPU Accelerated)...")
        ml_engine = EnhancedMLEngine(config_manager, use_gpu=True)
        
        print("🛡️ Initializing Risk Manager...")
        risk_manager = RiskManager(config_manager)
        
        print("📊 Initializing Position Manager...")
        position_manager = PositionManager(config_manager)
        
        print("🎯 Creating Trading Bot Instance...")
        bot = AdvancedTradingBot(config_manager, ml_engine, risk_manager, position_manager)
        
        # Start the trading bot
        print("✅ All systems initialized successfully!")
        print("💰 Starting live trading...")
        print("=" * 50)
        
        bot.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Received interrupt signal...")
        print("🔴 Shutting down trading bot gracefully...")
    except Exception as e:
        print(f"\n\n❌ Critical error: {str(e)}")
        logging.error(f"Critical error in main: {str(e)}", exc_info=True)
    finally:
        print("✅ Trading bot shutdown complete.")
        print("📊 Check logs/trading_bot.log for detailed operations.")

if __name__ == "__main__":
    main()