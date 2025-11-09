# main.py
#!/usr/bin/env python3
"""
GOLD SCALPING BOT - COMPLETE IMPLEMENTATION
XAUUSD M15 Scalping with Breakeven & Partial Profit Management
"""

import sys
import os
import time
import logging
import signal
import threading
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config_manager import ConfigManager
from src.data.gold_data_fetcher import GoldDataFetcher
from src.features.gold_feature_engineer import GoldFeatureEngineer
from src.ml.gold_predictor import GoldPredictor
from src.strategies.gold_momentum import GoldMomentumStrategy, TradeSignal
from src.risk.risk_manager import RiskManager
from src.trading.position_manager import PositionManager

# Load environment variables
load_dotenv()

class GoldScalpingBot:
    def __init__(self, live_trading: bool = False):
        self.live_trading = live_trading
        self.is_running = False
        self.last_trade_time = None
        self.setup_logging()
        
        # Initialize components
        self.logger.info("Initializing Gold Scalping Bot...")
        self.config = ConfigManager()
        self.data_fetcher = GoldDataFetcher()
        self.feature_engineer = GoldFeatureEngineer()
        self.predictor = GoldPredictor(self.config)
        self.strategy = GoldMomentumStrategy(self.config)
        self.risk_manager = RiskManager(self.config)
        self.position_manager = PositionManager(self.config, self.risk_manager)
        
        # Load ML models
        self.load_ml_models()
        
        self.logger.info("Gold Scalping Bot initialized successfully!")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/gold_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('GoldScalpingBot')
    
    def load_ml_models(self):
        """Load trained ML models"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                self.logger.info("Created models directory")
            
            self.predictor.load_models(models_dir)
            self.logger.info("ML models loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {e}. Training new models...")
            self.train_models()
    
    def train_models(self):
        """Train ML models with historical data"""
        self.logger.info("Training ML models with historical data...")
        
        try:
            # Connect to MT5 first
            if not self.data_fetcher.connect_mt5():
                self.logger.error("Failed to connect to MT5 for training")
                return False
            
            # Get historical data for training
            historical_data = self.data_fetcher.get_historical_data(days=60)
            self.data_fetcher.disconnect_mt5()
            
            if historical_data is None or len(historical_data) < 1000:
                self.logger.error("Insufficient historical data for training")
                return False
            
            # Generate features
            features = self.feature_engineer.generate_features(historical_data)
            
            if features.empty or len(features) < 500:
                self.logger.error("Insufficient features for training")
                return False
            
            # Train models
            success = self.predictor.train(features, historical_data)
            
            if success:
                self.predictor.save_models("models")
                self.logger.info("Models trained and saved successfully")
                return True
            else:
                self.logger.error("Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def connect_to_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        self.logger.info("Connecting to MT5...")
        
        if not self.data_fetcher.connect_mt5():
            self.logger.error("Failed to connect to MT5")
            return False
        
        # Verify symbol exists
        symbol = self.config.get_gold_config().get('symbol', 'XAUUSD')
        symbol_info = self.data_fetcher.get_symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Symbol {symbol} not found in MT5")
            return False
        
        self.logger.info(f"Connected to MT5 - Trading: {symbol}")
        return True
    
    def get_trading_decision(self) -> dict:
        """Get trading decision based on current market conditions"""
        try:
            # Get current market data
            current_data = self.data_fetcher.get_current_data(bars=100)
            if current_data is None or current_data.empty:
                return {"action": "HOLD", "reason": "No market data"}
            
            # Generate features
            features = self.feature_engineer.generate_features(current_data)
            if features.empty:
                return {"action": "HOLD", "reason": "No features generated"}
            
            # Get ML prediction
            ml_prediction, ml_confidence = self.predictor.predict(features)
            
            # Generate trading signal
            signal = self.strategy.generate_signal(current_data, ml_prediction, ml_confidence)
            
            return {
                "action": signal["signal"].value,
                "confidence": signal["confidence"],
                "position_size": signal.get("position_size", 0),
                "stop_loss": signal.get("stop_loss", 0),
                "take_profit": signal.get("take_profit", 0),
                "reason": signal.get("reason", ""),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {e}")
            return {"action": "HOLD", "reason": f"Error: {str(e)}"}
    
    def execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            # 1. Manage existing positions
            current_data = self.data_fetcher.get_current_data(bars=50)
            if current_data is not None:
                self.position_manager.manage_open_positions(current_data)
            
            # 2. Check if we should make new trading decision
            if not self._should_analyze():
                return
            
            # 3. Get trading decision
            decision = self.get_trading_decision()
            
            # 4. Log decision
            self.logger.info(f"Trading Decision: {decision['action']} (Confidence: {decision['confidence']:.3f}) - {decision['reason']}")
            
            # 5. Execute trade if not HOLD
            if decision['action'] != 'HOLD' and self.live_trading:
                self.execute_trade(decision)
            
            self.last_trade_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def _should_analyze(self) -> bool:
        """Check if we should analyze for new trades"""
        # Don't analyze too frequently
        if self.last_trade_time and (datetime.now() - self.last_trade_time).total_seconds() < 60:
            return False
        
        # Check trading sessions
        current_time = datetime.now().time()
        sessions = self.config.get_trading_sessions()
        
        london_start = self._parse_time(sessions.get('london_open', '08:00'))
        london_end = self._parse_time(sessions.get('london_close', '16:00'))
        ny_start = self._parse_time(sessions.get('ny_open', '13:00'))
        ny_end = self._parse_time(sessions.get('ny_close', '21:00'))
        
        # Check if current time is within trading sessions
        in_london = london_start <= current_time <= london_end
        in_ny = ny_start <= current_time <= ny_end
        
        return in_london or in_ny
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string to time object"""
        return datetime.strptime(time_str, '%H:%M').time()
    
    def execute_trade(self, decision: dict):
        """Execute a trade based on decision"""
        try:
            # Convert decision to signal format
            signal = {
                "signal": TradeSignal(decision['action']),
                "confidence": decision['confidence'],
                "position_size": decision['position_size'],
                "stop_loss": decision['stop_loss'],
                "take_profit": decision['take_profit'],
                "reason": decision['reason']
            }
            
            # Open position
            ticket = self.position_manager.open_position(signal)
            
            if ticket:
                self.logger.info(f"Trade executed successfully - Ticket: {ticket}")
            else:
                self.logger.warning("Trade execution failed")
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def run(self):
        """Main bot execution loop"""
        if not self.connect_to_mt5():
            self.logger.error("Cannot start bot without MT5 connection")
            return
        
        self.is_running = True
        self.logger.info("Gold Scalping Bot started!")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Main loop
        while self.is_running:
            try:
                self.execute_trading_cycle()
                
                # Print status every 10 minutes
                if datetime.now().minute % 10 == 0:
                    self.print_status()
                
                # Sleep between cycles (15 seconds for M15 scalping)
                time.sleep(15)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def print_status(self):
        """Print current bot status"""
        stats = self.risk_manager.get_daily_stats()
        
        status_msg = f"""
=== GOLD SCALPING BOT STATUS ===
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Live Trading: {self.live_trading}
Total Trades Today: {stats['total_trades']}
Open Positions: {stats['open_trades']}
Daily PnL: ${stats['daily_pnl']:.2f}
Win Rate: {stats['win_rate']:.1%}
Max Daily Loss Remaining: ${stats['max_daily_loss_remaining']:.2f}
================================
"""
        self.logger.info(status_msg)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating shutdown sequence...")
        self.is_running = False
        
        # Close all positions if in live trading
        if self.live_trading:
            self.logger.info("Closing all open positions...")
            self.position_manager.close_all_positions()
        
        # Disconnect from MT5
        self.data_fetcher.disconnect_mt5()
        
        self.logger.info("Gold Scalping Bot shutdown complete.")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gold Scalping Trading Bot')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: paper trading)')
    parser.add_argument('--train', action='store_true', help='Train models before starting')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = GoldScalpingBot(live_trading=args.live)
    
    # Train models if requested
    if args.train:
        bot.logger.info("Training models as requested...")
        if not bot.train_models():
            bot.logger.error("Model training failed. Exiting.")
            return
    
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.logger.info("Bot stopped by user")
    except Exception as e:
        bot.logger.error(f"Bot crashed: {e}")
    finally:
        bot.shutdown()

if __name__ == "__main__":
    main()