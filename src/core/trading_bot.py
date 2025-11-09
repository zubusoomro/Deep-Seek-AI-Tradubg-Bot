import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import time
import schedule
from threading import Thread

@dataclass
class TradeSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    lot_size: float
    stop_loss: float
    take_profit: float
    strategy: str
    timeframe: str
    rationale: str
    urgency: str = "normal"  # normal, high, immediate

class AdvancedTradingBot:
    def __init__(self, config_manager, ml_engine, risk_manager, position_manager):
        self.config = config_manager
        self.ml_engine = ml_engine
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.logger = self._setup_logging()
        
        self.active_symbols = []
        self.trading_mode = "day_trading"
        self.market_regime = "normal"
        self.last_signal_check = {}
        self.is_running = False
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        import os
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # File handler for detailed logging
        fh = logging.FileHandler('logs/trading_bot.log')
        fh.setLevel(logging.INFO)
        
        # Console handler for real-time monitoring
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection and configure symbols"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
            
            # Login to MT5 (you'll need to add your credentials)
            account = 547688
            password = "X2662!1m"
            server = "DooTechnology- Demo"
            
            if not mt5.login(account, password=password, server=server):
                self.logger.error("MT5 login failed")
                return False
            
            # Configure symbols
            for symbol in self.config.get_active_symbols():
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    self.logger.warning(f"Symbol {symbol} not found")
                    continue
                
                if not symbol_info.visible:
                    if not mt5.symbol_select(symbol, True):
                        self.logger.warning(f"Failed to select symbol {symbol}")
                        continue
                
                self.active_symbols.append(symbol)
                self.last_signal_check[symbol] = datetime.now() - timedelta(hours=1)
            
            self.logger.info(f"MT5 initialized with {len(self.active_symbols)} symbols: {self.active_symbols}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False
    
    def run(self):
        """Main trading bot execution loop"""
        self.logger.info("Starting Advanced Trading Bot...")
        
        if not self.initialize_mt5():
            self.logger.error("Failed to initialize MT5. Exiting...")
            return
        
        self.is_running = True
        
        # Schedule tasks
        schedule.every(1).minutes.do(self.run_trading_cycle)
        schedule.every(5).minutes.do(self.update_market_regime)
        schedule.every(10).minutes.do(self.update_trailing_stops)
        schedule.every(1).hours.do(self.performance_report)
        schedule.every().day.at("00:00").do(self.reset_daily_stats)
        
        # Start background scheduler
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        self.logger.info("Trading Bot started successfully!")
        
        # Keep main thread alive
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping Trading Bot...")
        self.is_running = False
        mt5.shutdown()
    
    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            self.logger.info("Starting trading cycle...")
            
            # Update market regime
            self.update_market_regime()
            
            # Check for high-impact news
            if self.is_high_impact_news():
                self.logger.info("High impact news detected - reducing exposure")
                self.close_all_positions()
                return
            
            # Check risk limits
            if not self.risk_manager.can_open_new_position("any"):
                self.logger.warning("Risk limits reached - skipping trading cycle")
                return
            
            # Generate and execute signals for all symbols
            for symbol in self.active_symbols:
                signals = self.generate_signals(symbol)
                self.execute_signals(signals)
            
            # Update ML models with latest data
            self.ml_engine.update_models()
            
            self.logger.info("Trading cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")


    def generate_signals_multi(self, symbol: str, multi_timeframe_data: Dict = None) -> List[TradeSignal]:
     """Generate trading signals for a symbol - FIXED DATA PASSING VERSION"""
     signals = []
    
     try:
        # Use provided multi-timeframe data if available (from backtester)
        if multi_timeframe_data is not None:
            # Backtester is providing the data directly
            available_timeframes = list(multi_timeframe_data.keys())
            self.logger.info(f"Using provided multi-timeframe data: {available_timeframes}")
        else:
            # Live trading - fetch data normally
            timeframes = self.config.get_trading_mode_config(self.trading_mode)['timeframes']
            multi_timeframe_data = {}
            
            for tf in timeframes:
                data = self.get_market_data(symbol, tf, bars=100)
                if data is not None and len(data) > 50:
                    multi_timeframe_data[tf] = data
                else:
                    self.logger.warning(f"No data returned for {symbol} {tf}")
        
        # If we don't have enough timeframes, return empty
        if not multi_timeframe_data:
            self.logger.warning(f"Insufficient data for {symbol}")
            return signals
        
        # Get ML confidence score - use any available timeframe
        primary_tf = list(multi_timeframe_data.keys())[0]
        ml_confidence = self.ml_engine.predict_confidence(symbol, multi_timeframe_data[primary_tf])
        
        self.logger.info(f"ML Confidence for {symbol}: {ml_confidence:.3f}")
        
        # Generate signals from different strategies
        active_strategies = self.config.get_active_strategies()
        
        for strategy_name in active_strategies:
            # Check if strategy is allowed
            symbol_config = self.config.get_symbol_config(symbol)
            mode_config = self.config.get_trading_mode_config(self.trading_mode)
            
            if (strategy_name not in symbol_config.get('allowed_strategies', []) or
                strategy_name not in mode_config.get('allowed_strategies', [])):
                continue
            
            strategy_signal = self._generate_strategy_signal(
                strategy_name, symbol, multi_timeframe_data, ml_confidence
            )
            
            if strategy_signal and strategy_signal.confidence > 0.4:
                signals.append(strategy_signal)
                self.logger.info(f"Generated {strategy_signal.action} signal from {strategy_name} with confidence {strategy_signal.confidence:.3f}")
        
        # Sort signals by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.info(f"Total signals generated for {symbol}: {len(signals)}")
        return signals
        
     except Exception as e:
        self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
        return signals
    
    def generate_signals(self, symbol: str) -> List[TradeSignal]:
        """Generate trading signals for a symbol"""
        signals = []
        
        try:
            # Get market data for multiple timeframes
            timeframes = self.config.get_trading_mode_config(self.trading_mode)['timeframes']
            multi_timeframe_data = {}
            
            for tf in timeframes:
                data = self.get_market_data(symbol, tf, bars=100)
                if data is not None and len(data) > 50:
                    multi_timeframe_data[tf] = data
            
            if not multi_timeframe_data:
                return signals
            
            # Get ML confidence score
            primary_tf = timeframes[0]
            ml_confidence = self.ml_engine.predict_confidence(symbol, multi_timeframe_data[primary_tf])
            
            # Generate signals from different strategies
            active_strategies = self.config.get_active_strategies()
            
            for strategy_name in active_strategies:
                # Check if strategy is allowed for this symbol and trading mode
                symbol_config = self.config.get_symbol_config(symbol)
                mode_config = self.config.get_trading_mode_config(self.trading_mode)
                
                if (strategy_name not in symbol_config.get('allowed_strategies', []) or
                    strategy_name not in mode_config.get('allowed_strategies', [])):
                    continue
                
                strategy_signal = self._generate_strategy_signal(
                    strategy_name, symbol, multi_timeframe_data, ml_confidence
                )
                
                if strategy_signal and strategy_signal.confidence > 0.4:  # Minimum threshold
                    signals.append(strategy_signal)
            
            # Sort signals by confidence (highest first)
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return signals
    
    def _generate_strategy_signal(self, strategy_name: str, symbol: str, 
                                multi_timeframe_data: Dict, ml_confidence: float) -> Optional[TradeSignal]:
        """Generate signal for specific strategy"""
        from src.strategies.ict_strategy import ICTStrategy
        from src.strategies.smc_strategy import SMCStrategy
        from src.strategies.multi_timeframe_strategy import MultiTimeframeStrategy
        
        try:
            if strategy_name == "ict_smart_money":
                strategy = ICTStrategy(self.config)
                return strategy.generate_signal(symbol, multi_timeframe_data, ml_confidence)
            
            elif strategy_name == "smc_advanced":
                strategy = SMCStrategy(self.config)
                return strategy.generate_signal(symbol, multi_timeframe_data, ml_confidence)
            
            elif strategy_name == "multi_timeframe_momentum":
                strategy = MultiTimeframeStrategy(self.config)
                return strategy.generate_signal(symbol, multi_timeframe_data, ml_confidence)
            
            # Add more strategies here...
            
        except Exception as e:
            self.logger.error(f"Error in {strategy_name} strategy: {str(e)}")
        
        return None
    
    def execute_signals(self, signals: List[TradeSignal]):
        """Execute trading signals"""
        for signal in signals:
            try:
                # Check if we already have a position for this symbol
                active_positions = self.position_manager.get_active_positions()
                has_active_position = any(
                    pos['symbol'] == signal.symbol for pos in active_positions.values()
                )
                
                if has_active_position and signal.action != "HOLD":
                    self.logger.info(f"Skipping {signal.symbol} - active position exists")
                    continue
                
                # Check minimum confidence threshold
                strategy_config = self.config.get_strategy_config(signal.strategy)
                min_confidence = strategy_config.get('required_confidence', 0.6)
                
                if signal.confidence < min_confidence:
                    continue
                
                # Check if enough time has passed since last signal
                last_check = self.last_signal_check.get(signal.symbol, datetime.now() - timedelta(hours=1))
                time_since_last = datetime.now() - last_check
                min_interval = timedelta(minutes=5)  # Minimum 5 minutes between trades
                
                if time_since_last < min_interval and signal.urgency != "immediate":
                    continue
                
                # Execute the trade
                if signal.action in ["BUY", "SELL"]:
                    success = self.position_manager.open_position(
                        symbol=signal.symbol,
                        action=signal.action,
                        lot_size=signal.lot_size,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        comment=f"{signal.strategy}-{signal.timeframe}-conf:{signal.confidence:.2f}"
                    )
                    
                    if success:
                        self.last_signal_check[signal.symbol] = datetime.now()
                        self.performance_stats['total_trades'] += 1
                        self.logger.info(f"Executed {signal.action} on {signal.symbol} "
                                       f"with confidence {signal.confidence:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error executing signal for {signal.symbol}: {str(e)}")
    
    def get_market_data(self, symbol: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get market data from MT5"""
        try:
            # Map timeframe string to MT5 constant
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            
            if rates is None:
                self.logger.warning(f"No data returned for {symbol} {timeframe}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def update_market_regime(self):
        """Detect and update current market regime"""
        try:
            # Use multiple symbols to determine overall market regime
            regime_scores = []
            
            for symbol in self.active_symbols[:2]:  # Use first 2 symbols for efficiency
                data = self.get_market_data(symbol, "H1", bars=200)
                if data is not None:
                    regime = self._analyze_market_regime(data)
                    regime_scores.append(regime)
            
            if not regime_scores:
                return
            
            # Determine overall regime (simplified logic)
            avg_volatility = np.mean([r['volatility'] for r in regime_scores])
            avg_trend_strength = np.mean([r['trend_strength'] for r in regime_scores])
            
            if avg_volatility > 0.02 and avg_trend_strength > 0.7:
                self.market_regime = "trending_high_vol"
            elif avg_volatility > 0.02 and avg_trend_strength < 0.3:
                self.market_regime = "ranging_high_vol"
            elif avg_volatility < 0.005 and avg_trend_strength > 0.7:
                self.market_regime = "trending_low_vol"
            elif avg_volatility < 0.005 and avg_trend_strength < 0.3:
                self.market_regime = "ranging_low_vol"
            else:
                self.market_regime = "normal"
            
            # Adjust trading mode based on regime
            self._adjust_trading_for_regime()
            
        except Exception as e:
            self.logger.error(f"Error updating market regime: {str(e)}")
    
    def _analyze_market_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze market regime from price data"""
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Trend strength (ADX-like calculation)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Simplified trend strength
        trend_strength = np.abs(data['close'].pct_change(20).iloc[-1])
        
        return {
            'volatility': volatility,
            'trend_strength': min(trend_strength, 1.0)
        }
    
    def _adjust_trading_for_regime(self):
        """Adjust trading parameters based on market regime"""
        regime_settings = self.config.strategies['market_regime_settings'].get(self.market_regime, {})
        
        if regime_settings:
            # Adjust risk multiplier
            risk_multiplier = regime_settings.get('risk_multiplier', 1.0)
            # This would be applied in the risk manager
            
            # Adjust preferred strategies
            preferred_strategies = regime_settings.get('preferred_strategies', [])
            # Could prioritize these strategies in signal generation
        
        self.logger.info(f"Market regime: {self.market_regime}")
    
    def is_high_impact_news(self) -> bool:
        """Check for high-impact news events"""
        # This would integrate with a news API
        # For now, return False
        return False
    
    def close_all_positions(self):
        """Close all active positions"""
        try:
            active_positions = self.position_manager.get_active_positions()
            for ticket in list(active_positions.keys()):
                self.position_manager.close_position(ticket)
            
            self.logger.info("All positions closed")
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
    
    def update_trailing_stops(self):
        """Update trailing stops for active positions"""
        self.position_manager.update_trailing_stops()
    
    def performance_report(self):
        """Generate performance report"""
        win_rate = (self.performance_stats['winning_trades'] / 
                   self.performance_stats['total_trades'] * 100) if self.performance_stats['total_trades'] > 0 else 0
        
        self.logger.info(
            f"Performance Report - "
            f"Total Trades: {self.performance_stats['total_trades']}, "
            f"Win Rate: {win_rate:.1f}%, "
            f"Total PnL: ${self.performance_stats['total_pnl']:.2f}, "
            f"Daily PnL: ${self.performance_stats['daily_pnl']:.2f}"
        )
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.performance_stats['daily_pnl'] = 0.0
        self.risk_manager.reset_daily_stats()
        self.logger.info("Daily statistics reset")