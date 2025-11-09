#!/usr/bin/env python3
"""
Model Training Script for Advanced Trading Bot
Trains ML models on historical data before live trading
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.core.config_manager import ConfigManager
from src.ml.ml_engine import EnhancedMLEngine
from src.ml.model_trainer import AdvancedModelTrainer
from src.data.mt5_data import MT5DataManager
import argparse

def setup_training_logging():
    """Setup logging for training process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/model_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_models_for_symbol(symbol, mt5_data, ml_engine, lookback_days=365):
    """Train ML models for a specific symbol"""
    logger = logging.getLogger('ModelTraining')
    
    try:
        logger.info(f"Starting model training for {symbol}...")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Get data for multiple timeframes
        data = mt5_data.get_symbol_data(symbol, 'H1', bars=5000)
        if data is None or len(data) < 1000:
            logger.warning(f"Insufficient data for {symbol}. Skipping...")
            return False
        
        # Generate features
        logger.info("Generating features...")
        feature_engineer = ml_engine.feature_engineer
        features = feature_engineer.generate_advanced_features(data)
        
        if features.empty:
            logger.warning(f"Could not generate features for {symbol}. Skipping...")
            return False
        
        # Prepare training data
        logger.info("Preparing training data...")
        X, y, feature_names = ml_engine.model_trainer.prepare_training_data(
            features, data, lookforward=5, return_threshold=0.002
        )
        
        if len(X) == 0:
            logger.warning(f"No training data generated for {symbol}. Skipping...")
            return False
        
        logger.info(f"Training data: {X.shape} features, {len(y)} samples")
        
        # Train neural network
        logger.info("Training neural network...")
        sequence_length = 30
        data_loaders = ml_engine.model_trainer.create_data_loaders(
            X, y, sequence_length=sequence_length, batch_size=32
        )
        
        if not data_loaders:
            logger.warning(f"Could not create data loaders for {symbol}. Skipping...")
            return False
        
        # Get the model for this symbol
        model = ml_engine.models[symbol]
        
        # Train the model
        training_history = ml_engine.model_trainer.train_neural_network(
            model, data_loaders, epochs=100, learning_rate=0.001
        )
        
        if training_history:
            final_metrics = training_history['final_metrics']
            accuracy = final_metrics.get('accuracy', 0)
            f1_score = final_metrics.get('f1_score', 0)
            
            logger.info(f"Successfully trained {symbol} model")
            logger.info(f"Model Performance - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
            
            # Save the trained model
            ml_engine.model_trainer.save_model(
                model, 
                f"{symbol}_model",
                metadata={
                    'symbol': symbol,
                    'training_date': datetime.now().isoformat(),
                    'feature_names': feature_names,
                    'training_samples': len(X),
                    'final_accuracy': accuracy,
                    'final_f1_score': f1_score
                }
            )
            
            # Plot training history
            ml_engine.model_trainer.plot_training_history(
                training_history, 
                save_path=f"models/{symbol}_training_history.png"
            )
            
            return True
        else:
            logger.error(f"Failed to train model for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        return False

def main():
    print("Advanced Trading Bot - Model Training")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ML models for trading bot')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--lookback', type=int, default=365, help='Lookback days for training data')
    parser.add_argument('--retrain', action='store_true', help='Force retrain existing models')
    args = parser.parse_args()
    
    # Setup logging
    setup_training_logging()
    logger = logging.getLogger('ModelTraining')
    
    # Initialize components
    config_manager = ConfigManager()
    mt5_data = MT5DataManager()
    ml_engine = EnhancedMLEngine(config_manager, use_gpu=True)
    
    # Connect to MT5
    logger.info("Connecting to MT5...")
    if not mt5_data.connect():
        logger.error("Failed to connect to MT5. Exiting...")
        return
    
    # Determine which symbols to train
    if args.symbols:
        symbols_to_train = args.symbols
    else:
        symbols_to_train = config_manager.get_active_symbols()
    
    logger.info(f"Training models for symbols: {symbols_to_train}")
    
    # Train models for each symbol
    results = {}
    for symbol in symbols_to_train:
        success = train_models_for_symbol(symbol, mt5_data, ml_engine, args.lookback)
        results[symbol] = success
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    
    successful = [s for s, success in results.items() if success]
    failed = [s for s, success in results.items() if not success]
    
    print(f"Successful: {len(successful)} symbols")
    for symbol in successful:
        print(f"   - {symbol}")
    
    if failed:
        print(f"Failed: {len(failed)} symbols")
        for symbol in failed:
            print(f"   - {symbol}")
    
    # Cleanup
    mt5_data.disconnect()
    logger.info("Model training completed!")

if __name__ == "__main__":
    main()