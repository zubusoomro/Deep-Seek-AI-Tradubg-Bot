# train_gold_model.py
#!/usr/bin/env python3
"""
Training Script for Gold Scalping Bot
"""

import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config_manager import ConfigManager
from src.data.gold_data_fetcher import GoldDataFetcher
from src.features.gold_feature_engineer import GoldFeatureEngineer
from src.ml.gold_predictor import GoldPredictor

def setup_training_logging():
    """Setup logging for training"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    print("GOLD SCALPING BOT - MODEL TRAINING")
    print("=" * 50)
    
    setup_training_logging()
    logger = logging.getLogger('ModelTraining')
    
    # Initialize components
    config = ConfigManager()
    data_fetcher = GoldDataFetcher()
    feature_engineer = GoldFeatureEngineer()
    predictor = GoldPredictor(config)
    
    # Connect to MT5
    logger.info("Connecting to MT5...")
    if not data_fetcher.connect_mt5():
        logger.error("Failed to connect to MT5")
        return
    
    try:
        # Get historical data
        logger.info("Fetching historical data...")
        historical_data = data_fetcher.get_historical_data(days=90)  # 90 days for training
        
        if historical_data is None or len(historical_data) < 1000:
            logger.error("Insufficient historical data")
            return
        
        logger.info(f"Retrieved {len(historical_data)} bars of historical data")
        
        # Generate features
        logger.info("Generating features...")
        features = feature_engineer.generate_features(historical_data)
        
        if features.empty or len(features) < 500:
            logger.error("Insufficient features generated")
            return
        
        logger.info(f"Generated {len(features)} feature samples with {len(features.columns)} features")
        
        # Train models
        logger.info("Starting model training...")
        success = predictor.train(features, historical_data)
        
        if success:
            # Save models
            os.makedirs("models", exist_ok=True)
            predictor.save_models("models")
            logger.info("Models trained and saved successfully!")
            
            # Print training summary
            print("\n" + "=" * 50)
            print("TRAINING SUMMARY")
            print("=" * 50)
            print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Symbol: XAUUSD")
            print(f"Timeframe: M15")
            print(f"Training Data: {len(historical_data)} bars")
            print(f"Feature Samples: {len(features)}")
            print(f"Features Used: {len(features.columns)}")
            print(f"Models Saved: LightGBM, Neural Network, Scaler")
            print("Training completed successfully!")
            
        else:
            logger.error("Model training failed")
            print("Training failed! Check logs for details.")
            
    except Exception as e:
        logger.error(f"Training process failed: {e}")
        print(f"Training failed with error: {e}")
    
    finally:
        data_fetcher.disconnect_mt5()
        logger.info("Training completed")

if __name__ == "__main__":
    main()