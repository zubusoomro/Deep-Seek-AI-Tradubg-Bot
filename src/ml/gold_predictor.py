# src/ml/gold_predictor.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from typing import Tuple, Optional

class GoldPredictor:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('GoldPredictor')
        
        self.lgb_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        # LightGBM
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Simple Neural Network
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42
        )
    
    def prepare_training_data(self, features: pd.DataFrame, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with labels"""
        try:
            # Create labels: 1 for BUY, 0 for SELL, based on future price movement
            future_returns = data['close'].pct_change(5).shift(-5)  # 5 bars ahead
            
            # Create binary labels: 1 if price goes up, 0 if down
            labels = (future_returns > 0).astype(int)
            
            # Align features and labels
            aligned_data = pd.concat([features, labels], axis=1).dropna()
            X = aligned_data.iloc[:, :-1].values
            y = aligned_data.iloc[:, -1].values
            
            self.logger.info(f"Training data: {X.shape} features, {len(y)} samples")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def train(self, features: pd.DataFrame, data: pd.DataFrame) -> bool:
        """Train the models"""
        try:
            X, y = self.prepare_training_data(features, data)
            
            if len(X) < 100:
                self.logger.warning("Insufficient training data")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train LightGBM
            self.lgb_model.fit(X_train, y_train)
            lgb_pred = self.lgb_model.predict(X_test)
            lgb_accuracy = accuracy_score(y_test, lgb_pred)
            
            # Train Neural Network
            self.nn_model.fit(X_train_scaled, y_train)
            nn_pred = self.nn_model.predict(X_test_scaled)
            nn_accuracy = accuracy_score(y_test, nn_pred)
            
            self.is_trained = True
            
            self.logger.info(f"Training completed - LGB Accuracy: {lgb_accuracy:.4f}, NN Accuracy: {nn_accuracy:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """Predict trading signal with confidence"""
        try:
            if not self.is_trained or features.empty:
                return "HOLD", 0.0
            
            # Get latest features
            latest_features = features.iloc[[-1]].values
            
            # LightGBM prediction
            lgb_proba = self.lgb_model.predict_proba(latest_features)[0]
            lgb_confidence = max(lgb_proba)
            lgb_prediction = self.lgb_model.predict(latest_features)[0]
            
            # Neural Network prediction
            latest_scaled = self.scaler.transform(latest_features)
            nn_proba = self.nn_model.predict_proba(latest_scaled)[0]
            nn_confidence = max(nn_proba)
            nn_prediction = self.nn_model.predict(latest_scaled)[0]
            
            # Ensemble prediction (weighted average)
            if lgb_prediction == nn_prediction:
                final_prediction = "BUY" if lgb_prediction == 1 else "SELL"
                final_confidence = (lgb_confidence + nn_confidence) / 2
            else:
                # Use higher confidence model
                if lgb_confidence > nn_confidence:
                    final_prediction = "BUY" if lgb_prediction == 1 else "SELL"
                    final_confidence = lgb_confidence
                else:
                    final_prediction = "BUY" if nn_prediction == 1 else "SELL"
                    final_confidence = nn_confidence
            
            # Apply confidence threshold
            min_confidence = self.config.get_gold_config().get('ml_config', {}).get('min_confidence_threshold', 0.60)
            if final_confidence < min_confidence:
                return "HOLD", final_confidence
            
            return final_prediction, final_confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return "HOLD", 0.0
    
    def save_models(self, path: str = "models"):
        """Save trained models"""
        try:
            joblib.dump(self.lgb_model, f"{path}/gold_lgb_model.pkl")
            joblib.dump(self.nn_model, f"{path}/gold_nn_model.pkl")
            joblib.dump(self.scaler, f"{path}/gold_scaler.pkl")
            self.logger.info("Models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, path: str = "models"):
        """Load trained models"""
        try:
            self.lgb_model = joblib.load(f"{path}/gold_lgb_model.pkl")
            self.nn_model = joblib.load(f"{path}/gold_nn_model.pkl")
            self.scaler = joblib.load(f"{path}/gold_scaler.pkl")
            self.is_trained = True
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")