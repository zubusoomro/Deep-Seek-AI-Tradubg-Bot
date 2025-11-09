import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Optional
import os
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ADD THIS IMPORT
from .model_trainer import AdvancedModelTrainer
from .feature_engineer import FeatureEngineer

class AdvancedTradingModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4, dropout: float = 0.3):
        super(AdvancedTradingModel, self).__init__()
        
        # Bidirectional LSTM with attention
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # BUY, SELL, HOLD
            nn.Softmax(dim=1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Positive output for volatility
        )
    
    def forward(self, x):
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection + layer norm
        residual = self.layer_norm1(attn_out + lstm_out)
        
        # Feedforward
        ff_out = self.feed_forward(residual[:, -1, :])  # Use last timestep
        ff_out = self.layer_norm2(ff_out)
        
        # Multiple outputs
        action_probs = self.action_head(ff_out)
        confidence = self.confidence_head(ff_out)
        volatility = self.volatility_head(ff_out)
        
        return action_probs, confidence, volatility, attn_weights

class EnhancedMLEngine:
    def __init__(self, config_manager, use_gpu: bool = True):
        self.config = config_manager
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.logger = logging.getLogger('MLEngine')
        
        self.models: Dict[str, AdvancedTradingModel] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.ensemble_models: Dict[str, Dict] = {}
        self.feature_engineer = FeatureEngineer()

        self.model_trainer = AdvancedModelTrainer(config_manager)

        
        self._initialize_models()
        self.logger.info(f"ML Engine initialized with device: {self.device}")
        self.logger.info(f"Model trainer initialized: {type(self.model_trainer).__name__}")
        
    
    def _initialize_models(self):
        """Initialize or load pre-trained models for all symbols"""
        symbols = self.config.get_active_symbols()
        
        for symbol in symbols:
            model_path = f"models/{symbol}_advanced_model.pth"
            ensemble_path = f"models/{symbol}_ensemble.pkl"
            scaler_path = f"models/{symbol}_scaler.pkl"
            
            # Neural Network Model
            if os.path.exists(model_path):
                self.models[symbol] = self._load_pytorch_model(model_path)
            else:
                input_size = 796  # Number of features from feature engineering
                self.models[symbol] = AdvancedTradingModel(
                    input_size=input_size,
                    hidden_size=256,
                    num_layers=4,
                    dropout=0.3
                ).to(self.device)
            
            # Ensemble Models
            if os.path.exists(ensemble_path):
                self.ensemble_models[symbol] = joblib.load(ensemble_path)
            else:
                self.ensemble_models[symbol] = {
                    'xgb': XGBClassifier(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'lgb': LGBMClassifier(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'catboost': CatBoostClassifier(
                        iterations=200,
                        depth=8,
                        learning_rate=0.1,
                        random_seed=42,
                        verbose=False
                    ),
                    'meta': RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                }
            
            # Scaler
            if os.path.exists(scaler_path):
                self.scalers[symbol] = joblib.load(scaler_path)
            else:
                self.scalers[symbol] = StandardScaler()
    
    def predict_confidence(self, symbol: str, data: pd.DataFrame) -> float:
     """Predict trading confidence using ensemble of advanced models - FIXED FEATURE NAMES"""
     try:
        if len(data) < 100:  # Need sufficient data
            return 0.0
        
        # Generate advanced features
        features = self.feature_engineer.generate_advanced_features(data)
        
        if len(features) < 50:
            return 0.0
        
        # FIX: Ensure all feature names are strings for scikit-learn
        features.columns = [str(col) for col in features.columns]
        
        # Scale features
        if hasattr(self.scalers[symbol], 'n_features_in_'):
            try:
                features_scaled = self.scalers[symbol].transform(features)
            except ValueError as e:
                # Retrain scaler if feature mismatch
                self.logger.warning(f"Scaler feature mismatch for {symbol}, refitting...")
                features_scaled = self.scalers[symbol].fit_transform(features)
        else:
            features_scaled = self.scalers[symbol].fit_transform(features)
        
        # Neural Network Prediction
        nn_confidence = self._nn_prediction(symbol, features_scaled)
        
        # Ensemble Prediction
        ensemble_confidence = self._ensemble_prediction(symbol, features_scaled)
        
        # Combine predictions with weighting
        final_confidence = (nn_confidence * 0.7 + ensemble_confidence * 0.3)
        
        # Adjust for market regime
        final_confidence = self._adjust_for_market_regime(final_confidence, data)
        
        return float(np.clip(final_confidence, 0.0, 1.0))
        
     except Exception as e:
        self.logger.error(f"Error in prediction for {symbol}: {str(e)}")
        return 0.0
    
    def _nn_prediction(self, symbol: str, features_scaled: np.ndarray) -> float:
        """Get neural network prediction"""
        try:
            # Prepare sequence data for LSTM
            sequence_length = 30
            if len(features_scaled) < sequence_length:
                return 0.0
            
            # Create sequences
            sequences = []
            for i in range(len(features_scaled) - sequence_length + 1):
                sequences.append(features_scaled[i:i + sequence_length])
            
            if not sequences:
                return 0.0
            
            sequences_array = np.array(sequences)
            features_tensor = torch.FloatTensor(sequences_array).to(self.device)
            
            # Model prediction
            self.models[symbol].eval()
            with torch.no_grad():
                action_probs, confidence, volatility, attn_weights = self.models[symbol](features_tensor)
            
            # Use the latest prediction
            latest_confidence = confidence[-1].cpu().numpy()[0]
            
            return latest_confidence
            
        except Exception as e:
            self.logger.error(f"Error in NN prediction: {str(e)}")
            return 0.0
    
    def _ensemble_prediction(self, symbol: str, features_scaled: np.ndarray) -> float:
        """Get ensemble model prediction"""
        try:
            ensemble_models = self.ensemble_models[symbol]
            predictions = []
            weights = [0.3, 0.3, 0.3, 0.1]  # XGB, LGB, CatBoost, Meta
            
            # Individual model predictions
            for i, (name, model) in enumerate(ensemble_models.items()):
                if name == 'meta':
                    continue  # Meta model needs base predictions
                
                if hasattr(model, 'predict_proba'):
                    try:
                        pred = model.predict_proba(features_scaled[-1:].reshape(1, -1))
                        confidence = pred.max() if len(pred) > 0 else 0.0
                        predictions.append(confidence * weights[i])
                    except:
                        predictions.append(0.0)
            
            ensemble_confidence = np.sum(predictions) if predictions else 0.0
            
            return ensemble_confidence
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {str(e)}")
            return 0.0
    
    def _adjust_for_market_regime(self, confidence: float, data: pd.DataFrame) -> float:
        """Adjust confidence based on market regime"""
        try:
            # Calculate market volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend strength
            price_trend = data['close'].rolling(20).mean()
            trend_strength = abs(data['close'].iloc[-1] - price_trend.iloc[-1]) / data['close'].iloc[-1]
            
            # Adjust confidence based on market conditions
            if volatility > 0.02:  # High volatility
                confidence *= 0.8  # Reduce confidence in high volatility
            elif trend_strength > 0.1:  # Strong trend
                confidence *= 1.1  # Increase confidence in strong trends
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Error adjusting for market regime: {str(e)}")
            return confidence
    
    def update_models(self):
        """Update models with new data (online learning)"""
        # This would be called periodically to retrain models with new data
        # For production, this would include:
        # 1. Collecting new training data
        # 2. Retraining models
        # 3. Model validation
        # 4. Model deployment
        
        self.logger.info("Model update scheduled - would retrain with new data")
    
    def _load_pytorch_model(self, model_path: str) -> AdvancedTradingModel:
        """Load PyTorch model from file"""
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            # Return new model if loading fails
            input_size = 45
            return AdvancedTradingModel(input_size=input_size).to(self.device)

class FeatureEngineer:
    """Advanced feature engineering for trading models"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
    
    def generate_advanced_features(self, data: pd.DataFrame) -> np.ndarray:
        """Generate comprehensive feature set for ML models"""
        features = []
        
        # Price-based features
        features.extend(self._price_features(data))
        
        # Volume-based features
        if 'volume' in data.columns:
            features.extend(self._volume_features(data))
        
        # Technical indicators
        features.extend(self.technical_indicators.calculate_all(data))
        
        # Statistical features
        features.extend(self._statistical_features(data))
        
        # Market microstructure features
        features.extend(self._microstructure_features(data))
        
        # Combine all features
        feature_df = pd.concat(features, axis=1).dropna()
        
        return feature_df
    
    def _price_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate price-based features"""
        features = []
        
        # Returns at different horizons
        for period in [1, 5, 10, 20]:
            features.append(data['close'].pct_change(period))
        
        # High-Low range
        features.append((data['high'] - data['low']) / data['close'])
        
        # Price position in daily range
        features.append((data['close'] - data['low']) / (data['high'] - data['low']))
        
        # Momentum features
        features.append(data['close'] / data['close'].rolling(5).mean() - 1)
        features.append(data['close'] / data['close'].rolling(10).mean() - 1)
        features.append(data['close'] / data['close'].rolling(20).mean() - 1)
        
        # Volatility features
        features.append(data['close'].pct_change().rolling(5).std())
        features.append(data['close'].pct_change().rolling(10).std())
        features.append(data['close'].pct_change().rolling(20).std())
        
        return features
    
    def _volume_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate volume-based features"""
        features = []
        
        # Volume changes
        features.append(data['volume'].pct_change())
        features.append(data['volume'] / data['volume'].rolling(20).mean())
        
        # Price-Volume correlation
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        features.append(price_change.rolling(10).corr(volume_change))
        
        return features
    
    def _statistical_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate statistical features"""
        features = []
        
        returns = data['close'].pct_change().dropna()
        
        # Skewness and Kurtosis
        features.append(returns.rolling(20).skew())
        features.append(returns.rolling(20).kurt())
        
        # Quantile features
        features.append(returns.rolling(20).quantile(0.05))
        features.append(returns.rolling(20).quantile(0.95))
        
        # Hurst exponent (simplified)
        features.append(self._calculate_hurst(returns))
        
        return features
    
    def _microstructure_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate market microstructure features"""
        features = []
        
        # Bid-Ask spread proxy (using High-Low as proxy)
        features.append((data['high'] - data['low']) / data['close'])
        
        # Price efficiency (variance ratio)
        features.append(self._variance_ratio(data['close']))
        
        return features
    
    def _calculate_hurst(self, returns: pd.Series, max_lag: int = 20) -> pd.Series:
        """Calculate Hurst exponent (simplified)"""
        # Simplified implementation
        lags = range(2, min(max_lag, len(returns)))
        tau = [np.std(returns.diff(lag).dropna()) for lag in lags]
        
        # Calculate Hurst as slope of log(tau) vs log(lag)
        if len(tau) > 1:
            hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
            return pd.Series([hurst] * len(returns), index=returns.index)
        else:
            return pd.Series([0.5] * len(returns), index=returns.index)
    
    def _variance_ratio(self, prices: pd.Series, lag: int = 5) -> pd.Series:
        """Calculate variance ratio for market efficiency"""
        returns = prices.pct_change().dropna()
        var_1 = returns.rolling(lag).var()
        var_lag = returns.rolling(lag).apply(lambda x: np.var(x[::lag]) if len(x) >= lag else np.nan)
        
        vr = var_lag / (lag * var_1)
        return vr

class TechnicalIndicators:
    """Comprehensive technical indicator calculations"""
    
    def calculate_all(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate all technical indicators"""
        indicators = []
        
        # Moving Averages
        indicators.extend(self._moving_averages(data))
        
        # Oscillators
        indicators.extend(self._oscillators(data))
        
        # Volatility indicators
        indicators.extend(self._volatility_indicators(data))
        
        # Trend indicators
        indicators.extend(self._trend_indicators(data))
        
        # Support/Resistance
        indicators.extend(self._support_resistance(data))
        
        return indicators
    
    def _moving_averages(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate moving averages and crossovers"""
        indicators = []
        
        periods = [5, 10, 20, 50, 100]
        ma_values = {}
        
        for period in periods:
            ma = data['close'].rolling(period).mean()
            ma_values[period] = ma
            indicators.append(ma / data['close'] - 1)  # Price deviation from MA
        
        # MA crossovers
        indicators.append(ma_values[5] / ma_values[20] - 1)  # Fast vs Slow MA
        indicators.append(ma_values[10] / ma_values[50] - 1)
        
        return indicators
    
    def _oscillators(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate oscillator indicators"""
        indicators = []
        
        # RSI
        indicators.append(self._rsi(data['close']))
        
        # MACD
        macd, signal = self._macd(data['close'])
        indicators.append(macd)
        indicators.append(signal)
        indicators.append(macd - signal)  # MACD histogram
        
        # Stochastic
        k, d = self._stochastic(data)
        indicators.append(k)
        indicators.append(d)
        
        # CCI
        indicators.append(self._cci(data))
        
        return indicators
    
    def _volatility_indicators(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate volatility indicators"""
        indicators = []
        
        # ATR
        indicators.append(self._atr(data))
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._bollinger_bands(data)
        indicators.append((data['close'] - bb_lower) / (bb_upper - bb_lower))  #%b
        indicators.append((bb_upper - bb_lower) / bb_middle)  # Band width
        
        return indicators
    
    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to 0-1
    
    def _macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['low'].rolling(k_period).min()
        high_max = data['high'].rolling(k_period).max()
        
        k = 100 * (data['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        
        return k / 100, d / 100  # Normalize to 0-1
    
    def _cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (tp - sma) / (0.015 * mad)
        return cci / 100  # Normalize
    
    def _atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr / data['close']  # Normalize by price
    
    def _bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = data['close'].rolling(period).mean()
        rolling_std = data['close'].rolling(period).std()
        
        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)
        
        return upper, lower, middle
    
    def _trend_indicators(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate trend indicators"""
        indicators = []
        
        # ADX
        indicators.append(self._adx(data))
        
        # Parabolic SAR
        indicators.append(self._parabolic_sar(data))
        
        return indicators
    
    def _adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._atr(data, period) * data['close']  # Convert back to points
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx / 100  # Normalize to 0-1
    
    def _parabolic_sar(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Parabolic SAR (simplified)"""
        # Simplified implementation
        high = data['high']
        low = data['low']
        
        # Simple SAR-like calculation
        sar = (high.rolling(5).max() + low.rolling(5).min()) / 2
        sar_position = (data['close'] - sar) / data['close']
        
        return sar_position
    
    def _support_resistance(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate support/resistance levels"""
        indicators = []
        
        # Pivot points
        pivot = (data['high'] + data['low'] + data['close']) / 3
        r1 = 2 * pivot - data['low']
        s1 = 2 * pivot - data['high']
        
        indicators.append((data['close'] - s1) / (r1 - s1))  # Price position between S1/R1
        
        # Recent support/resistance
        support = data['low'].rolling(20).min()
        resistance = data['high'].rolling(20).max()
        
        indicators.append((data['close'] - support) / (resistance - support))
        
        return indicators