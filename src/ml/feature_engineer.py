import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from scipy import stats

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger('FeatureEngineer')
        self.feature_names = []
    
    def generate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
     """Generate comprehensive feature set for ML models - FIXED COLUMN NAMES"""
     try:
        features_list = []
        
        # Price-based features
        features_list.extend(self._price_features(data))
        
        # Volume-based features
        if 'volume' in data.columns:
            features_list.extend(self._volume_features(data))
        
        # Technical indicators
        features_list.extend(self._technical_indicators(data))
        
        # Statistical features
        features_list.extend(self._statistical_features(data))
        
        # Market microstructure features
        features_list.extend(self._microstructure_features(data))
        
        # Pattern recognition features
        features_list.extend(self._pattern_features(data))
        
        # Combine all features and ensure proper column names
        if features_list:
            feature_df = pd.concat(features_list, axis=1).dropna()
            
            # FIX: Convert all column names to strings
            feature_df.columns = [str(col) for col in feature_df.columns]
            
            self.feature_names = list(feature_df.columns)
            return feature_df
        else:
            return pd.DataFrame()
            
     except Exception as e:
        self.logger.error(f"Error generating features: {str(e)}")
        return pd.DataFrame()
    
    def _price_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate price-based features"""
        features = []
        price = data['close']
        high = data['high']
        low = data['low']
        
        # Returns at different horizons
        for period in [1, 2, 5, 10, 20]:
            ret = price.pct_change(period)
            ret.name = f'return_{period}'
            features.append(ret)
        
        # Volatility features
        for period in [5, 10, 20]:
            vol = price.pct_change().rolling(period).std()
            vol.name = f'volatility_{period}'
            features.append(vol)
        
        # Price position features
        features.append((price - low) / (high - low))  # Price position in range
        features.append(price / price.rolling(5).mean() - 1)  # Price deviation from MA
        
        # High-Low features
        features.append((high - low) / price)  # Normalized range
        features.append((high - price.shift(1)) / price)  # Gap up
        features.append((price.shift(1) - low) / price)  # Gap down
        
        return features
    
    def _volume_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate volume-based features"""
        features = []
        volume = data['volume']
        price = data['close']
        
        # Volume trends
        features.append(volume / volume.rolling(20).mean())  # Volume ratio
        features.append(volume.pct_change())  # Volume change
        
        # Price-Volume relationship
        price_change = price.pct_change()
        volume_change = volume.pct_change()
        pv_corr = price_change.rolling(10).corr(volume_change)
        pv_corr.name = 'price_volume_correlation'
        features.append(pv_corr)
        
        # Volume volatility
        vol_vol = volume.pct_change().rolling(10).std()
        vol_vol.name = 'volume_volatility'
        features.append(vol_vol)
        
        return features
    
    def _technical_indicators(self, data: pd.DataFrame) -> List[pd.Series]:
        """Calculate technical indicators"""
        features = []
        price = data['close']
        high = data['high']
        low = data['low']
        
        # RSI
        rsi = self._calculate_rsi(price)
        rsi.name = 'rsi'
        features.append(rsi)
        
        # MACD
        macd, signal = self._calculate_macd(price)
        macd.name = 'macd'
        features.append(macd)
        features.append(macd - signal)  # MACD histogram
        
        # Bollinger Bands
        bb_position, bb_width = self._calculate_bollinger_bands(price)
        features.append(bb_position)
        features.append(bb_width)
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(data)
        features.append(stoch_k)
        features.append(stoch_d)
        
        # ATR
        atr = self._calculate_atr(data)
        atr.name = 'atr'
        features.append(atr)
        
        # ADX
        adx = self._calculate_adx(data)
        adx.name = 'adx'
        features.append(adx)
        
        # CCI
        cci = self._calculate_cci(data)
        cci.name = 'cci'
        features.append(cci)
        
        return features
    
    def _statistical_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate statistical features"""
        features = []
        price = data['close']
        returns = price.pct_change().dropna()
        
        if len(returns) < 20:
            return features
        
        # Rolling statistics
        for period in [10, 20, 50]:
            # Skewness
            skew = returns.rolling(period).skew()
            skew.name = f'skewness_{period}'
            features.append(skew)
            
            # Kurtosis
            kurt = returns.rolling(period).kurt()
            kurt.name = f'kurtosis_{period}'
            features.append(kurt)
            
            # Quantiles
            q05 = returns.rolling(period).quantile(0.05)
            q95 = returns.rolling(period).quantile(0.95)
            features.append(q05)
            features.append(q95)
        
        # Hurst exponent
        hurst = self._calculate_hurst_exponent(returns)
        features.append(hurst)
        
        # Variance ratio
        vr = self._variance_ratio(returns)
        features.append(vr)
        
        return features
    
    def _microstructure_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate market microstructure features"""
        features = []
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Bid-Ask spread proxy
        spread_proxy = (high - low) / close
        spread_proxy.name = 'spread_proxy'
        features.append(spread_proxy)
        
        # Price efficiency measures
        # Roll spread estimator
        price_changes = close.diff().dropna()
        if len(price_changes) >= 2:
            roll_spread = 2 * np.sqrt(-price_changes.rolling(2).cov())
            roll_spread.name = 'roll_spread'
            features.append(roll_spread)
        
        # Price impact measures
        range_efficiency = (close - low) / (high - low)
        range_efficiency.name = 'range_efficiency'
        features.append(range_efficiency)
        
        return features
    
    def _pattern_features(self, data: pd.DataFrame) -> List[pd.Series]:
        """Generate pattern recognition features"""
        features = []
        price = data['close']
        high = data['high']
        low = data['low']
        
        # Support and Resistance levels
        resistance = high.rolling(20).max()
        support = low.rolling(20).min()
        
        features.append((price - support) / (resistance - support))  # S/R position
        features.append(resistance - support)  # Trading range
        
        # Breakout detection
        breakout_strength = (price - resistance) / (resistance - support)
        breakout_strength.name = 'breakout_strength'
        features.append(breakout_strength)
        
        # Mean reversion indicators
        z_score = (price - price.rolling(20).mean()) / price.rolling(20).std()
        z_score.name = 'price_z_score'
        features.append(z_score)
        
        return features
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to 0-1
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series]:
        middle = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        bb_position = (prices - lower) / (upper - lower)  # %B
        bb_width = (upper - lower) / middle  # Band width
        
        return bb_position, bb_width
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = data['low'].rolling(k_period).min()
        high_max = data['high'].rolling(k_period).max()
        
        k = 100 * (data['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        
        return k / 100, d / 100  # Normalize to 0-1
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr / data['close']  # Normalize by price
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        # Simplified ADX calculation
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(data, period) * data['close']  # Convert back to points
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx / 100  # Normalize to 0-1
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci / 100  # Normalize
    
    def _calculate_hurst_exponent(self, returns: pd.Series, max_lag: int = 20) -> pd.Series:
        """Calculate Hurst exponent for market efficiency"""
        hurst_values = []
        
        for i in range(len(returns)):
            if i < max_lag:
                hurst_values.append(0.5)
                continue
            
            window_returns = returns.iloc[:i+1]
            lags = range(2, min(max_lag, len(window_returns)))
            
            if len(lags) < 2:
                hurst_values.append(0.5)
                continue
            
            tau = [np.std(window_returns.diff(lag).dropna()) for lag in lags]
            
            if len(tau) > 1:
                hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
                hurst_values.append(hurst)
            else:
                hurst_values.append(0.5)
        
        return pd.Series(hurst_values, index=returns.index)
    
    def _variance_ratio(self, returns: pd.Series, lag: int = 5) -> pd.Series:
        """Calculate variance ratio for random walk testing"""
        var_1 = returns.rolling(lag).var()
        var_lag = returns.rolling(lag).apply(lambda x: np.var(x[::lag]) if len(x) >= lag else np.nan)
        
        vr = var_lag / (lag * var_1)
        return vr
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names