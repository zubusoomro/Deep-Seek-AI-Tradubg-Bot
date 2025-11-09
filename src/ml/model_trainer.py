import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingDataset(Dataset):
    """PyTorch Dataset for trading data with sequences"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 30):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Validate inputs
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data for sequence length {sequence_length}")
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
            
        # Get sequence of features
        features_seq = self.features[idx:idx + self.sequence_length]
        
        # Get target for the next timestep
        target = self.targets[idx + self.sequence_length]
        
        return (
            torch.FloatTensor(features_seq),  # Shape: (sequence_length, n_features)
            torch.LongTensor([target])        # Shape: (1,)
        )

class AdvancedModelTrainer:
    """Complete model trainer for trading ML models"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('ModelTrainer')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"ModelTrainer initialized with device: {self.device}")
        
        # Training history
        self.training_history = {}
        
    def prepare_training_data(self, features: pd.DataFrame, 
                            price_data: pd.DataFrame, 
                            lookforward: int = 5,
                            return_threshold: float = 0.002) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with proper labeling and feature names"""
        try:
            self.logger.info("Preparing training data...")
            
            # Calculate future returns
            future_returns = price_data['close'].pct_change(lookforward).shift(-lookforward)
            
            # Create multi-class targets: 2=BUY, 1=HOLD, 0=SELL
            targets = np.zeros(len(future_returns))
            
            # BUY signal: strong positive return
            targets[(future_returns > return_threshold)] = 2
            
            # SELL signal: strong negative return  
            targets[(future_returns < -return_threshold)] = 0
            
            # HOLD signal: neutral (everything else)
            targets[(future_returns >= -return_threshold) & (future_returns <= return_threshold)] = 1
            
            # Convert to integer
            targets = targets.astype(int)
            
            # Align features and targets
            aligned_data = pd.concat([features, pd.Series(targets, index=price_data.index, name='target')], 
                                   axis=1).dropna()
            
            # Get feature columns
            feature_cols = [col for col in aligned_data.columns if col != 'target']
            feature_names = feature_cols
            
            X = aligned_data[feature_cols].values
            y = aligned_data['target'].values
            
            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            self.logger.info(f"Class distribution: {class_dist}")
            
            self.logger.info(f"Training data prepared: {X.shape} features, {y.shape} targets")
            
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([]), []
    
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray, 
                          sequence_length: int = 30, 
                          batch_size: int = 32,
                          validation_split: float = 0.2,
                          test_split: float = 0.1) -> Dict[str, DataLoader]:
        """Create train, validation, and test data loaders"""
        try:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42, shuffle=False
            )
            
            # Second split: train and validation
            val_size = validation_split / (1 - test_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, shuffle=False
            )
            
            # Create datasets
            train_dataset = TradingDataset(X_train, y_train, sequence_length)
            val_dataset = TradingDataset(X_val, y_val, sequence_length)
            test_dataset = TradingDataset(X_test, y_test, sequence_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            
            self.logger.info(f"Data loaders created: "
                           f"Train: {len(train_loader)} batches, "
                           f"Val: {len(val_loader)} batches, "
                           f"Test: {len(test_loader)} batches")
            
            return {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
        except Exception as e:
            self.logger.error(f"Error creating data loaders: {str(e)}")
            return {}
    
    def train_neural_network(self, model: nn.Module, 
                           data_loaders: Dict[str, DataLoader],
                           epochs: int = 100,
                           learning_rate: float = 0.001) -> Dict[str, Any]:
        """Complete neural network training with advanced monitoring"""
        try:
            self.logger.info("Starting neural network training...")
            
            # Training setup
            criterion = nn.CrossEntropyLoss()  # For multi-class classification
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'learning_rate': [],
                'epoch_times': []
            }
            
            model.to(self.device)
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 25
            
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_features, batch_targets in data_loaders['train']:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device).squeeze()
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    action_probs, confidence, volatility, _ = model(batch_features)
                    
                    # Calculate loss
                    loss = criterion(action_probs, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(action_probs.data, 1)
                    train_total += batch_targets.size(0)
                    train_correct += (predicted == batch_targets).sum().item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_val_predictions = []
                all_val_targets = []
                
                with torch.no_grad():
                    for batch_features, batch_targets in data_loaders['val']:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device).squeeze()
                        
                        action_probs, confidence, volatility, _ = model(batch_features)
                        loss = criterion(action_probs, batch_targets)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(action_probs.data, 1)
                        val_total += batch_targets.size(0)
                        val_correct += (predicted == batch_targets).sum().item()
                        
                        all_val_predictions.extend(predicted.cpu().numpy())
                        all_val_targets.extend(batch_targets.cpu().numpy())
                
                # Calculate metrics
                train_loss_epoch = train_loss / len(data_loaders['train'])
                val_loss_epoch = val_loss / len(data_loaders['val'])
                train_acc_epoch = train_correct / train_total
                val_acc_epoch = val_correct / val_total
                
                # Update history
                history['train_loss'].append(train_loss_epoch)
                history['val_loss'].append(val_loss_epoch)
                history['train_acc'].append(train_acc_epoch)
                history['val_acc'].append(val_acc_epoch)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                history['epoch_times'].append(time.time() - epoch_start_time)
                
                # Learning rate scheduling
                scheduler.step(val_loss_epoch)
                
                # Early stopping check
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'history': history
                    }, 'models/best_model.pth')
                    self.logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Logging
                if epoch % 10 == 0 or epoch == epochs - 1:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs}: "
                        f"Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, "
                        f"Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                        f"Patience: {patience_counter}/{patience}"
                    )
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Load best model
            checkpoint = torch.load('models/best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Final evaluation
            final_metrics = self.evaluate_model(model, data_loaders['test'])
            
            # Update history with final metrics
            history['final_metrics'] = final_metrics
            history['best_epoch'] = checkpoint['epoch']
            history['best_val_loss'] = best_val_loss
            
            self.training_history = history
            self.logger.info("Neural network training completed successfully")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error in neural network training: {str(e)}")
            return {}
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_confidence = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device).squeeze()
                
                action_probs, confidence, volatility, _ = model(batch_features)
                
                # Get predictions
                _, predicted = torch.max(action_probs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
                all_confidence.extend(confidence.cpu().numpy())
                all_probabilities.extend(action_probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Additional metrics
        confidence_mean = np.mean(all_confidence)
        confidence_std = np.std(all_confidence)
        
        # Class-wise metrics
        class_report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'class_report': class_report
        }
        
        self.logger.info(f"Model Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray, 
                            ensemble_models: Dict, 
                            test_size: float = 0.2) -> Dict[str, Any]:
        """Train ensemble of traditional ML models"""
        try:
            self.logger.info("Training ensemble models...")
            
            # Split data (no sequence for traditional models)
            X_flat = X.reshape(X.shape[0], -1)  # Flatten sequences for traditional models
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_flat, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            trained_models = {}
            model_performance = {}
            
            for name, model in ensemble_models.items():
                try:
                    self.logger.info(f"Training {name}...")
                    
                    # Train model
                    if hasattr(model, 'fit'):
                        model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_scaled)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    model_performance[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    
                    trained_models[name] = {
                        'model': model,
                        'scaler': scaler,
                        'performance': model_performance[name]
                    }
                    
                    self.logger.info(
                        f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
                    )
                    
                    # Save model
                    model_path = f'models/ensemble_{name}.pkl'
                    joblib.dump(trained_models[name], model_path)
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue
            
            # Ensemble performance
            ensemble_results = {
                'trained_models': trained_models,
                'performance': model_performance
            }
            
            self.logger.info("Ensemble training completed")
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error training ensemble models: {str(e)}")
            return {}
    
    def cross_validate_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                           n_splits: int = 5, sequence_length: int = 30,
                           epochs: int = 50) -> Dict[str, Any]:
        """Time-series cross-validation"""
        try:
            self.logger.info(f"Starting {n_splits}-fold time series cross-validation...")
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            fold_models = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                self.logger.info(f"Fold {fold + 1}/{n_splits}")
                
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create data loaders for this fold
                data_loaders = self.create_data_loaders(
                    X_train, y_train, sequence_length=sequence_length,
                    validation_split=0.2, test_split=0.0
                )
                
                if not data_loaders:
                    continue
                
                # Clone model for this fold
                fold_model = self._clone_model(model)
                
                # Train on this fold
                fold_history = self.train_neural_network(
                    fold_model, data_loaders, epochs=epochs
                )
                
                if fold_history:
                    # Evaluate on validation set
                    val_dataset = TradingDataset(X_val, y_val, sequence_length)
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                    
                    fold_metrics = self.evaluate_model(fold_model, val_loader)
                    cv_scores.append(fold_metrics)
                    
                    fold_models.append({
                        'model': fold_model,
                        'history': fold_history,
                        'metrics': fold_metrics
                    })
            
            # Aggregate results
            if cv_scores:
                avg_scores = {
                    'accuracy': np.mean([s['accuracy'] for s in cv_scores]),
                    'precision': np.mean([s['precision'] for s in cv_scores]),
                    'recall': np.mean([s['recall'] for s in cv_scores]),
                    'f1_score': np.mean([s['f1_score'] for s in cv_scores]),
                    'confidence_mean': np.mean([s['confidence_mean'] for s in cv_scores]),
                    'cv_scores': cv_scores,
                    'fold_models': fold_models
                }
                
                self.logger.info(f"Cross-validation completed: "
                               f"Avg Accuracy: {avg_scores['accuracy']:.4f}, "
                               f"Avg F1: {avg_scores['f1_score']:.4f}")
                
                return avg_scores
            else:
                self.logger.warning("No valid cross-validation results")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            return {}
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a clone of the model with same architecture"""
        model_class = type(model)
        clone = model_class(**self._get_model_args(model))
        clone.load_state_dict(model.state_dict().copy())
        return clone
    
    def _get_model_args(self, model: nn.Module) -> Dict[str, Any]:
        """Get model initialization arguments"""
        # This is a simplified version - you might need to adjust based on your model
        if hasattr(model, 'input_size'):
            return {
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout': model.dropout
            }
        return {}
    
    def plot_training_history(self, history: Dict[str, Any], save_path: str = None):
        """Comprehensive training visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Loss plot
            ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
            ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
            ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Learning rate plot
            ax3.plot(history['learning_rate'], color='red', linewidth=2)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            
            # Metrics bar plot
            if 'final_metrics' in history:
                metrics = history['final_metrics']
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                metric_values = [
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0)
                ]
                
                bars = ax4.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
                ax4.set_title('Final Model Metrics', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Score')
                ax4.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Training visualization saved to {save_path}")
            else:
                plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")
    
    def save_model(self, model: nn.Module, model_name: str, metadata: Dict = None):
        """Save model with metadata"""
        try:
            os.makedirs('models', exist_ok=True)
            
            model_path = f'models/{model_name}.pth'
            
            save_data = {
                'model_state_dict': model.state_dict(),
                'model_class': type(model).__name__,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            torch.save(save_data, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path: str, model_class: nn.Module = None) -> Optional[nn.Module]:
        """Load trained model"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if model_class is None:
                # You would need to implement model class resolution
                self.logger.error("Model class required for loading")
                return None
            
            model = model_class(**checkpoint.get('metadata', {}))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            self.training_history = checkpoint.get('training_history', {})
            
            self.logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

# Required import for time tracking
import time

# Make sure the models directory exists
os.makedirs('models', exist_ok=True)