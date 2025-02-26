import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import datetime, timedelta

class LoadForecaster:
    def __init__(self):
        self.model = None
        self.feature_cols = None
    
    def generate_features(self, df):
        """Generate time-based and lagged features for load forecasting"""
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('datetime')
            
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Create lagged features for load
        df['load_lag_24'] = df['load'].shift(24)  # Previous day
        df['load_lag_168'] = df['load'].shift(168)  # Previous week
        
        # Rolling means for load
        df['load_rolling_24h'] = df['load'].rolling(window=24).mean()
        df['load_rolling_168h'] = df['load'].rolling(window=168).mean()
        
        # Weather features (assuming temperature, windspeed, and ssrd are in columns)
        # Rolling means for weather variables
        df['temp_rolling_24h'] = df['temperature'].rolling(window=24).mean()
        df['wind_rolling_24h'] = df['windspeed'].rolling(window=24).mean()
        df['ssrd_rolling_24h'] = df['ssrd'].rolling(window=24).mean()
        
        return df
    
    def prepare_features(self, df):
        """Prepare final feature set for model"""
        self.feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'windspeed', 'ssrd',
            'temp_rolling_24h', 'wind_rolling_24h', 'ssrd_rolling_24h',
            'load_lag_24', 'load_lag_168',
            'load_rolling_24h', 'load_rolling_168h'
        ]
        
        return df[self.feature_cols]
    
    def train(self, train_df, valid_df=None, params=None):
        """Train the CatBoost model"""
        
        # Generate features
        train_df = self.generate_features(train_df.copy())
        if valid_df is not None:
            valid_df = self.generate_features(valid_df.copy())
        
        # Remove rows with NaN values (caused by lagged features)
        train_df = train_df.dropna()
        if valid_df is not None:
            valid_df = valid_df.dropna()
        
        # Prepare features
        X_train = self.prepare_features(train_df)
        y_train = train_df['load']
        
        if valid_df is not None:
            X_valid = self.prepare_features(valid_df)
            y_valid = valid_df['load']
            eval_set = [(X_valid, y_valid)]
        else:
            eval_set = None
        
        # Default CatBoost parameters
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': 100
        }
        
        # Update with custom parameters if provided
        if params:
            default_params.update(params)
        
        # Initialize and train model
        self.model = CatBoostRegressor(**default_params)
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            use_best_model=True
        )
        
        return self
    
    def predict(self, df):
        """Generate predictions for input data"""
        df = self.generate_features(df.copy())
        X = self.prepare_features(df)
        return self.model.predict(X)
    
    def evaluate(self, test_df):
        """Evaluate model performance"""
        y_true = test_df['load']
        y_pred = self.predict(test_df)
        
        results = {
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        return results
