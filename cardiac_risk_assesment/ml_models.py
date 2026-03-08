"""
Machine Learning Models for Cardiac Risk Prediction
Implements Logistic Regression and XGBoost with ensemble
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class CardiacRiskMLModel:
    """
    Machine learning models for cardiac risk prediction.
    Implements both Logistic Regression and XGBoost with ensemble capability.
    """
    
    def __init__(self):
        self.logistic_model = None
        self.xgb_model = None
        self.feature_names = None
        self.risk_thresholds = {
            'low': 0.05,      # < 5% probability
            'intermediate': 0.15,  # 5-15% probability
            'high': float('inf')   # > 15% probability
        }
    
    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        # Train model
        self.logistic_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.logistic_model.fit(X_train, y_train)
        
        # Training performance
        y_train_pred = self.logistic_model.predict(X_train)
        y_train_proba = self.logistic_model.predict_proba(X_train)[:, 1]
        
        print("\nTraining Performance:")
        self._print_metrics(y_train, y_train_pred, y_train_proba)
        
        # Validation performance
        if X_val is not None and y_val is not None:
            y_val_pred = self.logistic_model.predict(X_val)
            y_val_proba = self.logistic_model.predict_proba(X_val)[:, 1]
            
            print("\nValidation Performance:")
            self._print_metrics(y_val, y_val_pred, y_val_proba)
        
        return self.logistic_model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Use early stopping if validation data provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train)
        
        # Training performance
        y_train_pred = self.xgb_model.predict(X_train)
        y_train_proba = self.xgb_model.predict_proba(X_train)[:, 1]
        
        print("\nTraining Performance:")
        self._print_metrics(y_train, y_train_pred, y_train_proba)
        
        # Validation performance
        if X_val is not None and y_val is not None:
            y_val_pred = self.xgb_model.predict(X_val)
            y_val_proba = self.xgb_model.predict_proba(X_val)[:, 1]
            
            print("\nValidation Performance:")
            self._print_metrics(y_val, y_val_pred, y_val_proba)
        
        return self.xgb_model
    
    def predict_ensemble(self, X):
        """
        Make ensemble predictions by averaging both models.
        
        Args:
            X: Feature matrix
            
        Returns:
            dict: Contains probabilities and risk classifications
        """
        if self.logistic_model is None or self.xgb_model is None:
            raise ValueError("Both models must be trained before ensemble prediction")
        
        # Get predictions from both models
        prob_logistic = self.logistic_model.predict_proba(X)[:, 1]
        prob_xgb = self.xgb_model.predict_proba(X)[:, 1]
        
        # Average the probabilities
        prob_ensemble = (prob_logistic + prob_xgb) / 2
        
        # Convert to risk classes
        risk_classes = self._probability_to_risk_class(prob_ensemble)
        
        return {
            'probability_logistic': prob_logistic,
            'probability_xgb': prob_xgb,
            'probability_ensemble': prob_ensemble,
            'risk_class': risk_classes
        }
    
    def predict_single_patient(self, X_patient):
        """
        Predict risk for a single patient.
        
        Args:
            X_patient: Feature vector for one patient (1D array or 2D with shape (1, n_features))
            
        Returns:
            dict: Risk assessment results
        """
        # Ensure 2D shape
        if len(X_patient.shape) == 1:
            X_patient = X_patient.reshape(1, -1)
        
        # Get ensemble prediction
        predictions = self.predict_ensemble(X_patient)
        
        # Get individual probabilities
        prob_ensemble = predictions['probability_ensemble'][0]
        risk_class = predictions['risk_class'][0]
        
        return {
            'ml_risk_probability': float(prob_ensemble),
            'ml_risk_class': risk_class,
            'ml_risk_percentage': float(prob_ensemble * 100),
            'logistic_probability': float(predictions['probability_logistic'][0]),
            'xgb_probability': float(predictions['probability_xgb'][0])
        }
    
    def _probability_to_risk_class(self, probabilities):
        """Convert probability values to risk classifications."""
        risk_classes = []
        for prob in probabilities:
            if prob < self.risk_thresholds['low']:
                risk_classes.append('low')
            elif prob < self.risk_thresholds['intermediate']:
                risk_classes.append('intermediate')
            else:
                risk_classes.append('high')
        return np.array(risk_classes)
    
    def _print_metrics(self, y_true, y_pred, y_proba):
        """Print evaluation metrics."""
        print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    
    def evaluate_test_set(self, X_test, y_test):
        """
        Evaluate ensemble model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "="*60)
        print("TEST SET EVALUATION - ENSEMBLE MODEL")
        print("="*60)
        
        predictions = self.predict_ensemble(X_test)
        prob_ensemble = predictions['probability_ensemble']
        y_pred = (prob_ensemble >= 0.5).astype(int)
        
        self._print_metrics(y_test, y_pred, prob_ensemble)
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Disease', 'Disease']))
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from both models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance from both models
        """
        importance_dict = {}
        
        if self.logistic_model is not None:
            # For logistic regression, use absolute coefficients
            coef = np.abs(self.logistic_model.coef_[0])
            importance_dict['logistic'] = dict(zip(feature_names, coef))
        
        if self.xgb_model is not None:
            # For XGBoost, use feature_importances_
            importance_dict['xgboost'] = dict(
                zip(feature_names, self.xgb_model.feature_importances_)
            )
        
        return importance_dict
    
    def save_models(self, filepath_prefix='cardiac_risk_model'):
        """Save trained models."""
        if self.logistic_model is not None:
            with open(f"{filepath_prefix}_logistic.pkl", 'wb') as f:
                pickle.dump(self.logistic_model, f)
            print(f"Logistic model saved to {filepath_prefix}_logistic.pkl")
        
        if self.xgb_model is not None:
            with open(f"{filepath_prefix}_xgboost.pkl", 'wb') as f:
                pickle.dump(self.xgb_model, f)
            print(f"XGBoost model saved to {filepath_prefix}_xgboost.pkl")
        
        # Save thresholds
        with open(f"{filepath_prefix}_thresholds.pkl", 'wb') as f:
            pickle.dump(self.risk_thresholds, f)
    
    def load_models(self, filepath_prefix='cardiac_risk_model'):
        """Load trained models."""
        try:
            with open(f"{filepath_prefix}_logistic.pkl", 'rb') as f:
                self.logistic_model = pickle.load(f)
            print(f"Logistic model loaded")
        except FileNotFoundError:
            print("Logistic model file not found")
        
        try:
            with open(f"{filepath_prefix}_xgboost.pkl", 'rb') as f:
                self.xgb_model = pickle.load(f)
            print(f"XGBoost model loaded")
        except FileNotFoundError:
            print("XGBoost model file not found")
        
        try:
            with open(f"{filepath_prefix}_thresholds.pkl", 'rb') as f:
                self.risk_thresholds = pickle.load(f)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    print("Cardiac Risk ML Model Module")
    print("="*60)
    print("\nThis module provides:")
    print("1. Logistic Regression model")
    print("2. XGBoost model")
    print("3. Ensemble predictions (average of both)")
    print("4. Risk classification (low/intermediate/high)")