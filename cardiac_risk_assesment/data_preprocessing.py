"""
Data Preprocessing for UCI Heart Disease Dataset
Maps UCI features to cardiac risk features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


class HeartDiseasePreprocessor:
    """
    Preprocesses UCI Heart Disease dataset for cardiac risk modeling.
    
    UCI Dataset columns (from your image):
    - age: Age in years
    - sex: 1=Male, 0=Female
    - cp: Chest pain type (typical angina, asymptomatic, non-anginal, atypical angina)
    - trestbps: Resting blood pressure
    - chol: Cholesterol
    - fbs: Fasting blood sugar > 120 mg/dl (TRUE/FALSE)
    - restecg: Resting ECG results (lv hypertrophy, normal, etc)
    - thalch: Maximum heart rate achieved
    - exang: Exercise induced angina (TRUE/FALSE)
    - oldpeak: ST depression induced by exercise
    - num: Target (0=no disease, 1-4=disease severity)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load the UCI heart disease dataset."""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_target(self, df):
        """
        Create binary target variable.
        num: 0 = no disease, 1-4 = presence of disease
        We convert to binary: 0 = healthy, 1 = disease
        """
        df['target'] = (df['num'] > 0).astype(int)
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # For numeric columns, impute with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical, impute with mode
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional features that map to clinical risk factors.
        """
        df = df.copy()

        # Age risk
        df['age_over_65'] = (df['age'] >= 65).astype(int)

        # Sex encoding (robust to strings & numeric encodings)
        df['is_male'] = (
            df['sex']
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                'male': 1,
                'female': 0,
                '1': 1,
                '0': 0
            })
        )

        if df['is_male'].isna().any():
            raise ValueError(
                f"Unexpected values in sex column: {df['sex'].unique()}"
            )

        # Chest pain → ischemic angina
        df['has_angina'] = (df['cp'].str.lower() == 'typical angina').astype(int)

        # Exercise-induced angina
        df['exang_binary'] = df['exang'].map({
            'TRUE': 1, 'FALSE': 0, True: 1, False: 0
        })

        # Fasting blood sugar (diabetes proxy)
        df['fbs_binary'] = df['fbs'].map({
            'TRUE': 1, 'FALSE': 0, True: 1, False: 0
        })

        # Lipids
        df['high_cholesterol'] = (df['chol'] > 240).astype(int)

        # Blood pressure
        df['hypertension'] = (df['trestbps'] > 140).astype(int)

        # ECG stress markers
        df['lv_hypertrophy'] = (
            df['restecg'].str.lower() == 'lv hypertrophy'
        ).astype(int)

        # Exercise capacity
        df['reduced_exercise_capacity'] = (df['thalch'] < 120).astype(int)

        # ST depression
        df['significant_st_depression'] = (df['oldpeak'] > 2.0).astype(int)

        return df

    
    def select_features(self, df):
        """
        Select features for modeling.
        We'll use a mix of original and engineered features.
        """
        feature_columns = [
            # Demographics
            'age',
            'is_male',
            
            # Cardiac markers
            'has_angina',
            'exang_binary',
            'lv_hypertrophy',
            
            # Vital signs
            'trestbps',
            'thalch',
            'oldpeak',
            
            # Labs
            'chol',
            'fbs_binary',
            
            # Engineered features
            'age_over_65',
            'high_cholesterol',
            'hypertension',
            'reduced_exercise_capacity',
            'significant_st_depression'
        ]
        
        # Keep only features that exist
        available_features = [f for f in feature_columns if f in df.columns]
        self.feature_names = available_features
        
        return df[available_features]
    
    def prepare_data(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """
        Complete preprocessing pipeline and split data.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Create binary target
        df = self.create_target(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features
        X = self.select_features(df)
        y = df['target']
        
        print(f"\nFeatures used ({len(self.feature_names)}):")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i}. {feat}")
        
        print(f"\nTarget distribution:")
        print(f"  No disease (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
        print(f"  Disease (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train.values, y_val.values, y_test.values,
            X_train, X_val, X_test  # Return unscaled versions too for SHAP
        )
    
    def save_preprocessor(self, filepath='preprocessor.pkl'):
        """Save the fitted preprocessor."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.pkl'):
        """Load a fitted preprocessor."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    preprocessor = HeartDiseasePreprocessor()
    df = preprocessor.load_data('heart_diseases.csv')
    # You would load your actual data here
    print("UCI Heart Disease Data Preprocessor")
    print("=" * 60)
    print("\nTo use this preprocessor:")
    print("1. Load your CSV: df = preprocessor.load_data('heart_disease.csv')")
    print("2. Prepare data: X_train, X_val, X_test, y_train, y_val, y_test, ... = preprocessor.prepare_data(df)")
    print("3. Train models with the prepared data")