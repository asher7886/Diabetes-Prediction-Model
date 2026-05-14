import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set seed for reproducibility
np.random.seed(42)

def generate_data(n_samples=5000):
    print(f"Generating {n_samples} samples of clinical and lifestyle data...")
    
    # Features common in diabetes datasets
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'bmi': np.random.uniform(18.5, 45.0, n_samples),
        'blood_pressure': np.random.randint(60, 180, n_samples),
        'glucose_level': np.random.randint(70, 250, n_samples),
        'hemoglobin_a1c': np.random.uniform(4.0, 12.0, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'physical_activity': np.random.randint(0, 7, n_samples), # days per week
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'family_history': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on clinical logic
    # Diabetic: A1C >= 6.5 or Glucose >= 126
    # Pre-Diabetic: 5.7 <= A1C < 6.5 or 100 <= Glucose < 126
    # Non-Diabetic: A1C < 5.7 and Glucose < 100
    
    def classify(row):
        if row['hemoglobin_a1c'] >= 6.5 or row['glucose_level'] >= 126:
            return 'Diabetic'
        elif (5.7 <= row['hemoglobin_a1c'] < 6.5) or (100 <= row['glucose_level'] < 126):
            return 'Pre-Diabetic'
        else:
            return 'Non-Diabetic'
            
    df['status'] = df.apply(classify, axis=1)
    
    # Add some noise to make it realistic for ML
    noise_mask = np.random.random(n_samples) < 0.1
    df.loc[noise_mask, 'status'] = np.random.choice(['Diabetic', 'Pre-Diabetic', 'Non-Diabetic'], size=noise_mask.sum())
    
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    
    # 1. Handle Missing Values (Simulate some missingness first)
    df.loc[np.random.choice(df.index, 50), 'bmi'] = np.nan
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    # 2. Encode Categorical Features
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])
    df['family_history'] = le.fit_transform(df['family_history'])
    
    # 3. Target Encoding
    target_le = LabelEncoder()
    df['target'] = target_le.fit_transform(df['status'])
    # Store mapping: 0: Diabetic, 1: Non-Diabetic, 2: Pre-Diabetic (alphabetical)
    # Let's fix order: Non-Diabetic: 0, Pre-Diabetic: 1, Diabetic: 2
    status_map = {'Non-Diabetic': 0, 'Pre-Diabetic': 1, 'Diabetic': 2}
    df['target'] = df['status'].map(status_map)
    
    # 4. Split and Normalize
    X = df.drop(['status', 'target'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler, status_map

if __name__ == "__main__":
    df = generate_data()
    df.to_csv('diabetes_clinical_data.csv', index=False)
    print("Data saved to diabetes_clinical_data.csv")
    
    # Basic EDA stats
    print("\nTarget Distribution:")
    print(df['status'].value_counts())
    
    X_train, X_test, y_train, y_test, features, scaler, smap = preprocess_data(df)
    
    # Save processed data for modeling
    import joblib
    data_bundle = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features,
        'scaler': scaler,
        'status_map': smap
    }
    joblib.dump(data_bundle, 'processed_data.joblib')
    print("Processed data bundle saved.")
