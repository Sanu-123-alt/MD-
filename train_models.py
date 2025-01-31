import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_diabetes_model():
    print("\nTraining Diabetes Model...")
    # Load the dataset
    df = pd.read_csv('datasets/diabetes.csv')
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and print accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save both the model and scaler
    joblib.dump(model, 'diabetespred_model.sav')
    joblib.dump(scaler, 'diabetes_scaler.sav')

def train_heart_disease_model():
    print("\nTraining Heart Disease Model...")
    # Load the dataset
    df = pd.read_csv('datasets/heart.csv')
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and print accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save both the model and scaler
    joblib.dump(model, 'heartdisease_model.sav')
    joblib.dump(scaler, 'heart_scaler.sav')

def train_parkinsons_model():
    print("\nTraining Parkinsons Model...")
    # Load the dataset
    df = pd.read_csv('datasets/parkinsons.csv')
    
    # Separate features and target
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and print accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Parkinsons Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save both the model and scaler
    joblib.dump(model, 'parkinsons_model.sav')
    joblib.dump(scaler, 'parkinsons_scaler.sav')

if __name__ == "__main__":
    print("Starting model training...")
    
    # Train all models
    train_diabetes_model()
    train_heart_disease_model()
    train_parkinsons_model()
    
    print("\nAll models have been trained and saved successfully!")
