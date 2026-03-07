import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def create_dataset(filename="data/dataset.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.random.seed(42)
    n_samples = 200
    
    # Generate realistic data
    study_hours = np.random.normal(5, 2, n_samples)
    attendance = np.random.normal(85, 10, n_samples)
    previous_marks = np.random.normal(70, 15, n_samples)
    assignments = np.random.randint(0, 11, n_samples)
    
    # Keep them within realistic boundaries
    study_hours = np.clip(study_hours, 0, 15)
    attendance = np.clip(attendance, 0, 100)
    previous_marks = np.clip(previous_marks, 0, 100)
    
    # Introduce some noise and relationship
    score = (study_hours * 5) + (attendance * 0.5) + (previous_marks * 0.4) + (assignments * 2)
    score += np.random.normal(0, 10, n_samples)
    
    # Threshold for pass/fail
    threshold = np.median(score)
    result = (score >= threshold).astype(int)
    
    df = pd.DataFrame({
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PreviousMarks': previous_marks,
        'Assignments': assignments,
        'Result': result
    })
    
    # Introduce missing values for preprocessing requirement
    df.loc[5:10, 'StudyHours'] = np.nan
    df.loc[20:25, 'Attendance'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Dataset created with {n_samples} samples at {filename}")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Handle missing values
    df['StudyHours'].fillna(df['StudyHours'].mean(), inplace=True)
    df['Attendance'].fillna(df['Attendance'].mean(), inplace=True)
    
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We should save scaler as well since we need it for prediction
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_scaled, y

def perform_eda(df):
    os.makedirs('data', exist_ok=True)
    print("Performing EDA...")
    
    # Study Hours vs Result
    plt.figure(figsize=(8, 6))
    plt.scatter(df['StudyHours'], df['Result'], alpha=0.5, color='blue')
    plt.title('Study Hours vs Result')
    plt.xlabel('Study Hours')
    plt.ylabel('Result (1=Pass, 0=Fail)')
    plt.savefig('data/StudyHours_vs_Result.png')
    plt.close()
    
    # Attendance vs Result
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Attendance'], df['Result'], alpha=0.5, color='green')
    plt.title('Attendance vs Result')
    plt.xlabel('Attendance (%)')
    plt.ylabel('Result (1=Pass, 0=Fail)')
    plt.savefig('data/Attendance_vs_Result.png')
    plt.close()
    
    # Previous Marks vs Result
    plt.figure(figsize=(8, 6))
    plt.scatter(df['PreviousMarks'], df['Result'], alpha=0.5, color='purple')
    plt.title('Previous Marks vs Result')
    plt.xlabel('Previous Marks')
    plt.ylabel('Result (1=Pass, 0=Fail)')
    plt.savefig('data/PreviousMarks_vs_Result.png')
    plt.close()
    print("EDA graphs saved in 'data' folder.")

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    print("Training and comparing models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy {best_acc:.4f}")
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/student_model.pkl")
    print("Saved best model as models/student_model.pkl")

if __name__ == "__main__":
    df = create_dataset()
    perform_eda(df)
    X, y = preprocess_data(df)
    train_and_evaluate(X, y)
