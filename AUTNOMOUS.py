import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Simulated product dataset (length, weight, color_score, pass/fail)
def get_data():
    data = {
        'length': [10.1, 9.9, 10.0, 15.0, 9.8, 12.0, 11.5, 10.2],
        'weight': [100, 98, 102, 200, 97, 180, 190, 101],
        'color_score': [0.95, 0.9, 0.96, 0.5, 0.92, 0.45, 0.48, 0.94],
        'label': [1, 1, 1, 0, 1, 0, 0, 1]  # 1 = Pass, 0 = Fail
    }
    return pd.DataFrame(data)

# Train and save model
def train_model():
    df = get_data()
    X = df[['length', 'weight', 'color_score']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Model trained. Evaluation:")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, 'qc_model.pkl')
    print("Model saved to qc_model.pkl")

# Predict quality
def predict(length, weight, color_score):
    if not os.path.exists('qc_model.pkl'):
        print("Model not found. Training now...")
        train_model()

    model = joblib.load('qc_model.pkl')
    prediction = model.predict([[length, weight, color_score]])[0]
    return "PASS" if prediction == 1 else "FAIL"

# Simple interface
def run():
    print("=== AI-Driven Quality Control ===")

    while True:
        print("\nOptions:\n1. Train Model\n2. Check Product Quality\n3. Exit")
        option = input("Choose (1/2/3): ")

        if option == '1':
            train_model()

        elif option == '2':
            try:
                length = float(input("Enter length: "))
                weight = float(input("Enter weight: "))
                color_score = float(input("Enter color score (0-1): "))
                result = predict(length, weight, color_score)
                print(f"Product Quality Result: {result}")
            except ValueError:
                print("Invalid input. Please enter numeric values.")

        elif option == '3':
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")

if _name_ == "_main_":
    run()