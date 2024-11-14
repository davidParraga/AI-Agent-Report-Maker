import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == '__main__':
    # Paths to the data
    transactions_path = os.path.join('data/raw', 'transactions_data.csv')
    labels_path = os.path.join('data/raw/', 'train_fraud_labels.json')

    # Paths to save the model and encoders
    model_save_path = os.path.join('models', 'fraud_detection_model.pkl')
    le_use_chip_path = os.path.join('models', 'le_use_chip.pkl')
    le_merchant_state_path = os.path.join('models', 'le_merchant_state.pkl')
    le_mcc_path = os.path.join('models', 'le_mcc.pkl')

    transactions_df = pd.read_csv(transactions_path, dtype={'id': str})
    print("Transactions data loaded. Shape:", transactions_df.shape)

    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)

    if "target" in labels_dict:
        labels = labels_dict["target"]
    else:
        raise KeyError("La clave 'target' no se encontró en el archivo JSON de etiquetas.")

    labels_df = pd.DataFrame(list(labels.items()), columns=['id', 'fraud_label'])
    print("Labels data loaded. Shape:", labels_df.shape)

    labels_df['id'] = labels_df['id'].astype(str)

    data = transactions_df.merge(labels_df, on='id')
    print("Merged data shape:", data.shape)
    data['amount'] = data['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data['date'] = pd.to_datetime(data['date'])

    # Extract features from 'date'
    data['hour'] = data['date'].dt.hour
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data['month'] = data['date'].dt.month

    # Fill missing values
    data['merchant_state'] = data['merchant_state'].fillna('Unknown')
    data['use_chip'] = data['use_chip'].fillna('Unknown')
    data['mcc'] = data['mcc'].fillna('Unknown')

    # Label Encoding
    le_use_chip = LabelEncoder()
    data['use_chip_encoded'] = le_use_chip.fit_transform(data['use_chip'])

    le_merchant_state = LabelEncoder()
    data['merchant_state_encoded'] = le_merchant_state.fit_transform(data['merchant_state'])

    le_mcc = LabelEncoder()
    data['mcc_encoded'] = le_mcc.fit_transform(data['mcc'])

    # Feature selection
    X = data[['amount', 'hour', 'day', 'weekday', 'month',
              'use_chip_encoded', 'merchant_state_encoded', 'mcc_encoded']]
    y = data['fraud_label'].map({'No': 0, 'Yes': 1})  # Convert labels to binary

    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print("Label distribution:\n", y.value_counts())

    # Enough clases?
    if y.nunique() < 2:
        raise ValueError("El vector objetivo tiene menos de 2 clases. No se puede realizar la división con estratificación.")

    # Split the data for evaluation
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print("Training set size:", X_train.shape)
        print("Validation set size:", X_val.shape)
    except ValueError as e:
        print("Error en train_test_split:", e)
        # Intentar sin estratificación
        print("Intentando dividir sin estratificación...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Training set size:", X_train.shape)
        print("Validation set size:", X_val.shape)

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model
    y_pred = clf.predict(X_val)
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print('Balanced Accuracy Score:', balanced_accuracy_score(y_val, y_pred))

    # Save the model and encoders
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    joblib.dump(le_use_chip, le_use_chip_path)
    joblib.dump(le_merchant_state, le_merchant_state_path)
    joblib.dump(le_mcc, le_mcc_path)
    print("Model and encoders saved successfully.")