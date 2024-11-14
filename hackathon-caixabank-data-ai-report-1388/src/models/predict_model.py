import pandas as pd
import numpy as np
import os
import joblib
import json

if __name__ == '__main__':
    transactions_path = os.path.join('data', 'raw', 'transactions_data.csv')
    predictions_3_input_path = os.path.join('predictions', 'predictions_3.json')
    model_path = os.path.join('models', 'fraud_detection_model.pkl')
    le_use_chip_path = os.path.join('models', 'le_use_chip.pkl')
    le_merchant_state_path = os.path.join('models', 'le_merchant_state.pkl')
    le_mcc_path = os.path.join('models', 'le_mcc.pkl')

    # Output path
    predictions_3_output_path = os.path.join('predictions', 'predictions_3.json')

    clf = joblib.load(model_path)
    le_use_chip = joblib.load(le_use_chip_path)
    le_merchant_state = joblib.load(le_merchant_state_path)
    le_mcc = joblib.load(le_mcc_path)

    # Read the transaction IDs to predict
    with open(predictions_3_input_path, 'r') as f:
        predictions_input = json.load(f)

    transaction_ids = list(predictions_input['target'].keys())
    transactions_df = pd.read_csv(transactions_path, dtype={'id': str})
    data_to_predict = transactions_df[transactions_df['id'].isin(transaction_ids)].copy()

    # Preprocessing
    data_to_predict['amount'] = data_to_predict['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data_to_predict['date'] = pd.to_datetime(data_to_predict['date'])
    data_to_predict['hour'] = data_to_predict['date'].dt.hour
    data_to_predict['day'] = data_to_predict['date'].dt.day
    data_to_predict['weekday'] = data_to_predict['date'].dt.weekday
    data_to_predict['month'] = data_to_predict['date'].dt.month

    # Fill missing values
    data_to_predict['merchant_state'] = data_to_predict['merchant_state'].fillna('Unknown')
    data_to_predict['use_chip'] = data_to_predict['use_chip'].fillna('Unknown')
    data_to_predict['mcc'] = data_to_predict['mcc'].fillna('Unknown')

    # Replace unseen categories with 'Unknown'
    def replace_unseen_categories(value, encoder_classes):
        if value in encoder_classes:
            return value
        else:
            return 'Unknown'

    data_to_predict['use_chip'] = data_to_predict['use_chip'].apply(lambda x: replace_unseen_categories(x, le_use_chip.classes_))
    data_to_predict['merchant_state'] = data_to_predict['merchant_state'].apply(lambda x: replace_unseen_categories(x, le_merchant_state.classes_))
    data_to_predict['mcc'] = data_to_predict['mcc'].apply(lambda x: replace_unseen_categories(x, le_mcc.classes_))

    # Label Encoding
    data_to_predict['use_chip_encoded'] = le_use_chip.transform(data_to_predict['use_chip'])
    data_to_predict['merchant_state_encoded'] = le_merchant_state.transform(data_to_predict['merchant_state'])
    data_to_predict['mcc_encoded'] = le_mcc.transform(data_to_predict['mcc'])

    # Feature selection
    X_to_predict = data_to_predict[['amount', 'hour', 'day', 'weekday', 'month',
                                    'use_chip_encoded', 'merchant_state_encoded', 'mcc_encoded']]

    y_pred = clf.predict(X_to_predict)
    prediction_labels = np.where(y_pred == 1, 'Yes', 'No')
    predictions_dict = dict(zip(data_to_predict['id'], prediction_labels))

    # Handle transaction IDs not found in the data (assign 'No' or a default value)
    missing_ids = set(transaction_ids) - set(data_to_predict['id'])
    for missing_id in missing_ids:
        predictions_dict[missing_id] = 'No'  # or another default value

    output_dict = {"target": predictions_dict}
    with open(predictions_3_output_path, 'w') as f:
        json.dump(output_dict, f)

    print(f"Predictions saved to {predictions_3_output_path}")
