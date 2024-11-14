import pandas as pd
import numpy as np
import os
import json
from statsmodels.tsa.holtwinters import ExponentialSmoothing

if __name__ == '__main__':
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    transactions_path = os.path.join(base_dir, 'data', 'raw', 'transactions_data.csv')
    predictions_4_input_path = os.path.join(base_dir, 'predictions', 'predictions_4.json')
    predictions_4_output_path = os.path.join(base_dir, 'predictions', 'predictions_4.json')

    transactions_df = pd.read_csv(transactions_path, dtype={'id': str})
    transactions_df['amount'] = transactions_df['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    expenses_df = transactions_df[transactions_df['amount'] < 0].copy()
    expenses_df['year_month'] = expenses_df['date'].dt.to_period('M').dt.strftime('%Y-%m')
    client_monthly_expenses = expenses_df.groupby(['client_id', 'year_month'])['amount'].sum().reset_index()

    # Read the clients to predict and the months
    with open(predictions_4_input_path, 'r') as f:
        predictions_input = json.load(f)

    output_dict = {"target": {}}

    # For each client, forecast the expenses for the months specified
    for client_id, months_dict in predictions_input['target'].items():
        client_id_int = int(client_id)
        client_data = client_monthly_expenses[client_monthly_expenses['client_id'] == client_id_int].copy()

        if client_data.empty:
            # If no data for this client, output zeros
            output_dict['target'][client_id] = {month: 0 for month in months_dict.keys()}
            continue

        client_data = client_data.sort_values('year_month')

        # Set 'year_month' as index
        client_data.set_index('year_month', inplace=True)

        # Prepare data for forecasting
        y = client_data['amount']

        # Check if we have enough data points
        if len(y) < 3:
            # Not enough data to build a model, use mean of available data
            forecast_values = {month: y.mean() for month in months_dict.keys()}
        else:
            # Build forecasting model
            try:
                model = ExponentialSmoothing(y.astype(float), trend='add', seasonal=None, damped_trend=False)
                model_fit = model.fit(optimized=True)
                # Forecast for required months
                required_months = list(months_dict.keys())
                forecast_index = pd.PeriodIndex(required_months, freq='M').strftime('%Y-%m')
                forecast = model_fit.predict(start=forecast_index[0], end=forecast_index[-1])
                forecast_series = pd.Series(forecast.values, index=forecast_index)
                forecast_values = forecast_series.reindex(required_months, fill_value=y.mean()).to_dict()
            except Exception as e:
                # If model fails, use mean
                forecast_values = {month: y.mean() for month in months_dict.keys()}

        # Convert to regular floats (not numpy types) to ensure JSON serialization
        forecast_values = {k: float(round(v, 2)) for k, v in forecast_values.items()}

        # Add to output dictionary
        output_dict['target'][client_id] = forecast_values

    # Save the predictions
    with open(predictions_4_output_path, 'w') as f:
        json.dump(output_dict, f)

    print(f"Predictions saved to {predictions_4_output_path}")
