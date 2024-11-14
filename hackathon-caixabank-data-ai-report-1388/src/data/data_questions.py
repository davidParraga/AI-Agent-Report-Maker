import pandas as pd
from src.data.api_calls import *
import time


def question_1(cards_df):
    """
    Q1: - The `card_id` with the latest expiry date and the lowest credit limit amount.
    """
    cards_df['expires'] = pd.to_datetime(cards_df['expires'])
    latest_card = cards_df.sort_values(by=['expires', 'credit_limit'], ascending=[False, True]).iloc[0]
    return latest_card['id']


def question_2(client_df):
    """
    Q2: - The `client_id` that will retire within a year that has the lowest credit score and highest debt.
    Assume that we calculate retirement based on current_age and retirement_age.
    """
    client_df['total_debt'] = client_df['total_debt'].replace(
        '[\$,]', '', regex=True).astype(float)
    # Los años que le quedan para jubilarse se obtinen restando la edad de jubilación menos la edad actual
    client_df['years_to_retirement'] = client_df['retirement_age'] - client_df['current_age']
    retiring_soon = client_df[client_df['years_to_retirement'] <= 1]

    if not retiring_soon.empty:
        result = retiring_soon.sort_values(by=['credit_score', 'total_debt'], ascending=[True, False]).iloc[0]
        return result['id']
    return None


def question_3(transactions_df):
    """
    Q3: - The `transaction_id` of an Online purchase on a 31st of December with the highest absolute amount (either earnings or expenses).
    """
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    dec_31_transactions = transactions_df[transactions_df['date'].dt.month == 12]
    dec_31_transactions = dec_31_transactions[dec_31_transactions['date'].dt.day == 31]
    online_transactions = dec_31_transactions[dec_31_transactions['use_chip'].str.contains(
        "Online", na=False)]

    # Dar a amount el formato numérico adecuado
    online_transactions['amount'] = pd.to_numeric(online_transactions['amount'].replace('[\$,]', '', regex=True))

    highest_transaction = online_transactions.loc[online_transactions['amount'].abs().idxmax()]

    return highest_transaction['id']

def question_4(client_df, cards_df, transactions_df):
    """
    Q4: - Which client over the age of 40 made the most transactions with a Visa card in February 2016?
    Please return the `client_id`, the `card_id` involved, and the total number of transactions.
    """
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    client_df.rename(columns={'id': 'client_id'}, inplace=True)
    client_df = client_df[client_df['current_age'] > 40]

    cards_df = cards_df[cards_df['card_brand'] == 'Visa']
    merged_df = pd.merge(client_df, cards_df, on='client_id', how='inner', validate='1:m')

    full_df = pd.merge(merged_df, transactions_df, on='client_id', how='inner', validate='m:m')
    filtered_df = full_df[(full_df['date'].dt.year == 2016) & (full_df['date'].dt.month == 2)]
    result = filtered_df.groupby(['client_id', 'card_id']).size().reset_index(name='total_transactions')
    max_transactions = result[result['total_transactions'] == result['total_transactions'].max()]

    return max_transactions[['client_id', 'card_id', 'total_transactions']]


if __name__ == "__main__":
    # '../../data/raw/transactions_data.csv
    transactions_df = pd.read_csv('data/raw/transactions_data.csv')
    client_df = pd.read_csv('data/raw/users_data.csv')
    cards_df = pd.read_csv('data/raw/cards_data.csv')

# =====================Uso de API si no hubieran fallado=======================
#     client_ids = transaction_data['client_id'].unique()
#     client_data = []
#     for client in client_ids:
#         data = get_client_data(client)
#         if data:  # Asegurarse de que data no es None
#             client_data.append(data['values'])
#
#     card_data = []
#     for client in client_ids:
#         data = get_card_data(client)
#         if data:  # Asegurarse de que data no es None
#             card_data.append(data['values'])
# =============================================================================

    # Testing question 1
    result_question_1 = question_1(cards_df)

    # Testing question 2
    result_question_2 = question_2(client_df)

    # Testing question 3
    result_question_3 = question_3(transactions_df)

    # To free memory
    transactions_df = transactions_df[['date', 'card_id', 'client_id']]
    client_df = client_df[['id', 'current_age']]
    cards_df = cards_df[['id', 'client_id', 'card_brand']]
    
    # Testing question 4
    result_question_4 = question_4(client_df, cards_df, transactions_df)
