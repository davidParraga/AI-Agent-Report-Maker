import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def earnings_and_expenses(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a pandas DataFrame with the Earnings and Expenses total amount for the period range and user given.The expected columns are:
        - Earnings
        - Expenses
    The DataFrame should have the columns in this order ['Earnings','Expenses']. Round the amounts to 2 decimals.

    Create a Bar Plot with the Earnings and Expenses absolute values and save it as "reports/figures/earnings_and_expenses.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the earnings and expenses rounded to 2 decimals.

    """
    df['date'] = pd.to_datetime(df['date'])
    df_client = df[df['client_id'] == client_id]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask = (df_client['date'] >= start_date) & (df_client['date'] <= end_date)
    df_client = df_client.loc[mask]

    df_client['amount'] = df_client['amount'].replace({'\$': '', ',': ''}, regex=True)
    df_client['amount'] = df_client['amount'].astype(float)

    earnings = df_client[df_client['amount'] > 0]['amount'].sum()
    expenses = df_client[df_client['amount'] < 0]['amount'].sum()
    earnings = round(earnings, 2)
    expenses = round(expenses, 2)

    result_df = pd.DataFrame({'Earnings': [earnings], 'Expenses': [
                             expenses]}, columns=['Earnings', 'Expenses'])

    abs_earnings = abs(earnings)
    abs_expenses = abs(expenses)
    plt.figure()
    plt.bar(['Earnings', 'Expenses'], [abs_earnings,abs_expenses], color=['green', 'red'])
    plt.ylabel('Amount')
    plt.title('Earnings and Expenses')

    # Con os comprobamos si existe el directorio
    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'earnings_and_expenses.png'))
    plt.close()

    return result_df


def expenses_summary(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a Pandas Data Frame with the Expenses by merchant category. The expected columns are:
        - Expenses Type --> (merchant category names)
        - Total Amount
        - Average
        - Max
        - Min
        - Num. Transactions
    The DataFrame should be sorted alphabeticaly by Expenses Type and values have to be rounded to 2 decimals. Return the dataframe with the columns in the given order.
    The merchant category names can be found in data/raw/mcc_codes.json .

    Create a Bar Plot with the data in absolute values and save it as "reports/figures/expenses_summary.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the Expenses by merchant category.

    """
    df['date'] = pd.to_datetime(df['date'])
    df_client = df[df['client_id'] == client_id]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df_client['date'] >= start_date) & (df_client['date'] <= end_date)
    df_client = df_client.loc[mask]

    df_client['amount'] = df_client['amount'].replace( {'\$': '', ',': ''}, regex=True)
    df_client['amount'] = df_client['amount'].astype(float)

    # Filtrar por gastos
    df_expenses = df_client[df_client['amount'] < 0]

    # Obtener nombres de categoría de mercado
    mcc_codes_path = os.path.join('data', 'raw', 'mcc_codes.json')
    with open(mcc_codes_path, 'r') as f:
        mcc_codes_dict = json.load(f)

    df_expenses['mcc'] = df_expenses['mcc'].astype(str)

    # Mapear códigos 'mcc' codes con el nombre de categoría de mercado
    df_expenses['Expenses Type'] = df_expenses['mcc'].map(mcc_codes_dict)
    df_expenses['Expenses Type'] = df_expenses['Expenses Type'].fillna('Unknown')
    df_expenses['amount'] = df_expenses['amount'].abs()
    grouped = df_expenses.groupby('Expenses Type')

    summary_df = grouped['amount'].agg(
        Total_Amount=lambda x: x.sum(),
        Average=lambda x: x.mean(),
        Max=lambda x: x.max(),
        Min=lambda x: x.min(),
        Num_Transactions=lambda x: x.count()
    )

    summary_df = summary_df.abs()
    summary_df = summary_df.round(2)
    summary_df = summary_df.reset_index()
    summary_df = summary_df.sort_values('Expenses Type')
    summary_df = summary_df[['Expenses Type', 'Total_Amount', 'Average', 'Max', 'Min', 'Num_Transactions']]

    summary_df = summary_df.rename(columns={
        'Total_Amount': 'Total Amount',
        'Num_Transactions': 'Num. Transactions'
    })

    plot_df = summary_df.copy()
    plot_df['Total Amount'] = plot_df['Total Amount'].abs()

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df['Expenses Type'], plot_df['Total Amount'])
    plt.xticks(rotation=90)
    plt.xlabel('Expenses Type')
    plt.ylabel('Total Amount')
    plt.title('Expenses Summary by Merchant Category')

    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expenses_summary.png'))
    plt.close()

    return summary_df


def cash_flow_summary(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined by start_date and end_date (both inclusive), retrieve the available client data and return a Pandas DataFrame containing cash flow information.

    If the period exceeds 60 days, group the data by month, using the end of each month for the date. If the period is 60 days or shorter, group the data by week.

        The expected columns are:
            - Date --> the date for the period. YYYY-MM if period larger than 60 days, YYYY-MM-DD otherwise.
            - Inflows --> the sum of the earnings (positive amounts)
            - Outflows --> the sum of the expenses (absolute values of the negative amounts)
            - Net Cash Flow --> Inflows - Outflows
            - % Savings --> Percentage of Net Cash Flow / Inflows

        The DataFrame should be sorted by ascending date and values rounded to 2 decimals. The columns should be in the given order.

        Parameters
        ----------
        df : pandas DataFrame
           DataFrame  of the data to be used for the agent.
        client_id : int
            Id of the client.
        start_date : str
            Start date for the date period. In the format "YYYY-MM-DD".
        end_date : str
            End date for the date period. In the format "YYYY-MM-DD".


        Returns
        -------
        Pandas Dataframe with the cash flow summary.

    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)
    df = df[df['client_id'] == client_id]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    period_length = (end_date - start_date).days

    if period_length > 60:
        df['period'] = df['date'].dt.to_period('M')
        grouping = df.groupby('period')
        date_format = 'monthly'
    else:
        df['period'] = df['date'].dt.to_period('W-SUN')
        grouping = df.groupby('period')
        date_format = 'weekly'

    def aggregate(group):
        # Inflows: sum of positive amounts
        inflows = group.loc[group['amount'] > 0, 'amount'].sum()
        # Outflows: sum of absolute values of negative amounts
        outflows = group.loc[group['amount'] < 0, 'amount'].abs().sum()
        # Net Cash Flow: Inflows - Outflows
        net_cash_flow = inflows - outflows
        # % Savings: Net Cash Flow / Inflows * 100
        if inflows != 0:
            percent_savings = (net_cash_flow / inflows) * 100
        else:
            percent_savings = np.nan

        return pd.Series({
            'Inflows': inflows,
            'Outflows': outflows,
            'Net Cash Flow': net_cash_flow,
            '% Savings': percent_savings
        })

    agg_df = grouping.apply(aggregate).reset_index()

    if date_format == 'monthly':
        agg_df['Date'] = agg_df['period'].astype(str)
    else:
        agg_df['Date'] = agg_df['period'].apply(lambda x: x.end_time.strftime('%Y-%m-%d'))

    agg_df[['Inflows', 'Outflows', 'Net Cash Flow', '% Savings']] = agg_df[['Inflows', 'Outflows', 'Net Cash Flow', '% Savings']].round(2)
    agg_df = agg_df.sort_values('Date')
    agg_df = agg_df[['Date', 'Inflows', 'Outflows', 'Net Cash Flow', '% Savings']]
    agg_df = agg_df.reset_index(drop=True)

    return agg_df
