import os
import json
import pandas as pd
from langchain import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from fpdf import FPDF
from src.agent.tools import prompt_text
import re

from src.data.data_functions import (
    earnings_and_expenses,
    expenses_summary,
    cash_flow_summary,
)


def run_agent(df: pd.DataFrame, client_id: int, input: str) -> dict:
    """
    Create a simple AI Agent that generates PDF reports using the three functions from Task 2 (src/data/data_functions.py).
    The agent should generate a PDF report only if valid data is available for the specified client_id and date range.
    Using the data and visualizations from Task 2, the report should be informative and detailed.

    The agent should return a dictionary containing the start and end dates, the client_id, and whether the report was successfully created.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame of the data to be used for the agent.
    client_id : int
        Id of the client making the request.
    input_text : str
        String with the client input for creating the report.

    Returns
    -------
    variables_dict : dict
        Dictionary of the variables of the query.
            {
                "start_date": "YYYY-MM-DD",
                "end_date" : "YYYY-MM-DD",
                "client_id": int,
                "create_report" : bool
            }
    """
    # Initialize LLM
    llm = Ollama(model='llama3.2:1b')
    
    prompt = PromptTemplate(
        input_variables=['input'],
        template=prompt_text
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain to get the dates
    response = chain.run(input_text=input)

    # Parse the response to get start_date and end_date
    date_pattern = r'start_date:\s*([\d-]+)\s*end_date:\s*([\d-]+)'
    match = re.search(date_pattern, response)

    if match:
        start_date = match.group(1)
        end_date = match.group(2)
    else:
        # Handle the case where LLM output is not in expected format
        lines = response.strip().splitlines()
        start_date = None
        end_date = None
        for line in lines:
            if 'start_date:' in line:
                start_date = line.split('start_date:')[-1].strip()
            if 'end_date:' in line:
                end_date = line.split('end_date:')[-1].strip()
        if not start_date or not end_date:
            variables_dict = {
                'start_date': None,
                'end_date': None,
                'client_id': client_id,
                'create_report': False
            }
            return variables_dict

    # Now check if data is available for the client_id and date range
    df_client = df[df['client_id'] == client_id]
    if df_client.empty:
        variables_dict = {
            'start_date': start_date,
            'end_date': end_date,
            'client_id': client_id,
            'create_report': False
        }
        return variables_dict

    # Convert dates to datetime
    try:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
    except Exception as e:
        variables_dict = {
            'start_date': start_date,
            'end_date': end_date,
            'client_id': client_id,
            'create_report': False
        }
        return variables_dict

    # Filter data for date range
    df_client['date'] = pd.to_datetime(df_client['date'])
    df_filtered = df_client[(df_client['date'] >= start_date_dt) & (df_client['date'] <= end_date_dt)]
    if df_filtered.empty:
        variables_dict = {
            'start_date': start_date,
            'end_date': end_date,
            'client_id': client_id,
            'create_report': False
        }
        return variables_dict

    # Now generate the report using functions from data_functions.py
    earnings_expenses_df = earnings_and_expenses(df, client_id, start_date, end_date)
    expenses_summary_df = expenses_summary(df, client_id, start_date, end_date)
    cash_flow_df = cash_flow_summary(df, client_id, start_date, end_date)

    # Create the PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Client ID: {client_id}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Report for period: {start_date} to {end_date}", ln=True, align='L')

    # Earnings and Expenses
    pdf.cell(200, 10, txt="Earnings and Expenses:", ln=True, align='L')
    earnings_expenses_str = earnings_expenses_df.to_string(index=False)
    pdf.multi_cell(0, 10, earnings_expenses_str)
    pdf.image('reports/figures/earnings_and_expenses.png', x=None, y=None, w=100)

    # Expenses Summary
    pdf.cell(200, 10, txt="Expenses Summary:", ln=True, align='L')
    expenses_summary_str = expenses_summary_df.to_string(index=False)
    pdf.multi_cell(0, 10, expenses_summary_str)
    pdf.image('reports/figures/expenses_summary.png', x=None, y=None, w=100)

    # Cash Flow Summary
    pdf.cell(200, 10, txt="Cash Flow Summary:", ln=True, align='L')
    cash_flow_str = cash_flow_df.to_string(index=False)
    pdf.multi_cell(0, 10, cash_flow_str)

    # Save the PDF
    if not os.path.exists('reports'):
        os.makedirs('reports')
    report_filename = f'reports/report_client_{client_id}_{start_date}_{end_date}.pdf'
    pdf.output(report_filename)

    variables_dict = {
        'start_date': start_date,
        'end_date': end_date,
        'client_id': client_id,
        'create_report': True
    }
    
    return variables_dict
