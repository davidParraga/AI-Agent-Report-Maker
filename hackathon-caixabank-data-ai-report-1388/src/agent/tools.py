# Use this file to implement any tools or helpers you need.

prompt_text = """
You are an assistant that extracts dates from text and outputs them in a structured format.

Given a text, extract the start date and end date, and output them in the format:

start_date: YYYY-MM-DD
end_date: YYYY-MM-DD

If only one date or a specific month is provided, infer the start and end dates accordingly.

Examples:

Text: "Create a pdf report from 2018-01-01 to 2018-05-31"

Output:
start_date: 2018-01-01
end_date: 2018-05-31

Text: "Create a pdf report for the fourth month of 2017"

Output:
start_date: 2017-04-01
end_date: 2017-04-30

Now, process the following text:

Text: "{input_text}"

Output:
"""
