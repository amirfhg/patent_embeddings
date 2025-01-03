
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import re

# Commented out IPython magic to ensure Python compatibility.
import os
import pandas as pd



# all_g_patents is the .csv file containing all the granted patents' abstracts owned by US public companies from 1970-2024
# all_g_patents is available here 'https://drive.google.com/file/d/1rASRTCKwMOoT6ec0HgiBra8M7wphnkA1/view?usp=sharing' (Original data obtained from PatentsView)
df = pd.read_csv('./all_g_patents.csv')
df['patent_id'] = df['patent_id'].astype(int)

# delta_cosine is the patent-level cosine similarity with monthly innovation trend vector that is calculated in 'innovation_trend.py' file in this repository.  
delta_cosine = pd.read_csv('./cosine_similarity_12_60.csv')
delta_cosine['filing_date'] = pd.to_datetime(delta_cosine['filing_date'])
# Create the 'year_month' column in 'YYYY-MM' format
delta_cosine['year_month'] = delta_cosine['filing_date'].dt.to_period('M').astype(str)

"""Below we set a target month (Jan 2023) and find the most alogned patents in that month with the innovation trend calculated for that month"""

target_year_month = '2023-01'
# Filter for the specific year_month '2023-01'
delta_cosine_filtered = delta_cosine[delta_cosine['year_month'] == target_year_month]

# Calculate the threshold for the top 5% of 'cos_sim_delta'
threshold = delta_cosine_filtered['cos_sim_delta'].quantile(0.95)

# Collect the 'patent_id' of rows with 'cos_sim_delta' above the threshold
top_patent_id = delta_cosine_filtered[delta_cosine_filtered['cos_sim_delta'] >= threshold]['patent_id'].tolist()

# Collect the 'patent_abstract' for the matching 'patent_id' in top_patent_id
top_patents = df[df['patent_id'].isin(top_patent_id)]['patent_abstract'].tolist()

# Combine abstracts into a single string separated by "|"
top_patents_count = len(top_patents)
top_patents = '|'.join(top_patents)

"""Below we use RAG and the retrieved patents to describe innovation trends for Jan 2023"""

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    api_key="?",
    temperature=0
)

# Build the prompt template
template = f"""Use the following context to respond to the request at the end.
The context is a list of patent abstracts separated by '|'.
There are {top_patents_count} patent abstracts in the list.
Read the information in the context and use your general knowledge related to
that information to produce a concise and less technical answer to the question.
Do not give information not mentioned in the context.
If you don't know the answer, just say that you don't know, don't try to make up an answer
Your answer should include "Summary of innovations for year-month {target_year_month} are:" in the beginning of your response.
{{context}}
Request: {{request}}
"""

req = "Based on the patents in the context provided, describe the general areas of innovation/technology and their application in the real world."
prompt = PromptTemplate(input_variables=["request", "context"], template=template)

# Create the LangChain processing chain
chain = prompt | model | StrOutputParser()
results = chain.invoke({"request": req, "context": top_patents})

"""Below we first retrieve most aligned patents with innovation trends for each month of 2023. And second collect LLM's describtion of innvation trends for all months in a list called results_list"""

from datetime import datetime, timedelta

start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2022-12-31", "%Y-%m-%d")

# Generate all months correctly
target_period = []
current_date = start_date

while current_date <= end_date:
    target_period.append(current_date.strftime("%Y-%m"))
    # Increment by one month
    next_month = current_date.month % 12 + 1
    year_increment = current_date.month // 12
    current_date = current_date.replace(year=current_date.year + year_increment, month=next_month)

results_list = []

for target_year_month in target_period:
    # Filter for the specific year_month
    delta_cosine_filtered = delta_cosine[delta_cosine['year_month'] == target_year_month]

    # Calculate the threshold for the top 5% of 'cos_sim_delta'
    threshold = delta_cosine_filtered['cos_sim_delta'].quantile(0.95)

    # Collect the 'patent_id' of rows with 'cos_sim_delta' above the threshold
    top_patent_id = delta_cosine_filtered[delta_cosine_filtered['cos_sim_delta'] >= threshold]['patent_id'].tolist()

    # Collect the 'patent_abstract' for the matching 'patent_id' in top_patent_id
    top_patents = df[df['patent_id'].isin(top_patent_id)]['patent_abstract'].tolist()

    # Combine abstracts into a single string separated by "|"
    top_patents_count = len(top_patents)
    top_patents = '|'.join(top_patents)

    # Build the prompt template
    template = f"""Use the following context to respond to the request at the end.
    The context is a list of patent abstracts separated by '|'.
    There are {top_patents_count} patent abstracts in the list.
    Read the information in the context and use your general knowledge related to
    that information to produce a concise and less technical answer to the question.
    Do not give information not mentioned in the context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Your answer should include "Summary of innovations for year-month {target_year_month} are:" in the beginning of your response.
    {{context}}
    Request: {{request}}
    """

    req = "Based on the patents in the context provided, describe the general areas of innovation/technology and their application in the real world."
    prompt = PromptTemplate(input_variables=["request", "context"], template=template)

    # Create the LangChain processing chain
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"request": req, "context": top_patents})

    # Append the result to results_list
    results_list.append(result)

"""Below we ask LLM to read the items in results_list to identif potential innvation themes and possible future innovation trajectories"""

template = f"""Use the following context to respond to the request at the end.
    The context is a list of summaries of innvations for each multiple months.
    In your answer pay atention to the dates provided in the context.
    Read the information in the context and use your general knowledge related to
    that information to produce a concise and less technical answer to the question.
    Do not give information not mentioned in the context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {{context}}
    Request: {{request}}
    """

req = """Based on the information in the context provided, describe whether there are innovation themes that are persistent over the period.
What do you predict the trajectory of innovations be in the future based on the chronological order of information provided in the context?"""

prompt = PromptTemplate(input_variables=["request", "context"], template=template)

# Create the LangChain processing chain
chain = prompt | model | StrOutputParser()
result = chain.invoke({"request": req, "context": results_list})
