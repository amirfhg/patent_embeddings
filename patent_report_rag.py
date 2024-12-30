from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import re


import os
import pandas as pd

# all_g_patents is the .csv file containing all the granted patents' abstracts owned by US public companies from 1970-2024
# all_g_patents is available here 'https://drive.google.com/file/d/1rASRTCKwMOoT6ec0HgiBra8M7wphnkA1/view?usp=sharing' (Original data obtained from PatentsView)
df = pd.read_csv('./all_g_patents.csv')
df['patent_id'] = df['patent_id'].astype(int)

# delta_cosine is the patent_level cosine similarity with monthly innovation_trend vector that is calculated in 'innovation_trend.py' in this repository. 
delta_cosine = pd.read_csv('./cosine_similarity_12_60.csv')
delta_cosine['filing_date'] = pd.to_datetime(delta_cosine['filing_date'])

# Create the 'year_month' column in 'YYYY-MM' format
delta_cosine['year_month'] = delta_cosine['filing_date'].dt.to_period('M').astype(str)

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

req = "Describe the general areas of innovation related to the patents in the context provided"
prompt = PromptTemplate(input_variables=["request", "context"], template=template)

# Create the LangChain processing chain
chain = prompt | model | StrOutputParser()
results = chain.invoke({"request": req, "context": top_patents})

print(results)
