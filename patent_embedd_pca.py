import pandas as pd
import os
import zipfile
import numpy as np
import re
import string
from sklearn.decomposition import PCA
# NLTK for tokenization and n-grams
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter


# all_g_patents is the .csv file containing all the granted patents' abstracts owned by US public companies from 1970-2024
# all_g_patents is available here 'https://drive.google.com/file/d/1rASRTCKwMOoT6ec0HgiBra8M7wphnkA1/view?usp=sharing' (Original data obtained from PatentsView)
df = pd.read_csv('./all_g_patents.csv')
df['patent_id'] = df['patent_id'].astype(str)

# Function to clean and tokenize text, returning list of tokens
def clean_and_tokenize(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split
    tokens = text.split()
    # Remove stopwords (we keep numbers, but remove common words)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

final_results = pd.DataFrame()
years = list(range(1970, 2025))

for i in range(0, len(years), 5):
    five_years = years[i:i + 5]

    # Initialize an empty DataFrame to gather patent embedding data for the current year
    # Remember patent embedding data was created and saved as .zip files for each year in vectorize_patent_abstract.py 
    
    temp = pd.DataFrame()

    for year in five_years:
        temp_zip = os.path.join("zip_vectors/", f"{year}.zip")

        # Open and extract files if the zip file exists
        if os.path.exists(temp_zip):
            with zipfile.ZipFile(temp_zip, 'r') as zf:
                if f"{year}.csv" in zf.namelist():
                    with zf.open(f"{year}.csv") as file:
                        df_ = pd.read_csv(file)

                        # Clean up 'patent_id'
                        df_['patent_id'] = df_['patent_id'].str.extract(r'^\d+_(\d+)_\d+$')

                        # Concatenate into the temporary DataFrame
                        temp = pd.concat([temp, df_], ignore_index=True)

    # Once we've appended the data for this year to temp, perform PCA
    # if temp is not empty and has the right columns:
    vector_columns = [f"column_{col_i}" for col_i in range(1, 769)]
    if not temp.empty and all(col in temp.columns for col in vector_columns):
            pca = PCA(n_components=3)

            # Fit the PCA on the 768-dimensional vectors
            pca.fit(temp[vector_columns])

            # Create a DataFrame for the top 3 components
            pca_df = pd.DataFrame(
                pca.components_,          # shape (3, 768)
                columns=vector_columns    # each column is one of the 768 features
            )

            # Add additional metadata columns
            pca_df['pc_name'] = ['PC1', 'PC2', 'PC3']  # Specify which principal component
            pca_df['variance_explained'] = pca.explained_variance_ratio_[:3]  # Variance for each component
            pca_df['years'] = f"{five_years[0]}-{five_years[-1]}"
            pca_df['top_terms'] = ""

            # ----------------------------------------------------------------------
            # Collect abstracts for patents with highest cosine similarity (top 5%)
            # to each principal component (PC1, PC2, PC3). Then gather top unigrams & bigrams.
            # ----------------------------------------------------------------------

    for idx, pc_name in enumerate(['PC1', 'PC2', 'PC3']):
                pc_vector = pca.components_[idx]  # shape (768,)

                # Patent vectors (N, 768) matrix
                patent_matrix = temp[vector_columns].values  # shape (N, 768)

                # Norms
                pc_norm = np.linalg.norm(pc_vector)
                patent_norms = np.linalg.norm(patent_matrix, axis=1)

                # Dot products
                dot_products = np.dot(patent_matrix, pc_vector)

                # Cosine similarities
                cosine_sims = dot_products / (pc_norm * patent_norms)

                # Top 5%
                threshold = np.quantile(cosine_sims, 0.95)
                top_indices = np.where(cosine_sims >= threshold)[0]

                # Patents for top 5%
                top_patent_id = temp.iloc[top_indices]['patent_id'].unique().tolist()

                # Collect their abstracts from df
                top_abstracts_df = df[df['patent_id'].isin(top_patent_id)]

                # Combine all abstracts into one big string
                big_text = " ".join(str(a) for a in top_abstracts_df['patent_abstract'].dropna())

                # Clean & tokenize
                tokens = clean_and_tokenize(big_text)


                # Build bigrams
                bigrams_list = list(zip(tokens, tokens[1:]))
                bigrams_list = [" ".join(bg) for bg in bigrams_list]

                # Count them

                bigram_counter = Counter(bigrams_list)


                top_100_bigrams = bigram_counter.most_common(100)



                # Extract only the words
                words = [word for word, _ in top_100_bigrams]

                # Join them into a comma-separated string
                result = ",".join(words)
                pca_df.loc[pca_df['pc_name'] == pc_name, 'top_terms'] = result

    # Drop the high-dimensional feature columns from pca_df
    pca_df = pca_df.drop(columns=vector_columns)

    # Concatenate the results into final_results
    final_results = pd.concat([final_results, pca_df], ignore_index=True)

    # At the end of the five_years loop, delete temp
    del temp
    del pca_df
    print(f"Processed years: {five_years}")

"""Using OpenAI's API we ask gpt-4o to extract bigrams from column 'top_terms' in final_results that are related to piece of technology or innovation"""

!pip install langchain
!pip install langchain_openai

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import re

# Initialize the new column for technology terms
final_results['technology'] = ''

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    api_key="?",
    temperature=0
)

# Build the prompt template
template = """In the following list of terms find and return any words/terms that refer
to a particular technology or innovation.
Please return output words/terms in a list format in Python.
Do not put anything else in the list. Only technology and innovation related terms.
List of Terms: {list_of_terms}
"""
prompt = PromptTemplate(input_variables=["list_of_terms"], template=template)

# Create the LangChain processing chain
chain = prompt | model | StrOutputParser()

# Iterate over the rows of final_results
for idx, row in final_results[1:34].iterrows():
    # Extract the list of terms from the 'top_terms' column
    list_of_terms = row['top_terms']

    # Skip empty or invalid rows
    if not isinstance(list_of_terms, str) or list_of_terms.strip() == "":
        continue

    # Invoke the chain
    try:
        selected_terms = chain.invoke(list_of_terms)
        # Split the selected terms into a list
        selected_terms = selected_terms.split(',')
        # Combine all strings into one
        combined = "".join(selected_terms)
        # Extract technology terms inside double quotes
        tech_terms = re.findall(r'"(.*?)"', combined)
        # Update the 'technology' column
        final_results.at[idx, 'technology'] = tech_terms
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        final_results.at[idx, 'technology'] = None

final_results.to_csv('./technological_areas.csv', index=False)
