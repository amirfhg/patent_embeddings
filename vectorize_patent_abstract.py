
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

# Load the embedding model (I use sentence transformers) and semantic chunker
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
text_splitter = SemanticChunker(embeddings)

# all_g_patents is the .csv file containing all the granted patents' abstracts owned by US public companies from 1970-2024
# all_g_patents is available here 'https://drive.google.com/file/d/1rASRTCKwMOoT6ec0HgiBra8M7wphnkA1/view?usp=sharing' (Original data obtained from PatentsView)
df = pd.read_csv('./all_g_patents.csv')

years = df['year'].unique()
for y in years:
  period_str = str(y)
  subset_df = df[df['year'] == y]
  abstract_list = []
  patent_id_list = []
  permno_list = []
  for index, row in subset_df.iterrows():
    abstract_list.append(row['patent_abstract'])
    patent_id_list.append(row['patent_id'])
    permno_list.append(row['PERMNO'])

  # Initialize lists
  embeddings_list = []
  ids_list = []

  # Iterate through each patent text along with its corresponding id number
  for i, (patent_text, id_number, permno_number) in enumerate(zip(abstract_list, patent_id_list, permno_list)):
    # Create a Document object for the abstract of each patent
    doc_to_split = Document(page_content=patent_text)

    # Split the document into smaller parts if necessary based on semantic chunking method
    docs = text_splitter.split_documents([doc_to_split])

    # Extract content from split documents
    docs_to_encode = [doc.page_content for doc in docs]

    # Initialize an empty list to store embeddings for each patent's abstract
    embeddings_list_ = []

    # Encode documents to get the embeddings
    for doc_text in docs_to_encode:
      # Encode the document to get the embedding
      embedding = model.encode([doc_text])
      # Append the embedding to the embeddings list
      embeddings_list_.append(embedding)

    # Generate IDs for each embedding
    ids = [str(permno_number) + "_" + str(id_number) + "_" + str(i) for i in range(len(docs))]

    # Append embeddings and ids to the respective lists
    embeddings_list.extend(embeddings_list_)
    ids_list.extend(ids)
    print(f"Iteration number: {i + 1}")

    # Flatten the embeddings list
    vector_embeddings_list = [arr.flatten().tolist() for arr in embeddings_list]

    # Create a dataframe to store 768 dimensional vector embeddings for all patents from year y
    # save the dataframe as a .zip file
    temp = pd.DataFrame({
      'patent_id': ids_list,
      **{f'column_{i+1}': [vec[i] for vec in vector_embeddings_list] for i in range(768)}
      })
    zip_path = f'zip_vectors/{period_str}.zip'
    # Create an in-memory bytes buffer
    buffer = BytesIO()
    # Save the DataFrame as a CSV to the buffer with period_str as the file name
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_buffer:
      with zip_buffer.open(f'{period_str}.csv', 'w') as csv_file:
        temp.to_csv(csv_file, index=False)

    # Write the buffer to the zip file
    with open(zip_path, 'wb') as f:
      f.write(buffer.getvalue())
