# Objective
Track technological trends and breakthroughs through patent embedding analysis. Patent embeddings are also used to measure the value of patents.

# Methodology 
I use sentence transformers (HuggingFace model "all-mpnet-base-v2") to vectorize the title and abstract of granted patents to US public companies.
The result is a vector database of hundreds of thousands of patents in a 768-dimensional embedding space. 
Each direction in embedding space corresponds to a specific domain of technology/innovation (although not all dimensions are interpretable for humans).
The main idea here is that variations in the direction of centroids (vectors representing the average of vector embeddings over a cluster of patents) in the embedding space point to the newly emerging innovation hotspots. 


![embed_space](https://github.com/user-attachments/assets/da2b3832-c26f-4b20-b709-3efd9a4be357)
