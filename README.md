# Objective
Track technological trends and breakthroughs through patent embedding analysis. Patent embeddings are also used to measure the value of patents.

# Methodology 
I use sentence transformers (HuggingFace model "all-mpnet-base-v2") to vectorize the title and abstract of granted patents to US public companies.
The result is a vector database of hundreds of thousands of patents in a 768-dimensional embedding space. 
Any direction in embedding space may correspond to a specific domain of innovation (although not all dimensions may be interpretable). 

# Principal Component Analysis (PCA)


# Technoloical Trends and Patent Value
The main idea is that variations in the direction of centroids (vectors representing the average of vector embeddings over patent clusters) in the embedding space point to the newly emerging innovation hotspots. 

<img src="https://github.com/user-attachments/assets/da2b3832-c26f-4b20-b709-3efd9a4be357" alt="embed_space" width="400"/>

