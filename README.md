# Objective
Track innovation trends and technological breakthroughs through patent embedding analysis. Patent embeddings are also used to measure the value of patents.

# Methodology 
I use sentence transformers (HuggingFace model "all-mpnet-base-v2") to vectorize the title and abstract of granted patents to US public companies.
The result is a vector database of hundreds of thousands of patents in a 768-dimensional embedding space. 'vectorize_patent_abstract.py' step by step goes through vectorizing each patent's abstract in the sample. 

Any direction in embedding space may correspond to a specific domain of innovation (although not all dimensions may be interpretable). 

# Innovation Trends 
The main idea is that variations in the direction of centroids (vectors representing the average of vector embeddings over patent clusters) in the embedding space point to the newly emerging areas of innovation. 

I study the variations in the main centroid, which is the average vector embeddings of all patents filed during a period. A more in-depth analysis can first cluster patents and study centroids associated with each cluster separately over time. 
Each month, to identify innovation trends I calculate the main centroid for the past 60 and 12 months and calculate their difference.

Below is a simplified illustration of embedding space in three dimensions. The vectors in gray are each patent's embedding, and the centroids over the past 12 and 60 months are shown in dark vectors as averages of patent embeddings over the past 12 and 60 months. The vector $\Delta_{12-60}$ represents the direction in embedding space where innovation concentration is growing.   

<img src="https://github.com/user-attachments/assets/da2b3832-c26f-4b20-b709-3efd9a4be357" alt="embed_space" width="400"/>

# Breakdown of Innovation Trends (using RAG)
Innovation trends are vectors that point in a direction in our embedding space. This vector may refer to a set of technological domains that can be revealed using the most aligned patents with it. For example, we can describe the innovation trend in month 't' by finding the top 5% of patents regarding cosine similarity and prompt LLMs to read, identify, and interpret the technological domain that is receiving the most attention according to innovation trend in month 't'. 'innovation_trend.py' implements this idea by first calculating innovation trends for every month, finding most similar patents to it, and using RAG to prompt gpt-4o to produce innovation trend reports based on those patents. 


# Patent Value
