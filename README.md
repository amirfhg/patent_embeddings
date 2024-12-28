# Objective
Track innovation trends and technological breakthroughs through patent embedding analysis. Patent embeddings are also used to measure the value of patents.

# Embedding Technique 
I use sentence transformers (HuggingFace model "all-mpnet-base-v2") to vectorize the title and abstract of granted patents to US public companies.
The result is a vector database of hundreds of thousands of patents in a 768-dimensional embedding space. 

'vectorize_patent_abstract.py' step by step goes through vectorizing each patent's abstract in the sample. 

Directions in embedding space can correspond to a specific domain of innovation (although not all dimensions may be interpretable). 

# Innovation Trends 
The main idea is that the variations in the direction of centroids -- vectors representing the average of patent vector embeddings -- over time can reveal the newly emerging areas of innovation. $$centroid_t$$ is the vector that is the average of vector embeddings of patents filed in the last 't' months. I study the variations in this centroid over different periods to identify the changes in the theme of patents filed over time. A more in-depth analysis can first cluster patents (e.g. by industry, sector) and study centroids associated with each cluster separately over time. To identify monthly innovation trends, I calculate the difference between $$\vec{centroid_12}$$ and $$centroid_60$$, representing the average embeddings in the past 12 and 60 months. These values are arbitrary but the idea is that $$centroid_12$$ represents a more recent set of patents in the innovation space compared to $$centroid_60$$ and the difference between the two reveals the dimensions/direction that receive more attention in the embedding space. I produce the results with other periods besides 12 and 60 months to ensure the validity of my methodology. 

Below is a simplified illustration of embedding space in three dimensions. The vectors in gray are each patent's embedding, and the centroids over the past 12 and 60 months are shown in dark vectors as averages of patent embeddings over the past 12 and 60 months. The vector $\Delta_{12-60}$ represents the direction in embedding space where innovation concentration is growing.   

<img src="https://github.com/user-attachments/assets/da2b3832-c26f-4b20-b709-3efd9a4be357" alt="embed_space" width="400"/>

# Breakdown of Innovation Trends (using Retrieval-Augmented Generation (RAG))
Innovation trends are vectors that point in a direction in our embedding space. This vector may refer to a set of technological domains that can be revealed using the most aligned patents with it. For example, we can describe the innovation trend in month 't' by finding the top 5% of patents by cosine similarity $$\[
\cos(\vec{\text{patent}}, \vec{Delta_{12-60}})
\]$$. We then prompt LLMs to read, identify, and retrieve the name and description of technological domains mentioned in most similar patents to $\Delta_{12-60}$. This is essentially a RAG method that rely on temporal trends in vector embedding space to retrieve pieces of context (patent) with most relevance to emerging trends.  

'innovation_trend.py' implements this idea by first calculating innovation trends for every month, finding most similar patents to it, and using RAG prompt gpt-4o to produce innovation trend reports based on those patents. Below is the example of brief reports for the month of January 2023:

""

Alternatively, one can rely on LLM's ability to read patents and identify innovation trends as a complement to the methodology discussed above. This way the model's response benefits from focusing on the content of a carefully selected set of patents -- among hundreds of thousands of patents -- based on our methodology. The aim here is to improve LLM's ability to identify emerging innovation hotspots by filtering our 95% of less aligned (relevant) patents based on $$\[
\cos(\vec{\text{patent}}, \vec{Delta_{12-60}})
\]$$. This reduces the noise and improves the relevance and interpretability in LLM's response.

To implement this, I first collect the innovation reports for multiple months during a period (e.g. Jan-Dec 2023) produced by gpt-4o. Then I prompt the model to read the reports and describe possible innovation themes emerging over that period. Below is an example of such report for the year 2023: 

""

# Innovation Trend & Patent Value
