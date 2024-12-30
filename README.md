# Objective
I track innovation trends and technological breakthroughs through patent embedding analysis. 
I also establish that there is a link between innovation trends and patent value. 

# Embedding Technique 
I use sentence transformers (HuggingFace model "all-mpnet-base-v2") to vectorize the title and abstract of US public companies granted patents. 
The result is a vector database of hundreds of thousands of patents, each being a 768-dimensional vector in embedding space. 

The code in 'vectorize_patent_abstract.py' shows how I vectorize patent  abstracts in the sample. 

Directions in embedding space can correspond to a specific domain of innovation (although not all dimensions may be interpretable). 

# Innovation Trends 
The main idea is that the variations in the direction of centroids -- vectors representing the average of patent vector embeddings -- over time can reveal the newly emerging areas of innovation. $$centroid_t$$ is the vector that is the average of vector embeddings of patents filed in the last 't' months. I study the variations in this centroid over different periods to identify the changes in the theme of patents filed over time. 
A more in-depth analysis can first cluster patents (e.g. by industry, sector) and study centroids associated with each cluster separately over time.

To identify monthly innovation trends, I calculate the difference between $$\vec{centroid_{12}}$$ and $$\vec{centroid_{60}}$$, representing the average embeddings in the past 12 and 60 months. These values are arbitrary but the idea is that $$\vec{centroid_{12}}$$ represents a more recent set of patents in the innovation space compared to $$\vec{centroid_{60}}$$ and the difference between the two reveals the dimensions/direction that receive more attention in the embedding space. I produce the results with other periods besides 12 and 60 months to ensure the validity of my methodology. 

Below is a simplified illustration of embedding space in three dimensions. The vectors in gray are each patent's embedding $$\vec{patent}$$, and $$\vec{centroid_{12}}$$ and $$\vec{centroid_{60}}$$ are shown in dark vectors as averages of patent embeddings over the past 12 and 60 months. The vector $$\vec{\Delta_{12-60}}$$ represents the direction in embedding space where innovation concentration is growing.   

<p align="center">
  <img src="https://github.com/user-attachments/assets/acafd219-1b74-4f38-aa78-e0085fb466c2" alt="embed_space" width="50%" />
</p>


# Breakdown of Innovation Trends (using Retrieval-Augmented Generation (RAG))
So far I have constructed innovation trends which are vectors in the embedding space, but how how to interpret these vectors? are there technologies associated with innovation trends? To investigate this, I identify and summarize the patents most similar to innovation trends in the embedding space. This demonstrates the set of technological domains discussed in the patents associated with innovation trends. For example, I can describe the innovation trend in month 't' by finding the top 5% of patents by Cosine similarity $$\[
Cosine(\vec{\text{patent}}, \vec{\Delta_{12-60}})
\]$$. To automate reading the patents, I prompt LLMs to to do so. LLMs retrieve the name and description of technological domains mentioned in most similar patents to $$\vec{\Delta_{12-60}}$$. This is essentially a RAG method that rely on temporal trends in vector embedding space to retrieve pieces of context (patent) with most relevance to emerging trends.  

The code in 'patent_report_rag.py' implements this idea by first calculating innovation trends for every month, finding most similar patents to it, and using RAG prompt gpt-4o to produce innovation trend reports based on those patents. 
Below is the example of brief reports for January 2023 generated using RAG:

# Summary of Innovations for January 2023

> **Summary of innovations for year-month 2023-01 are:**
>
> 1. **Data Management and Storage**: Innovations focus on efficient data parity, memory management, and data migration across different memory tiers. These technologies improve data integrity, storage efficiency, and performance in memory systems, which are crucial for data centers and cloud storage solutions.
>
> 2. **Display Technology**: Advances in display devices involve optimizing pixel arrangements and driving mechanisms to enhance image quality and power efficiency. This is applicable in consumer electronics like TVs, monitors, and smartphones.
>
> 3. **Memory Systems**: Techniques for memory operation optimization, including maintenance command interfaces and memory access parameter tuning, aim to enhance memory performance and reliability. These are vital for improving the speed and efficiency of computing devices.
>
> 4. **Database and Data Processing**: Innovations include near-memory database accelerators and dynamic data processing in multi-cluster warehouses. These technologies aim to handle large data volumes more efficiently, benefiting industries relying on big data analytics and real-time data processing.
>
> 5. **Image Systems**: Dynamic updating of drive sequences based on sensor input enhances display performance and adaptability, applicable in devices requiring high-quality visual output, such as augmented reality systems.
>
> 6. **Data Synchronization and Replication**: Persistent inflight tracking and data synchronization across storage clusters ensure data consistency and reliability, important for disaster recovery and data backup solutions.
>
> 7. **Legal Compliance in Data Transfer**: Methods for handling data with legal restrictions ensure compliance during data transfer processes, crucial for industries dealing with sensitive information, such as finance and healthcare.
>
> 8. **Video Content Generation**: Techniques for generating video content based on user input can be used in media production and interactive entertainment, allowing for personalized content creation.
>
> 9. **Trace Data Management**: Efficient handling and storage of trace data improve system monitoring and debugging processes, essential for maintaining the performance and reliability of complex software systems.




Alternatively, one can rely on LLM's ability to read patents and identify innovation trends as a complement to the methodology discussed above. This way the model's response benefits from focusing on the content of a carefully selected set of patents -- among hundreds of thousands of patents -- based on my methodology. The aim here is to improve LLM's ability to identify emerging innovation hotspots by filtering out 95% of less aligned (relevant) patents based on $$\[
Cosine(\vec{\text{patent}}, \vec{\Delta_{12-60}})
\]$$. This reduces the noise and improves the relevance and interpretability in LLM's response.

To implement this, I first collect the innovation reports for multiple months during a period (e.g. Jan-Dec 2023) produced by gpt-4o. Then I prompt the model to read the reports and describe possible innovation themes emerging over that period. Below is an example of such report for the year 2023: 

""

# Innovation Trend & Patent Value
The study *Technological Innovation, Resource Allocation, and Growth* by Kogan, Papanikolaou, Seru, and Stoffman (2017) proposes a measure of patent economic value that relies on the initial market response to the news of the patent grant. They argue that investors are forward-looking and their response to the news of the patent grant provides an estimation of the private value of the patent. The market response is calculated by the movements in stock prices following the news of patent grant. Specifically, the movement is measured as a stock's turnover: 

$$\[
\text{Share Turnover} = \frac{\text{Volume of Shares Traded}}{\text{Shares Outstanding}}
\]$$

A greater stock turnover ratio during the days following a patent grant indicates that the news holds important information for the market.
I hypothesize that the degree of the patent's relevance to recent innovation trends is a predictor of its value as it exposes the issuing firm to the economic value generated by that innovation trend. Following this, I test whether patent embeddings can predict market response to its grant news which is a proxy for its value based on Kogan, Papanikolaou, Seru, and Stoffman (2017). 


<p align="center">
  <img src="https://github.com/user-attachments/assets/5c8ceb6d-74ed-489b-a61b-0206a857c8a8" alt="boxplot_turnover" width="50%" />
</p>

### References

Kogan, Leonid, Papanikolaou, Dimitris, Seru, Amit, and Stoffman, Noah. (2017). 
"Technological Innovation, Resource Allocation, and Growth." 
*The Quarterly Journal of Economics*, 132(2), 665–712. 
Oxford Academic.
