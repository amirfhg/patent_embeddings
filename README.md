# Objective
I track innovation trends and technological breakthroughs through patent embedding analysis. 
I also establish that there is a link between innovation trends and patent value. 

# Embedding Technique 
I use sentence transformers (HuggingFace model "all-mpnet-base-v2") to vectorize the title and abstract of US public companies granted patents. Granted patent data, including abstracts, are collected from PatentsView at https://patentsview.org/download/data-download-tables. The result is a vector database of hundreds of thousands of patents, each being a 768-dimensional vector in embedding space. 

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
\]$$. 

To automate reading the patents, I prompt LLMs to to do so. LLMs retrieve the name and description of technological domains mentioned in most similar patents to $$\vec{\Delta_{12-60}}$$. This is essentially a RAG method that rely on temporal trends in vector embedding space to retrieve pieces of context (patents) with most relevance to emerging trends. This way the model's response benefits from focusing on the content of a carefully selected set of patents -- among hundreds of thousands of patents -- based on my methodology. The aim here is to improve LLM's ability to identify emerging innovation hotspots by filtering out 95% of less aligned (relevant) patents based on $$\[
Cosine(\vec{\text{patent}}, \vec{\Delta_{12-60}})
\]$$. This reduces the noise and improves the relevance and interpretability in LLM's response. 

The code in 'patent_report_rag.py' implements this idea by first calculating innovation trends for every month, finding most similar patents to it, and using RAG prompt gpt-4o to produce innovation trend reports based on those patents. 
Below is the example of brief reports for January 2023 generated using RAG:

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

According to LLM's response above, we can see that technologies with applications in areas like data center solutions and augmented reality are most aligned with $$\vec{\Delta_{12-60}}$$ calculated for January 2023. 

The method above can also be used to find innovation themes that are emerging over a period of a year or more and the potential future trajectory of these innovations. To illustrate this for the year 2022, I first collect the innovation reports for multiple months during the period (e.g. Jan-Dec 2022) produced by gpt-4o using the RAG method discussed above. Then I prompt the model to read the reports generated in the first stage (stage above) and describe possible innovation themes emerging over that period. Here is an example of such a report for the year 2022: 

> Based on the information provided, several innovation themes are persistent over the period from January to December 2022. These recurring themes include:
>
> 1. **Memory Management and Optimization**: Throughout the year, there are consistent innovations aimed at improving memory efficiency, reliability, and performance. This includes techniques for error detection and correction, memory access optimization, and power management.
>
> 2. **Data Storage and Retrieval**: Innovations in data storage systems, including methods for data redundancy, efficient data retrieval, and storage optimization, are a constant focus. These are crucial for enhancing the performance and reliability of storage solutions.
>
> 3. **Error Detection and Correction**: Techniques for maintaining data integrity through error detection and correction are repeatedly highlighted. This is essential for ensuring data accuracy and system stability across various applications.
>
> 4. **Cloud Computing and Distributed Systems**: There is a continuous emphasis on improving cloud infrastructure, data management, and distributed storage systems. Innovations in this area aim to enhance data availability, disaster recovery, and system resilience.
>
> 5. **Database and Data Management**: Throughout the year, there are advancements in database management, including query optimization and data consistency checks, which are vital for efficient data handling.
>
> 6. **Display Technology**: Although not as frequent as other themes, display technology innovations appear multiple times, focusing on improving visual quality and energy efficiency in electronic devices.
>
> The trajectory of innovations in the future is likely to continue focusing on these themes, with an increasing emphasis on enhancing efficiency, reliability, and scalability. As technology demands grow, we can expect further advancements in memory and data management, particularly in areas like artificial intelligence, machine learning, and real-time data processing. Additionally, innovations in energy efficiency and environmental adaptability will likely become more prominent as sustainability becomes a critical concern in technology development.


# Innovation Trend & Patent Value
The study *Technological Innovation, Resource Allocation, and Growth* by Kogan, Papanikolaou, Seru, and Stoffman (2017) proposes a measure of patent economic value that relies on the initial market response to the news of the patent grant. They argue that investors are forward-looking and their response to the news of the patent grant provides an estimation of the private value of the patent. The market response is calculated by the movements in stock prices following the news of patent grant. Specifically, the movement is measured as a stock's turnover: 

$$\[
\text{Share Turnover} = \frac{\text{Volume of Shares Traded}}{\text{Shares Outstanding}}
\]$$

A greater stock turnover ratio during the days following a patent grant indicates that the news holds important information for the market.
I hypothesize that the degree of the patent's relevance to recent innovation trends is a predictor of its value as it exposes the issuing firm to the economic value generated by that innovation trend. Following this, I test whether patent embeddings can predict market response to its grant news which is a proxy for its value based on Kogan, Papanikolaou, Seru, and Stoffman (2017). 

The figure below is a box plot of share turnover for about 140, 000 observations of patent grant events for 3177 U.S. public companies for the period between 1976 and 2023. I present the share turnovers by deciles based on $$\[
Cosine(\vec{\text{patent}}, \vec{\Delta_{12-60}})
\]$$ in the sample to illustrate the relation between the patent's cosine similarity with $$\vec{\Delta_{12-60}}$$ and share turnover following the news of the patent grant. The first plot presents the same-day share turnover as the patent grant news is announced. Other plots present share turnovers in the following two days. As we can see the median share turnover is positively associated with $$\[
Cosine(\vec{\text{patent}}, \vec{\Delta_{12-60}})
\]$$. This suggests that the share turnover -- a proxy for patent value based on Kogan, Papanikolaou, Seru, and Stoffman (2017) -- is positively related to the patent's cosine similarity with $$\vec{\Delta_{12-60}}$$.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5c8ceb6d-74ed-489b-a61b-0206a857c8a8" alt="boxplot_turnover" width="50%" />
</p>


<div align="center">
  <img src="https://github.com/user-attachments/assets/8e345e67-377e-4b26-aa62-651cce296db0" alt="patent_table" width="50%" />
</div>


### References

Kogan, Leonid, Papanikolaou, Dimitris, Seru, Amit, and Stoffman, Noah. (2017). 
"Technological Innovation, Resource Allocation, and Growth." 
*The Quarterly Journal of Economics*, 132(2), 665–712. 
Oxford Academic.
