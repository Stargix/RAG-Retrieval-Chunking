# RAG Retrieval Pipeline: Optimization and Evaluation

## Project Overview
This project implements and evaluates a Retrieval Augmented Generation (RAG) pipeline, focusing on the critical chunking and embedding components that affect retrieval performance. The primary objective is to analyze how different chunking strategies and retrieval parameters impact the quality of information retrieval from a document corpus.

## Dataset
This implementation uses the **State of the Union** dataset, which contains:
- Presidential addresses (corpus)
- Relevant queries
- Golden excerpts (reference documents)

## Components

### Chunking Algorithm
- Implemented the `FixedTokenChunker` algorithm for splitting documents into smaller chunks.
- The algorithm works as follows:
  1. Text is tokenized into individual tokens (words/subwords)
  2. The document is divided into chunks of a specified fixed token count
  3. Each chunk overlaps with the next by a configurable number of tokens
  4. The algorithm maintains chunk boundaries that preserve textual coherence
  5. Original position information is tracked to enable precise evaluation metrics
- This token-based approach ensures consistent chunk sizes regardless of sentence structure or paragraph boundaries
- Explored different chunking parameters:
    - **Chunk sizes**: 10-50 tokens
    - **Overlap values**: 1-15 tokens
    - **Number of retrieved chunks**: 1-5

### Embedding Models
- **Primary model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
    - Selected for balanced computational efficiency and semantic representation quality.
    - Produces strong results on semantic similarity tasks with modest computational resources.
- **Alternative model tested**: `multi-qa-mpnet-base-dot-v1`

### Reranking Strategy
- Implemented a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) to address limitations in embedding-based retrieval:
    - Semantic gap issues
    - Chunk boundary problems
    - Context limitations
- The two-stage retrieval approach (embedding-based retrieval followed by reranking) significantly improved retrieval quality.

## Evaluation Metrics

### Precision, Recall, and IoU

We compute **IoU (Intersection over Union)** for a given query in the chunked corpus as:

$$
IoU_q(C) = \frac{|t_e \cap t_r|}{|t_e| + |t_r| - |t_e \cap t_r|}
$$

Here:
- \( t_e \): Tokens in the golden excerpt (relevant tokens).
- \( t_r \): Tokens in the retrieved chunks.

In the numerator, each \( t_e \) among \( t_r \) is counted only once, while the denominator includes all retrieved tokens in \( |t_r| \). This adjustment accounts for redundancy when overlapping chunking strategies are used, ensuring a fair evaluation.

Alongside IoU, we use **precision** and **recall** metrics at the token level, defined as follows:

- **Precision**:

$$
Precision_q(C) = \frac{|t_e \cap t_r|}{|t_r|}
$$

Measures the ratio of relevant tokens to all retrieved tokens.

- **Recall**:

$$
Recall_q(C) = \frac{|t_e \cap t_r|}{|t_e|}
$$

Measures the ratio of relevant tokens retrieved to all relevant tokens.

These metrics provide a comprehensive evaluation of retrieval performance, balancing relevance and coverage.

## Experiments and Results

### Hyperparameter Optimization
- Used Optuna for systematic exploration of chunking parameters.
- **Optimal configuration found**:
    - **Chunk Size**: 35 tokens
    - **Chunk Overlap**: 11 tokens
    - **Retrieved Chunks**: 1 per query
- **Performance**:
    - Precision: 0.53
    - Recall: 0.58
    - IoU: 0.4

### Key Findings

#### Chunk Size Impact
- **Small chunks (15-25 tokens)**: Higher precision (0.543) but lower recall (0.452).
- **Medium chunks (30-40 tokens)**: Best overall balance with 0.445 precision and 0.587 recall.
- **Large chunks (45-50 tokens)**: Similar to medium chunks but with more irrelevant information.

#### Overlap Level Impact
- **Minimal overlap (1-4 tokens)**: Efficient processing with higher precision (0.518).
- **Moderate overlap (5-9 tokens)**: Better balance between precision (0.472) and recall (0.553).
- **High overlap (10-15 tokens)**: Maintains cohesion but introduces redundancy.

#### Number of Retrieved Chunks
- **Single chunk**: Greater coherence with 0.487 precision and 0.527 recall.
- **Multiple chunks**: Greater coverage (0.681 recall) but lower precision (0.358).

### Query Performance Analysis
- **High-performing queries (IoU > 0.7)**: Specific factual questions about policies or statistics.
- **Moderate-performing queries (IoU 0.3-0.7)**: Questions about healthcare policies, foreign relations, etc.
- **Low-performing queries (IoU < 0.3)**: Questions requiring synthesis across multiple sections.

## Optimal Configurations by Query Type

| Query Type    | Best Configuration                | Justification                              |
|---------------|-----------------------------------|--------------------------------------------|
| General       | 15-20 tokens, 1-3 overlap, 3-4 chunks | Improves recall for distributed information |
| Precise       | 20-25 tokens, 3-4 overlap, 1 chunk | Maximizes precision for concrete information |
| Explanatory   | 30-40 tokens, 7-11 overlap, 1 chunk | Balance between context and precision       |
| Multi-aspect  | 30-40 tokens, 9-11 overlap, 2 chunks | Greater coverage for complex answers        |

## Project Structure
## Project Structure

- `opt_chunk.py`: Core implementation of the RAG retrieval pipeline, including:
    - `CachedEncoder`: Efficient embedding generation with caching to reduce redundant computations.
    - `Reranker`: Cross-encoder reranking component to improve retrieval quality.
    - `Chunker_RAG`: Main class managing chunking, embedding generation, and evaluation processes.
- `fixed_token_chunker.py`: Implementation of the `FixedTokenChunker` algorithm for document chunking.
- `final.ipynb`: Jupyter notebook containing detailed experiments, visualizations, and analysis.
- `Results/`: Directory storing evaluation results, metrics, and performance summaries.
- `requirements.txt`: File listing all dependencies required to run the project.
- `Corpus/`: Directory containing the dataset, including:
    - `questions_df.csv`: Queries and golden excerpts for evaluation.
    - `state_of_the_union.md`: Contains the State of the Union addresses used as the document corpus.
- `README.md`: Documentation and instructions for running the project.


## Running the Code

### Prerequisites
- Python 3.8+
- Required packages: Install dependencies using `pip install -r requirements.txt`.

### Dataset Preparation
1. Place the corpus files in the `./Corpus/` directory.
2. Ensure the queries and golden excerpts are available in `./Corpus/questions_df.csv`.

### Running Experiments
1. Execute the notebook `final.ipynb` for detailed analysis and visualizations.
2. For custom experiments, use the `Chunker_RAG` class.

### Hyperparameter Optimization
Run the optimization process to find the best chunking parameters for your specific corpus.

## Potential Improvements
- Exploring adaptive chunking approaches that consider semantic coherence.
- Implementing techniques to handle questions requiring multi-part answers.
- Considering query reformulation strategies to better match information structure in the corpus.
- Exploring more advanced reranking techniques and hybrid retrieval methods.
- Testing with different embedding models and dimensionality reduction methods.

## Conclusion
This project demonstrates that optimizing chunking parameters is critical for RAG system performance. The results highlight a clear tradeoff between precision and recall based on chunk size, overlap, and number of retrieved chunks. The optimal configuration depends on the specific use case, with medium-sized chunks (30-40 tokens) and moderate overlap providing the best general-purpose performance for this corpus.

The implementation of a two-stage retrieval approach (embedding-based retrieval followed by cross-encoder reranking) has proven particularly effective at enhancing retrieval quality, especially for complex queries where semantic understanding beyond keyword matching is critical.
