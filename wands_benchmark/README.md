# WANDS Search & Retrieval Benchmark

## 📋 Project Overview

This project implements a comprehensive benchmark for evaluating search and retrieval systems using the WANDS (Web Attributional NLP Dataset for Search) dataset. The pipeline includes data preparation, query generation, retrieval using SPLADE, ranking with LLMs, and evaluation.

## 🎯 Goals

1. Create a reproducible benchmark for search quality evaluation
2. Compare different LLM ranking capabilities (GPT-4 vs GPT-3.5)
3. Generate comprehensive metrics and visualizations
4. Ensure code quality through testing and documentation

## 🏗️ Project Structure

```
wands_benchmark/
├── data/
│   ├── raw/                  # Original WANDS data
│   └── processed/            # Cleaned and processed data
├── embeddings/               # Cached embeddings
│   └── splade/
├── queries/
│   ├── generated/           # Generated query files
│   └── retrieval/           # Retrieval results
├── ranking_results/          # Final ranking outputs
├── scripts/                  # Main pipeline scripts
│   ├── 01_prepare_dataset.py
│   ├── 02_generate_queries.py
│   ├── 03_retrieve_candidates.py
│   ├── 04_rank_candidates.py
│   ├── 05_evaluate.py
│   └── utils.py
├── tests/                    # Test suite
│   ├── unit/
│   └── integration/
├── notebooks/               # Jupyter notebooks for analysis
├── visualizations/           # Generated charts and plots
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md               # Project documentation
```

## 🛠️ Technical Details

### Data Preparation

The data preparation module handles loading, cleaning, and preprocessing the WANDS dataset. It implements:

- Feature standardization and cleaning
- SUPERKEY processing for consistent attribute representation
- Price generation based on product categories
- Filtering of low-quality records

### Query Generation

The query generation module creates synthetic queries from product attributes. It supports:

- Multiple query types (mini, regular, advanced)
- Structured JSON format with query parameters
- Query analysis with space types, negation, and hard filters detection

### SPLADE Retrieval

The retrieval module implements sparse vector search using SPLADE. Features include:

- Document indexing and embedding caching
- Efficient retrieval with similarity scoring
- Support for batch processing

### LLM Ranking

The ranking module implements pairwise ranking using GPT-4o and GPT-3.5. It includes:

- Pairwise comparison of products
- Confidence score collection
- Sliding window optimization

### Evaluation

The evaluation module calculates metrics and generates visualizations. Metrics include:

- nDCG@k (k=3,5,10)
- MRR (Mean Reciprocal Rank)
- Precision@k and Recall@k
- Confidence analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository

   ```sh
   git clone https://github.com/andreaparker/WANDS.git
   cd WANDS
   ```

2. Install dependencies

   ```sh
   pip install -r wands_benchmark/requirements.txt
   ```

3. Set up environment variables

   ```sh
   cp wands_benchmark/.env.example wands_benchmark/.env
   # Edit .env with your API keys
   ```

### Running the Pipeline

1. Prepare the dataset

   ```sh
   cd wands_benchmark/scripts
   python 01_prepare_dataset.py
   ```

2. Generate queries

   ```sh
   python 02_generate_queries.py
   ```

3. Retrieve candidates

   ```sh
   python 03_retrieve_candidates.py
   ```

4. Rank candidates

   ```sh
   python 04_rank_candidates.py
   ```

5. Evaluate results

   ```sh
   python 05_evaluate.py
   ```

## 📊 Results

After running the pipeline, you can find:

- Processed data in `data/processed/`
- Generated queries in `queries/generated/`
- Retrieval results in `queries/retrieval/`
- Ranking results in `ranking_results/`
- Evaluation metrics and visualizations in `visualizations/`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
