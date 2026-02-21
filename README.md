# Indic Language NLP: Embedding & Clustering Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project explores Natural Language Processing (NLP) techniques applied to Indian languages (Indic languages). It involves data collection, generating high-dimensional embeddings using transformer-based models, and performing clustering analysis to understand linguistic relationships.

## 🚀 Features

- **Data Processing**: Pipeline for handling Indic script data.
- **Advanced Embeddings**: Utilizing `sentence-transformers` for multilingual embeddings.
- **Clustering Analysis**: Dynamic K-Means clustering with optimal K selection.
- **Interactive Visualization**: High-performance UMAP projections visualized with Plotly.
- **Evaluation Metrics**: Quantitative assessment using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

## 📁 Project Structure

```text
├── Notebooks/
│   ├── 1_Data_Collection.ipynb      # Scraping and processing raw text
│   ├── 2_Embedding_Generation.ipynb # Generating vector representations
│   └── 3_Clustering_Analysis.ipynb  # Clustering, evaluation, and visualization
├── data/                            # Raw and processed datasets (ignored by git)
├── streamlit/                       # Interactive demo application
└── requirements.txt                 # Project dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Final NLP"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

Follow the notebooks in chronological order:
1. **1_Data_Collection**: Prepares the dataset from CSV source files.
2. **2_Embedding_Generation**: Generates `embeddings.npy` using a transformer model.
3. **3_Clustering_Analysis**: Runs the core analysis and generates interactive plots.

> [!NOTE]
> Detailed visualizations in `3_Clustering_Analysis.ipynb` require a browser to view the interactive Plotly outputs.

## 📊 Visualizations

The project uses UMAP (Uniform Manifold Approximation and Projection) to project the high-dimensional language embeddings into 2D space, revealing natural clusters between Indo-Aryan and Dravidian language families.

---
*Created as part of an NLP research study on Indian Languages.*
