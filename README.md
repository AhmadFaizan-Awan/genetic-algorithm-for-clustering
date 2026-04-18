# Genetic Algorithm Clustering Explorer

A small data mining project demonstrating clustering with a genetic algorithm and an interactive Streamlit dashboard.

## Overview

This project implements a genetic algorithm for clustering and provides a Streamlit app to explore results empirically. It supports:

- Iris dataset
- Synthetic blobs dataset
- Mall Customers dataset (`Annual Income` vs `Spending Score`)

The objective is to minimize within-cluster sum of squares (WCSS) while finding natural data groups.

## Files

- `streamlit_app.py` — interactive Streamlit dashboard
- `clustering_utils.py` — genetic algorithm implementation and dataset loaders
- `Mall_Customers.csv` — customer segmentation dataset
- `clustering_project.ipynb` — notebook for analysis and experimentation
- `enhance_notebook.py` — script to enhance the Jupyter notebook with detailed explanations, analyses, and final report
- `project_task.txt` — project assignment description for the semester project on GA clustering
- `main.py` — simple entry point placeholder
- `README.md` — project documentation

## Usage

```bash
# Install dependencies:
   uv sync
# Run the application:
   uv run streamlit run streamlit_app.py
```

Then open the URL shown by Streamlit in your browser.

## Dependencies

- Python 3.x
- streamlit
- numpy
- pandas
- scikit-learn
- plotly

## Notes

- The genetic algorithm uses:
  - population initialization
  - tournament selection
  - one-point crossover
  - mutation on centroid coordinates
- Mall Customers clustering is based on annual income and spending score.

## Future improvements

- add PCA visualization for high-dimensional datasets
- support additional datasets
- enhance GA operators and convergence criteria
