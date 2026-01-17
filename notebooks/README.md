# Analysis Notebooks

This directory contains Jupyter notebooks for data analysis.

## Notebooks

| Notebook | Purpose | 
|----------|---------|
| `01_data_exploration.ipynb` | Dataset statistics and composition | 
| `02_model_analysis.ipynb` | Representation probing and error 
| `03_ablation_visualization.ipynb` | Ablation study figures | 

## Running Notebooks

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter server
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

## Dependencies

Notebooks require these additional packages:
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0` (for probing analysis)

## Notes

- Notebooks use pre-computed statistics for portability
- To regenerate from raw data, uncomment data loading cells
