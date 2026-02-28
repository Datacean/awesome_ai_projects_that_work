# Gradient Boosting Decision Trees

A comprehensive guide to tree-based machine learning algorithms, covering Decision Trees, Gradient Boosting Decision Trees (GBDT), and Random Forests with scikit-learn and LightGBM.

## What You'll Learn

- Decision Trees fundamentals (Gini index vs Entropy)
- Gradient Boosting Decision Trees (GBDT) with LightGBM
- Random Forests and bagging techniques
- Tree visualization techniques
- Practical tips for parameter tuning and avoiding overfitting

## Prerequisites

Graphviz system binary is needed for tree visualization:

```bash
sudo apt-get install -y graphviz
```

## Quick Start

```bash
uv venv datacean --python 3.12
source datacean/bin/activate
uv pip install -r requirements.txt
jupyter notebook Trees.ipynb
```

## Models Covered

| Algorithm | Library | Key Idea |
|-----------|---------|----------|
| **Decision Tree** | scikit-learn | Split feature space to minimize impurity (Gini/Entropy) |
| **GBDT** | LightGBM | Sequentially ensemble weak trees, each correcting residuals |
| **Random Forest** | LightGBM | Bagging — train trees on random subsets, majority vote |
