# Source of part of this code:
# https://github.com/miguelgfierro/ai_projects/


import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import lightgbm as lgb


def visualize_classifier(model, X, y, ax=None, cmap="rainbow"):
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(
        X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3
    )
    ax.axis("tight")
    # ax.axis("off")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Predict with the estimator
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        levels=np.arange(n_classes + 1) - 0.5,
        cmap=cmap,
        clim=(y.min(), y.max()),
        zorder=1,
    )

    ax.set(xlim=xlim, ylim=ylim)


def visualize_tree_sklearn(model, figsize=(7, 7), **kwargs):
    fig = plt.figure(figsize=figsize)
    _ = tree.plot_tree(model, filled=True, rounded=True, **kwargs)


def visualize_tree_lightgbm(model, figsize=(10, 10), **kwargs):
    lgb.plot_tree(
        model,
        figsize=figsize,
        orientation="vertical",
        show_info=["split_gain", "leaf_count"],
    )
