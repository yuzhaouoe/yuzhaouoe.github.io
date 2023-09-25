---
title: A primer on Graph Neural Networks with Pytorch Geometric
seo_title: Colab Notebook training a simple Graph convolutional network for graph classification on Mutag dataset with pytorch geometric.
---
# A primer on Graph Neural Networks with Pytorch Geometric

In this Colab Notebook we show how to use explainability methods on Graph Neural Networks.


The MUTAG dataset contains molecules, represented as graphs. Each node is an atom, and each edge is a bond between atoms. We first train a simple graph classifier, then explain its predictions with different interpretability methods.


We will use Pytorch Geometric for the manipulation of the graph data structures and the design of the Graph Neural Network. We will use Captum and GNNExplainer for graph interpretability.

For a better experience: <a href="https://colab.research.google.com/github/alessiodevoto/notebooks/blob/main/A_Primer_on_Explainability_for_GNNs_(Liverpool).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>