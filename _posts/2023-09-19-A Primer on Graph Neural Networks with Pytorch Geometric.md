---
layout: single
author_profile: true
title: A primer on Graph Neural Networks with Pytorch Geometric
seo_title: Colab Notebook training a simple Graph convolutional network for graph classification on Mutag dataset with pytorch geometric.
---
# A primer on Graph Neural Networks with Pytorch Geometric
In this Colab Notebook we show how to train a simple Graph Neural Network on the MUTAG dataset. 


The MUTAG dataset contains molecules, represented as graphs. Each node is an atom, and each edge is a bond between atoms.

The goal is to predict whether a graph will mutate or not when in contact with Salmonella Typhimurium. 

We will use Pytorch Geometric for the manipulation of the graph data structures and the design of the Graph Neural Network.

For a better experience: <a href="https://colab.research.google.com/github/alessiodevoto/notebooks/blob/main/A_Primer_on_Graph_Neural_Networks_(Liverpool).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>