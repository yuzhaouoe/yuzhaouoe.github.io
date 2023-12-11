---
layout: single
author_profile: true
title: Manifold learning
seo_title: Simple short exaplanation of manifold learning in machine learning. 

tags:
    - theory

published: true
---
I stumbled across this concept a lot of times, so here I am writing a brief recap to myself.

### Manifold

A manifold is a [topological space](https://en.wikipedia.org/wiki/Topological_space) that locally resembles a Euclidean space near each point. More precisely, an n-dimensional manifold is a topological space with the property that each point has a neighborhood that is [homeomorphic](https://mathworld.wolfram.com/Homeomorphic.html) to an open subset of an n-dimensional Euclidean space.

A manifold is essentially a generalization of Euclidean space such that locally (small areas) are approximately the same as a Euclidean space, but the entire space fails to have the same properties of Euclidean space when observed in its entirety. 

![A two-dimensional manifold](https://scikit-learn.org/stable/_images/sphx_glr_plot_compare_methods_001.png)
*A two-dimensional manifold is any 2-D shape that can be made to fit in a higher-dimensional space by twisting or bending it, loosely speaking*

### Manifold hypothesis

The **manifold hypothesis** is the hypothesis that many high-dimensional data sets that occur in the real world actually lie along low-dimensional latent manifolds inside that high-dimensional space.[1][2][3] As a consequence of the manifold hypothesis, many data sets that appear to initially require many variables to describe, can actually be described by a comparatively small number of variables, likened to the local coordinate system of the underlying manifold.

### Manifold learning

Manifold learning is an approach to non-linear dimensionality reduction. Algorithms for this task are based on the idea that the dimensionality of many data sets is only artificially high, i.e. on the manifold hypothesis.

Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data. Though supervised variants exist, the typical manifold learning problem is unsupervised: it learns the high-dimensional structure of the data from the data itself, without the use of predetermined classifications. 

### Reference

1. Manifold learning [https://scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)
2. Manifold Hypothesis [https://deepai.org/machine-learning-glossary-and-terms/manifold-hypothesis#:~:text=The Manifold Hypothesis states that,within the high-dimensional space.&text=For every whole number there,similar to the cartesian plane](https://deepai.org/machine-learning-glossary-and-terms/manifold-hypothesis#:~:text=The%20Manifold%20Hypothesis%20states%20that,within%20the%20high%2Ddimensional%20space.&text=For%20every%20whole%20number%20there,similar%20to%20the%20cartesian%20plane).
3. Manifold [https://www.analyticsvidhya.com/blog/2021/02/a-quick-introduction-to-manifold-learning/#:~:text=In simpler terms%2C it means,lie is called Manifold Learning](https://www.analyticsvidhya.com/blog/2021/02/a-quick-introduction-to-manifold-learning/#:~:text=In%20simpler%20terms%2C%20it%20means,lie%20is%20called%20Manifold%20Learning).
4. Manifold learning comparison [https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py)