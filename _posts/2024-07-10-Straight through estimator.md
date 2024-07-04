---
layout: single
classes: wide
author_profile: true
title: Straight Through Estimator
seo_title: A blog post to explain what is the straight through estimator which is used a lot in deep learning.

tags:
    - theory

published: false
---
The Straight-Through Estimator is a method used to approximate the gradients of non-differentiable functions, especially in the context of backpropagation. In deep learning, we often deal with neural networks where the loss function needs to be differentiable to perform gradient-based optimization. However, some operations, such as those involving binary or discrete variables, are inherently non-differentiable. The STE provides a way to bypass this issue by approximating the gradient during the backward pass. 

Neural networks typically require differentiable activation functions to propagate error gradients backward through the network. However, certain operations, like the binary step function, are non-differentiable. For instance, when dealing with binary neural networks, the activation function might be a sign function, which outputs -1 or 1. The gradient of this function is zero almost everywhere, making it impossible to use standard backpropagation.


In applications like model compression and deployment on hardware with limited precision (e.g., mobile devices), we often quantize weights and activations to binary or low-bit representations. Training these quantized networks directly using traditional gradient descent methods is impractical because the quantization process is non-differentiable.

The STE is conceptually simple. During the forward pass, the network uses the actual non-differentiable function (e.g., a binary step function). During the backward pass, instead of computing the true gradient, which may be zero or undefined, the STE approximates the gradient using a differentiable surrogate function.





<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>




| Metric                  | explanation               | primarily affects                                | primarily affected by        | hardware independent |
|------------------------ |---------------- |----------------------------------------|------------------------|--------------|
| Parameters size         | memory footprint of model                | latency, storage                     | model architecture & implementation,  floating point precision | ✅|
| FLOPs ( and MACs)       | number of operations required by model                 | latency, energy         | model architecture & implementation | ✅|
| Latency                 | time required for one inference                |  end user experience | Parameters size, FLOPs, Hardware| ❌ |
| Energy                  | power consumed by operation                | training/inference cost | model architecture, implementation | ❌ |
| Throughput               | inferences per time frame | end user experience, energy  | all |  ❌ |


#### References
- [The Efficiency Misnomer](https://arxiv.org/abs/2110.12894)
- [Efficient AI MIT course](https://hanlab.mit.edu/courses/2023-fall-65940)
- [Arithmetic intensity](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understand-perf)
- [Floating point precision at Meta](https://engineering.fb.com/2018/11/08/ai-research/floating-point-math/)