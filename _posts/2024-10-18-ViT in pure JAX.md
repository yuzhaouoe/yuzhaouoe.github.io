---
layout: single
classes: wide
author_profile: true
title: Vision Transformer in *pure* JAX.
seo_title: Implementation of a Vision Transformers in pure JAX, no other frameworks/libraries.

published: true
---

I decided to do this for two reasons. The first reason is that, for years, I had to bear my Ph.D. advisor coming into the lab while I was happily coding my Pytorch mdodel, slowly sneaking at my back, stare at my screen and say - with a disappointed look - "you should definitely do this in JAX". The second reason is this nice [blog post](https://neel04.github.io/my-website/blog/pytorch_rant/) from Neel Gupta.

Every time I tried to use JAX, I ended up using Flax instead, which offers a kind of object oriented interface (similar to torch). While Flax is great, it introduces additional layers of abstraction and therefore I ended up wondering "why am I doing this". There are other frameworks as well, with different functionalities, like equinox, but they always add "another layer".

This time, I wanted to take a different path and stick to **pure JAX** without relying on any external libraries or abstractions.
What I do here is just a basic implementation of a Vision Transfomer. It's far from efficient, the code could be cleaner etc. etc. I just wanted to train a small model from scratch in JAX while using its "signature" functionalities like `vmap` and `jit`.

It was instructive for a series of reasons. First of all, if you use oure JAX, you see a model for what it really is: a bunch of numbers stored somewhere in your local machine + a function on those models. You end up thinking of a lot of things that in torch you just take for granted, like parameters initialization and the batch size (yes, batch size, more on this later).


For a better experience, open in Colab:  <a href="https://colab.research.google.com/drive/1wBA1UUde72yMDvZ7ITS8cFAx90HDwD5D#scrollTo=SUBw2ZtVN7Lr" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### Vision Transfomer
If you are not familiar with the Vision Transformer (ViT) architecture, you can take a look [here](https://arxiv.org/abs/2010.11929). Basically, ViTs treat image patches as tokens (like words in NLP models) and process them using transformer layers with bidirectional (non masked) attention. In this post, we’ll build a small ViT that can train on the Imagenette dataset, and you can even run it on your local GPU.


Speaking of GPUs, JAX offers seamless handling of hardware acceleration. It automatically detects and utilizes available GPUs without requiring explicit code changes.


```python
import jax

print("Available devices:", jax.devices()) # JAX will take care of the device placement for you
```

    Available devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]


### Initializing the model

Since JAX it’s a fully functional framework, you have to initialize all your parameters *outside* the model. This gives you a nice, low-level feel for how the model works. It’s not a magical black box onr an object, you’re working with something very "real and concrete": a set of weights + a function that processes them. The model itself is composed of: a set of parameteres (that we will store in a dictionary) + a function acting on those parameters.

In JAX, you always need a random key to initialize layers. This is great for ML practitioners, and you know what I'm referring to if you ever had to use [torch random seeding](https://neel04.github.io/my-website/blog/pytorch_rant/#seeding) and ended up with reproducibility errors. The main reason for JAX explicitly tracking the random keys without using a state is that this would compromise the execution of parallel code, that is one of the main perks of JAX. You can read more about randomness in JAX [here](https://jax.readthedocs.io/en/latest/jax.random.html)

Let's initiliaze the weights for our small ViT. What parameters are we going to need?  We need:

- a CLS (classification) token
- a projection to transform the patches into tokens
- a positional encoding
- N transformer blocks made of multihead attention and Feed Forward MLP
- a final head for classification

In order to make everything cleaner, we'll have a function to initialize each submodule of our Vit.
First let's define the hypeparameters of our model.


> This part is might be somewhat boring especially if you come from torch, you might want to skip and jump to the coding of the model function itself.


```python
# Vision Transformer hyper-parameters
image_size = 64
patch_size = 4
num_patches = (image_size // patch_size) ** 2

num_layers = 4
hidden_dim = 192
mlp_dim = 192*4


num_classes = 10
num_heads = 4
head_dim = hidden_dim//num_heads

```

Now we create a dictionary to store our weights.


```python
# initialize vit parameters
vit_parameters = {
    'patch_embed': None,
    'positional_encoding': None,
    'layers': [],
    'final_layer_norm': None,
    'head': [],
    'cls_token': None
}
```

We’ll also need a random key for parameter initialization:


```python
from jax import random

key = random.PRNGKey(42)
key, *layer_keys = random.split(key, num_layers+1)
```

As we mentioned before, we need to initialize each set of weights separately. Again, we could use a library for this, like optax, but we want to go through the process manually to better understand what's happening under the hood.

Let's first initialize class token, patch_embed, positional_encoding and head.


```python
import jax.numpy as jnp # this is what we use to manipulate tensors in JAX. It is supers similar to numpy
```


```python
# for the class token, we just need a single vector of the same size as a token
cls_token_key, key = random.split(key)
cls_token = jnp.zeros((1,hidden_dim))
vit_parameters['cls_token'] = cls_token

# for the patch embedding, we need to consider each patch and 3 channels and project it into the hidden dimension
patch_embed_key, key = random.split(key)
patch_embed = random.normal(patch_embed_key, ((3 * patch_size * patch_size), hidden_dim)) * jnp.sqrt(2.0 / (hidden_dim))
vit_parameters['patch_embed'] = patch_embed

# the positional encoding is just a value we add to each patch in the image
pos_enc_key, key = random.split(key)
pos_enc = random.normal(pos_enc_key, (num_patches,  hidden_dim)) * 0.02
vit_parameters['positional_encoding'] = pos_enc

# The head will consider only the class token and project it into the number of classes
head_key, key = random.split(key)
head_params = random.normal(head_key, (hidden_dim, num_classes)) * jnp.sqrt(6.0 / (hidden_dim))
head_bias = jnp.zeros(num_classes)
vit_parameters['head'] = (head_params, head_bias)

```

Now, it's time to initialize the transformer blocks. Each transformer block is made up of `attention`, `mlp` and `layer normalization`. We define a function to initilize each of these components.

I'll do it using [Xavier intialization](https://paperswithcode.com/method/xavier-initialization), but this is not crucial.


```python

def initialize_mlp(hidden_dim, mlp_dim, key):
    w1_key, w2_key = random.split(key)

    # Xavier uniform limit for w1 and w2
    limit = jnp.sqrt(6.0 / (hidden_dim + mlp_dim))

    # Xavier uniform initialization for weights
    w1 = random.uniform(w1_key, (hidden_dim, mlp_dim), minval=-limit, maxval=limit)
    b1 = jnp.zeros(mlp_dim)

    w2 = random.uniform(w2_key, (mlp_dim, hidden_dim), minval=-limit, maxval=limit)
    b2 = jnp.zeros(hidden_dim)

    return w1, b1, w2, b2


def initialize_attention(hidden_dim, num_heads, key):
    q_key, k_key, v_key = random.split(key, 3)

    # Limit for Xavier uniform
    fan_in = hidden_dim
    fan_out = head_dim * num_heads
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))

    # Random weights from uniform distribution
    q_w = random.uniform(q_key, (fan_in, fan_out), minval=-limit, maxval=limit)
    q_b = jnp.zeros(fan_out)
    k_w = random.uniform(k_key, (fan_in, fan_out), minval=-limit, maxval=limit)
    k_b = jnp.zeros(fan_out)
    v_w = random.uniform(v_key, (fan_in, fan_out), minval=-limit, maxval=limit)
    v_b = jnp.zeros(fan_out)

    return q_w, k_w, v_w, q_b, k_b, v_b


def initialize_layer_norm(hidden_dim):
    gamma = jnp.ones(hidden_dim)
    beta = jnp.zeros(hidden_dim)
    return gamma, beta

```

We are now ready to initialize all the weights in each transformer layer! We create a set of parameters for each layer and store them in our dictionary.


```python
key = random.PRNGKey(42)
key, *layer_keys = random.split(key, num_layers+1)

for i in range(num_layers):
    mlp_params = initialize_mlp(hidden_dim, mlp_dim, layer_keys[i])
    attn_params = initialize_attention(hidden_dim, num_heads, layer_keys[i])
    ln1_params = initialize_layer_norm(hidden_dim)
    ln2_params = initialize_layer_norm(hidden_dim)
    vit_parameters['layers'].append((mlp_params, attn_params, ln1_params, ln2_params))



# we also have a final layer norm outside the loop
final_layer_norm_key, key = random.split(key)
final_layer_norm_params = initialize_layer_norm(hidden_dim)
vit_parameters['final_layer_norm'] = final_layer_norm_params


```

### The Model is Just a Function

One thing we quickly notice about JAX is that everything is a function — including models. So once we’ve got our parameters ready, we’ll write the forward pass as a function. The ViT function will loop through a series of transformer blocks.

Here we come to another special feature o JAX: parallelization. We'll just code the model as if there were no batch dimension and then use `vmap` to automagically handle batches. This is a great improvement as we don't have to reason in one additional dimension and there will be no need for stuff like `batch,sequence,dim = input.shape` like in torch.

So from now on, forget about batching.



```python
# First, some utility functions: a relu and a softmax

def relu(input):
    return jnp.maximum(0, input)


def softmax(x, axis=-1):
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)
```

Now, the real "logic" of the model. Don't forget that each block is nothing but a *function* over the **model parameters** and an **input**.


```python

def mlp(x, mlp_params):

    # unpack the parameters
    w1, b1, w2, b2 = mlp_params

    # out = (Relu(x*w1 + b1))*w2 + b2
    up_proj = relu(jnp.matmul(x, w1) + b1)
    down_proj = jnp.matmul(up_proj, w2) + b2

    return down_proj


def self_attention(x, attn_params):

    # unpack the parameters
    q_w, k_w, v_w, q_b, k_b, v_b = attn_params

    # n and d_k are the sequence length of the input and the hidden dimension
    n, d_k = x.shape

    # project the input into the query, key and value spaces
    q = jnp.matmul(x, q_w) + q_b
    k = jnp.matmul(x, k_w) + k_b
    v = jnp.matmul(x, v_w)  + v_b


    # reshape to have heads
    q = q.reshape(num_heads, n, head_dim)
    k = k.reshape(num_heads, n, head_dim)
    v = v.reshape(num_heads, n, head_dim)

    # perform multi-head attention
    attention_weights_heads = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(head_dim)
    attention_weights_heads = jax.nn.softmax(attention_weights_heads, axis=-1)

    # output projection
    output = jnp.matmul(attention_weights_heads, v)
    output = output.reshape(n, d_k)

    return output


def layer_norm(x, layernorm_params):
    # a simple layer norm
    gamma, beta = layernorm_params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + 1e-6) + beta


def transformer_block(inp, block_params):

    # unpack the parameters
    mlp_params, attn_params, ln1_params, ln2_params = block_params

    # attention
    x = layer_norm(inp, ln1_params)
    x = self_attention(x, attn_params)
    skip = x + inp

    # mlp
    x = layer_norm(skip, ln2_params)
    x = mlp(x, mlp_params)
    x = x + skip

    return x



```

Now we have all the components, let's stack them in a transformer.
In order to transform an image into a set of tokens, we use [einops](https://einops.rocks/), which offers a highly expressive interface to reshape tensors.


```python
from einops import rearrange
def transformer(patches, vit_parameters):

    # reshape image from c,h,w -> num_patches, patch_size*patch_size
    patches = rearrange (patches, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

    # embed the patches
    patches = jnp.matmul(patches, vit_parameters['patch_embed'])

    # add positional encoding
    patches = patches + vit_parameters['positional_encoding']

    # append class token to sequence
    cls_token = vit_parameters['cls_token']
    patches = jnp.concatenate([cls_token, patches], axis=0)


    # forward through all transformer blocks
    for layer, block_params in enumerate(vit_parameters['layers']):
        patches = transformer_block(patches, block_params)

    # final layer norm
    patches = layer_norm(patches, vit_parameters['final_layer_norm'])

    # get the class token and apply the final head
    patches = patches[0, :]
    logits = jnp.matmul(patches, vit_parameters['head'][0]) + vit_parameters['head'][1]
    return logits

```

Let's test by forwarding a sample image


```python
sample_image = random.normal(key, (3 ,image_size, image_size))
prediction = transformer(sample_image, vit_parameters)
print("Output shape:", prediction.shape) # should be (num_classes,)

```

    Output shape: (10,)


### Vectorized Mapping with `vmap`

Before jumping into training, we'll look at one of the coolest features JAX offers: `vmap`. This allows you to vectorize your functions, meaning you can apply them over batches of data without writing explicit loops. In a way, it’s like automatic batching. You write a function that works on a single example, and `vmap` will apply it to all examples in a batch in one go.

For example, if you have a function that processes a single image, you can turn it into a function that processes an entire batch of images with just one line:

```python
batched_fn = jax.vmap(single_image_fn)
```

This can come in handy when applying the model over a batch of data. This means that we can run one pass of our transformer over a batch of images very easily. Let's try:



```python
bsize = 5
sample_images = random.normal(key, (bsize, 3 ,image_size, image_size))

# if we apply the transformer to a batch of images, we should get a batch of logits
# but this will raise an error
# prediction = transformer(sample_images, vit_parameters)
```

Let's apply `vmap`. We need to map each input in the batch (first dimension is 0) to *all* the parameters (second dimension in `None`).


```python
prediction = jax.vmap(transformer, in_axes=(0, None))(sample_images, vit_parameters)
print("Prediction shape:", prediction.shape)
```

    Prediction shape: (5, 10)


### Loss Function

Next up is the loss function. We’ll use the Cross-Entropy Loss, which is a standard choice for classification tasks.


```python
def cross_entropy_loss(patches, vit_parameters, ground_truth):
    prediction = jax.vmap(transformer, in_axes=(0, None))(patches, vit_parameters)
    logs = jax.nn.log_softmax(prediction)
    l = -jnp.mean(jnp.sum(ground_truth * logs, axis=-1))
    return l
```


```python
l = cross_entropy_loss(sample_images, vit_parameters, jnp.zeros((bsize, 10)).at[0, 1].set(1))
print("Loss:", l)
```

    Loss: 0.49248534


### Dataset Loading (Stealing From PyTorch)

For dataset loading, I’m going to steal some code from PyTorch. PyTorch’s data utilities are fantastic, and since this isn’t a post about data loading, we’ll skip the hassle of reinventing the wheel here.


```python
from torchvision.datasets import CIFAR10, Imagenette
from torchvision import transforms
from torch.utils.data import DataLoader

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


train_dataset = Imagenette(
    root='/home/aledev/datasets/imagenette3',
    size="160px",
    split='train',
    download=False,
    transform=transforms.Compose([transforms.Resize((image_size,image_size)),  transforms.ToTensor(), transforms.Normalize(mean, std)])
    )
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


test_dataset = Imagenette(
    root='/home/aledev/datasets/imagenette3',
    size="160px",
    split='val',
    download=False,
    transform=transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    )
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
```

Let's code a simple evaluation function that loops over the test data and computes accuracy


```python
from tqdm import tqdm


def eval(vit_parameters):

  correct = 0

  for(img, target) in tqdm(test_loader, desc="Eval", unit="item"):

    img = jnp.asarray(img, dtype=jnp.float32)
    target = jnp.asarray(target)

    logits = jax.vmap(transformer, in_axes=(0, None))(img, vit_parameters)
    prediction = jnp.argmax(logits, axis=-1)
    correct += jnp.sum(prediction == target).item()


  acc = correct / len(test_dataset)

  return acc

accuracy = eval(vit_parameters)
print("Accuracy before training", accuracy)
```

    Eval: 100%|██████████| 16/16 [00:18<00:00,  1.13s/item]

    Accuracy before training 0.06394904458598726


    


### Training and Just in Time compilation

Before we dive into training, let’s talk about `jit`. One of JAX’s biggest selling points is its ability to automatically compile and optimize your code using just-in-time (JIT) compilation. With JAX, you can wrap your functions in `jax.jit()` to make them faster by turning them into optimized code. It’s a one-liner and can massively speed up your training loop. In essence, `jit` lets you write Python code, and JAX will magically optimize it behind the scenes. It’s not even hard to use, so there’s no reason *not* to take advantage of it!

Here’s how you can JIT-compile your training step:

```python
@jax.jit
def train_step(params, data):
    # your training step logic here
```

#### Parameter updates

This is where JAX differs most from PyTorch. In PyTorch, you call `.backward()` on your loss, and it handles everything, i.e. computes loss and gradients, that you'll find stored in your model parameters. In JAX, you need to manually compute gradients and update parameters yourself, which gives you a more hands-on experience with the inner workings of optimization.

To perform gradient descent, we’ll compute the gradient of the loss with respect to the parameters. In JAX, you can do this using the `jax.values_and_grad` function:

```python
loss, grads = value_and_grad(cross_entropy_loss, argnums=1)(input, parameters, targets)
```

What will the gradients look like ? The gradients are just going to be a dictionary (pytree) with the same keys as the model parameters, but instead of holding the parameters, they will hold the gradients.


```python
from jax import value_and_grad

# fake labels and images
sample_images = random.normal(key, (bsize, 3 ,image_size, image_size))
sample_target = jnp.zeros((bsize, 10)).at[0, 1].set(1)
current_loss, grads = value_and_grad(cross_entropy_loss, argnums=1)(sample_images, vit_parameters, sample_target)

print("Current loss:", current_loss)
print("Gradients:", grads.keys())
```

    Current loss: 0.49248534
    Gradients: dict_keys(['cls_token', 'final_layer_norm', 'head', 'layers', 'patch_embed', 'positional_encoding'])


We now have a dictionary of gradients that mirrors the structure of our parameters. To update the parameters, we’ll perform simple gradient descent. The update rule is:

$$ \theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla_\theta L(\theta) $$

Where:
- $\theta$ is the parameter we’re updating, in our case the dictionary.
- $\eta$ is the learning rate.
- $\nabla_\theta L(\theta)$ is the gradient of the loss with respect to the parameter, in our case the gradients dictionary.

JAX has some great libraries for optimization, like `optax`, but for simplicity, we’ll just manually update the parameters using vanilla SGD. Notice that to do this we'd have to go throught the dictionary and update all values that have the same key. Fortunately, JAX has a function that does that for us: `jax.tree.map`.

We just have to tell the gradient descent rule:

```
updated_params = jax.tree.map(lambda p, g: p - 0.01 * g, vit_parameters, grads)
```

Putting everything together, the training step will look like this:


```python
from jax import jit, value_and_grad

@jit
def train_step(patches, vit_parameters, target_one_hot):
    # compute gradients
    current_loss, grads = value_and_grad(cross_entropy_loss, argnums=1)(patches, vit_parameters, target_one_hot)

    # update parameters
    updated_params = jax.tree.map(lambda p, g: p - 0.01 * g, vit_parameters, grads)

    return current_loss, updated_params
```

Finally, let's train the model. We don't expect any special results because we training without any optimization and with a super small model. Also, I'll only train for 30 epochs here, but you can let it go on for longer.


```python
import jax
from jax import value_and_grad
from jax import jit
from tqdm import tqdm

num_epochs = 20


for epoch in range(num_epochs):

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    #for (data, target) in tqdm(train_loader, desc=f'Train epoch {epoch}'):
    for i, (data, target) in progress_bar:

        # convert to numpy
        data = jnp.asarray(data)
        target = jnp.asarray(target)

        # reshape and get one hot fot loss
        target_one_hot = jax.nn.one_hot(target, num_classes)


        current_loss, vit_parameters = train_step(data, vit_parameters, target_one_hot)

        progress_bar.set_postfix({'loss': current_loss})


    eval_acc = eval(vit_parameters)
    print(f'Epoch: {epoch}, Eval acc: {eval_acc}')





```

    Epoch 1/20: 100%|██████████| 37/37 [00:29<00:00,  1.26it/s, loss=2.1218076]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.17item/s]


    Epoch: 0, Eval acc: 0.2573248407643312


    Epoch 2/20: 100%|██████████| 37/37 [00:17<00:00,  2.07it/s, loss=2.162859]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.10item/s]


    Epoch: 1, Eval acc: 0.25095541401273885


    Epoch 3/20: 100%|██████████| 37/37 [00:18<00:00,  2.01it/s, loss=1.9915687]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.06item/s]


    Epoch: 2, Eval acc: 0.27923566878980893


    Epoch 4/20: 100%|██████████| 37/37 [00:18<00:00,  2.03it/s, loss=1.9971651]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.13item/s]


    Epoch: 3, Eval acc: 0.27847133757961784


    Epoch 5/20: 100%|██████████| 37/37 [00:17<00:00,  2.08it/s, loss=2.0186331]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.18item/s]


    Epoch: 4, Eval acc: 0.2721019108280255


    Epoch 6/20: 100%|██████████| 37/37 [00:17<00:00,  2.10it/s, loss=1.965151]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.18item/s]


    Epoch: 5, Eval acc: 0.3072611464968153


    Epoch 7/20: 100%|██████████| 37/37 [00:17<00:00,  2.10it/s, loss=2.0264215]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.16item/s]


    Epoch: 6, Eval acc: 0.3100636942675159


    Epoch 8/20: 100%|██████████| 37/37 [00:17<00:00,  2.10it/s, loss=1.8914598]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.16item/s]


    Epoch: 7, Eval acc: 0.3228025477707006


    Epoch 9/20: 100%|██████████| 37/37 [00:18<00:00,  2.05it/s, loss=1.8803376]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.09item/s]


    Epoch: 8, Eval acc: 0.29987261146496813


    Epoch 10/20: 100%|██████████| 37/37 [00:17<00:00,  2.07it/s, loss=1.7691764]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.15item/s]


    Epoch: 9, Eval acc: 0.32203821656050957


    Epoch 11/20: 100%|██████████| 37/37 [00:18<00:00,  2.04it/s, loss=1.7844617]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.13item/s]


    Epoch: 10, Eval acc: 0.3370700636942675


    Epoch 12/20: 100%|██████████| 37/37 [00:18<00:00,  2.04it/s, loss=1.7681731]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.10item/s]


    Epoch: 11, Eval acc: 0.3327388535031847


    Epoch 13/20: 100%|██████████| 37/37 [00:18<00:00,  2.03it/s, loss=1.8342143]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.07item/s]


    Epoch: 12, Eval acc: 0.32636942675159236


    Epoch 14/20: 100%|██████████| 37/37 [00:18<00:00,  2.03it/s, loss=1.765596]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.11item/s]


    Epoch: 13, Eval acc: 0.34547770700636943


    Epoch 15/20: 100%|██████████| 37/37 [00:18<00:00,  2.00it/s, loss=1.7353936]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.11item/s]


    Epoch: 14, Eval acc: 0.34853503184713375


    Epoch 16/20: 100%|██████████| 37/37 [00:18<00:00,  2.02it/s, loss=1.7163022]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.11item/s]


    Epoch: 15, Eval acc: 0.3510828025477707


    Epoch 17/20: 100%|██████████| 37/37 [00:18<00:00,  2.02it/s, loss=1.7733448]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.08item/s]


    Epoch: 16, Eval acc: 0.32152866242038214


    Epoch 18/20: 100%|██████████| 37/37 [00:18<00:00,  2.04it/s, loss=1.5902878]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.11item/s]


    Epoch: 17, Eval acc: 0.34191082802547773


    Epoch 19/20: 100%|██████████| 37/37 [00:18<00:00,  2.03it/s, loss=1.7550975]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.13item/s]


    Epoch: 18, Eval acc: 0.3385987261146497


    Epoch 20/20: 100%|██████████| 37/37 [00:18<00:00,  2.02it/s, loss=1.6626304]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.13item/s]

    Epoch: 19, Eval acc: 0.34343949044585986


    


Hope you enjoyed this, please reach me at https://alessiodevoto.github.io/ if you have any questions or find inconsistencies! .
