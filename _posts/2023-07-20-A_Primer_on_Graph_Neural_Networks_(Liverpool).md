# A Primer on Graph Neural Networks

**Author**: [Alessio Devoto](https://alessiodevoto.github.io/)

This is an introductory tutorial to [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) a library for the design of deep Graph Neural Networks(GNNs). The first part is a re-adaptation of the documentation from the PyG website, training a GNN for graph classification on the [MUTAG](https://paperswithcode.com/dataset/mutag) dataset (using [PyTorch Lightning](https://www.pytorchlightning.ai/) for the training loop). The notebook is inspired by [Simone Scardapane](https://sscardapane.it/)'s material on GNNs.




## 1. 🚗 Setup the colab environment



```python
# We use a cpu based installation for torch geometric
# More info here https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install pytorch-lightning --quiet
```

    Collecting torch_geometric
      Downloading torch_geometric-2.3.1.tar.gz (661 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m661.6/661.6 kB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (4.66.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (1.23.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (1.11.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (3.1.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (2.31.0)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (3.1.1)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (1.2.2)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (5.9.5)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch_geometric) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (3.2.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (2023.7.22)
    Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch_geometric) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch_geometric) (3.2.0)
    Building wheels for collected packages: torch_geometric
      Building wheel for torch_geometric (pyproject.toml) ... [?25l[?25hdone
      Created wheel for torch_geometric: filename=torch_geometric-2.3.1-py3-none-any.whl size=910454 sha256=c6b0ce00952f98502c40ce5e77fc409ed8c16b2a623c3c7f6971b5f06ca7998e
      Stored in directory: /root/.cache/pip/wheels/ac/dc/30/e2874821ff308ee67dcd7a66dbde912411e19e35a1addda028
    Successfully built torch_geometric
    Installing collected packages: torch_geometric
    Successfully installed torch_geometric-2.3.1
    Looking in links: https://data.pyg.org/whl/torch-2.0.0+cpu.html
    Collecting pyg_lib
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/pyg_lib-0.2.0%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (627 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m627.0/627.0 kB[0m [31m16.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch_scatter
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_scatter-2.1.1%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (504 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m504.1/504.1 kB[0m [31m949.6 kB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch_sparse
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_sparse-0.6.17%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (1.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m8.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch_cluster
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_cluster-1.6.1%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (732 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m732.3/732.3 kB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch_spline_conv
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (205 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m205.7/205.7 kB[0m [31m592.4 kB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch_sparse) (1.11.2)
    Requirement already satisfied: numpy<1.28.0,>=1.21.6 in /usr/local/lib/python3.10/dist-packages (from scipy->torch_sparse) (1.23.5)
    Installing collected packages: torch_spline_conv, torch_scatter, pyg_lib, torch_sparse, torch_cluster
    Successfully installed pyg_lib-0.2.0+pt20cpu torch_cluster-1.6.1+pt20cpu torch_scatter-2.1.1+pt20cpu torch_sparse-0.6.17+pt20cpu torch_spline_conv-1.2.2+pt20cpu
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m727.7/727.7 kB[0m [31m6.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m764.8/764.8 kB[0m [31m9.9 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
# PyTorch imports
import torch
from torch.nn import functional as F
```


```python
# PyTorch-related imports
import torch_geometric as ptgeom
import torch_scatter, torch_sparse
```


```python
import pytorch_lightning as ptlight
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
```


```python
# Other imports
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
```


```python
matplotlib.rcParams['figure.dpi'] = 120 # I like higher resolution plots :)
```

## 2. 💾 Data

### 2.1 Download & Explore Dataset

Pytorch Geometric provides a number of datasets to use off-the-shelf, for all graph related tasks (graph, node or edge level tasks). Find a complete list [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html).

In this tutorial, we will use the **MUTAG** dataset. See the MUTAG page on [Papers With Code](https://paperswithcode.com/dataset/mutag) and related papers for more information about the dataset. This is a toy version, so we do not care too much about the final performance.


```python
# Download the dataset
mutag = ptgeom.datasets.TUDataset(root='.', name='MUTAG') # just like a plain Torch Dataset
```

    Downloading https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip
    Extracting ./MUTAG/MUTAG.zip
    Processing...
    Done!



```python
# Useful info stored in the dataset class

print(len(mutag))
print(mutag.num_classes) # Binary (graph-level) classification
print(mutag.num_features) # One-hot encoding for each node type (atom)
print(mutag.num_edge_features) # One-hot encoding for the bond type (we will ignore this)
```

    188
    2
    7
    4



```python
# Each graph in the dataset is represented as an instance of the generic Data object:
# https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
mutag_0 = mutag[0]
mutag_0
```




    Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])



What is the meaning of each of these fields ?

![](https://raw.githubusercontent.com/alessiodevoto/gnns_xai_liverpool/main/images/simple_graph.png)

⚠ Other datasets (e.g. for node level tasks) only contain one single *huge* Data object.


```python
# node features
mutag_0.x.shape
```




    torch.Size([17, 7])




```python
# graph class (remember we only have two classes as this is a binary classfication problem)
mutag_0.y
```




    tensor([1])




```python
# the adjacency matrix stored in COO format
mutag_0.edge_index.shape
```




    torch.Size([2, 38])




```python
# let's take a look at the first 4 edges
mutag_0.edge_index[:, :4]
```




    tensor([[0, 0, 1, 1],
            [1, 5, 0, 2]])




```python
# Inside utils (https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html)
# there are a number of useful tools, e.g., we can check that the graph is undirected (the adjacency matrix is symmetric)
ptgeom.utils.is_undirected(mutag_0.edge_index)
```




    True




```python
# are there self loops in this graph ?
ptgeom.utils.contains_self_loops(mutag_0.edge_index)
```




    False




```python
# any isolated components ?
ptgeom.utils.contains_isolated_nodes(mutag_0.edge_index)
```




    False



### 2.2 Data Visualization

As always, it is crucial to visualize (if possible) the data structures we are dealing with.

In the case of graphs, this can be prohibitive due to very high number of nodes. Luckily all our molecules are quite small.

Let's define a simple function to plot a graph, using `matplotlib` and the `networkx` package


```python
# This one is copy-pasted from: https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from matplotlib.pyplot import figure

# transform the pytorch geometric graph into networkx format
def to_molecule(data: ptgeom.data.Data) -> nx.classes.digraph.DiGraph:
    ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F',
                'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = ATOM_MAP[data['x'].index(1.0)]
        del data['x']
    return g

# plot the molecule
def draw_molecule(g, edge_mask=None, draw_edge_labels=True, draw_node_labels=True, ax=None, figsize=None):
    figure(figsize = figsize or (4, 3))

    # check if it's been already converted to a nx graph
    if not isinstance(g, nx.classes.digraph.DiGraph):
      g = to_molecule(g)

    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels if draw_node_labels else None, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                    font_color='red', ax=ax)

    if ax is None:
      plt.show()

# Let's try it
draw_molecule(mutag_0)
```


    
![png](_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_files/_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_23_0.png)
    



```python
# @title Visualize some graphs { run: "auto" }
mutag_idx = 5 # @param {type:"slider", min:0, max:187, step:1}

draw_molecule(mutag[mutag_idx])

```


    
![png](_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_files/_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_24_0.png)
    


### 2.3: Transformations

Transformations are a quick way to include standard preprocessing when loading the graphs (e.g., automatically computing edge from the nodes positions). They work pretty much like torchvision's transforms.

See the full list of available transformations here:

https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html


```python
# As an experiment, we load the graph with a sparse adjacency format instead of the COO list
mutag_adj = ptgeom.datasets.TUDataset(root='.', name='MUTAG', transform=ptgeom.transforms.ToSparseTensor())
```


```python
# The format has a number of useful methods that are already implemented: https://github.com/rusty1s/pytorch_sparse
# For example, we can perform a single step of diffusion on the node features efficiently with a sparse-dense matrix multiplication

mutag_adj[0]
```




    Data(x=[17, 7], edge_attr=[38, 4], y=[1], adj_t=[17, 17, nnz=38])




```python
mutag_adj[0].adj_t
```




    SparseTensor(row=tensor([ 0,  0,  1,  1,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  6,  6,  7,  7,
                                8,  8,  8,  9,  9,  9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14,
                               15, 16]),
                 col=tensor([ 1,  5,  0,  2,  1,  3,  2,  4,  9,  3,  5,  6,  0,  4,  4,  7,  6,  8,
                                7,  9, 13,  3,  8, 10,  9, 11, 10, 12, 11, 13, 14,  8, 12, 12, 15, 16,
                               14, 14]),
                 size=(17, 17), nnz=38, density=13.15%)



🔥 **Warmup Exercise no. 1**

Imagine you are a (probably crazy) chemist and you want to *add self loops* to all of the molecules in your dataset.

What would you do? Plot a graph after adding self loops. Hint: [this transform](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddSelfLoops.html#torch_geometric.transforms.AddSelfLoops)


```python
# We load the graph and add self loops to all nodes (probably doesn't make sense from a chemical point of view 🤔)
mutag_self = ptgeom.datasets.TUDataset(root='.', name='MUTAG', transform=ptgeom.transforms.AddSelfLoops())
mutag_self_0 = mutag_self[0]
draw_molecule(mutag_self_0)
```


    
![png](_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_files/_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_30_0.png)
    


### 2.4 Data loading

Data loaders are a nice utility to automatically build mini-batches (either a subset of graphs, or a subgraph extracted from a single graph) from the dataset.


Pytorch Geometric manages the batching by [stacking the adjacency matrices](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html) into a single huge graph.




```python
# Plain MUTAG without self loops
mutag = ptgeom.datasets.TUDataset(root='.', name='MUTAG') # just like a plain Torch Dataset
```


```python
# First, we split the original dataset into a training and test spart with a stratified split on the class
train_idx, test_idx = train_test_split(range(len(mutag)), stratify=[m.y[0].item() for m in mutag], test_size=0.25, random_state=11)
```


```python
# Build the two loaders
train_loader = ptgeom.loader.DataLoader(mutag[train_idx], batch_size=32, shuffle=True)
test_loader = ptgeom.loader.DataLoader(mutag[test_idx], batch_size=32)
```


```python
# Let us inspect the first batch of data
batch = next(iter(train_loader))
batch
```




    DataBatch(edge_index=[2, 1258], x=[568, 7], edge_attr=[1258, 4], y=[32], batch=[568], ptr=[33])




```python
# The batch is built by considering all the subgraphs as a single giant graph with unconnected components
print(batch.x.shape) # All the nodes of the 32 graphs are put together
print(batch.y.shape) # A single label for each graph
print(batch.edge_index.shape)
```

    torch.Size([568, 7])
    torch.Size([32])
    torch.Size([2, 1258])


🔥 **Warmup Exercise no. 2**

As we said, PyTorch Geometric creates batches by stacking together small graphs into a single large one.

Create a dataloader with batch_size = 2 and plot the first batch to check it's content.


```python
# Don't do this with large batch size
draw_molecule(next(iter(ptgeom.loader.DataLoader(mutag[train_idx], batch_size=2, shuffle=True))))
```


    
![png](_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_files/_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_38_0.png)
    



```python
# If we built this new huge graph, how do we keep track of all the small subgraphs 🤔 ?
# There is an additional property linking each node to its corresponding graph index
print(batch.batch.shape)
print(batch.batch[0:30])
```

    torch.Size([568])
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 2])


![](https://raw.githubusercontent.com/alessiodevoto/gnns_xai_liverpool/main/images/batching.png)


```python
# We can perform a graph-level average with torch_scatter, see the figure here for a visual explanation:
# https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
torch_scatter.scatter_sum(batch.x, batch.batch, dim=0).shape
```




    torch.Size([32, 7])




```python
# Alternatively, PyG has this implemented as a functional layer
ptgeom.nn.global_mean_pool(batch.x, batch.batch).shape
```




    torch.Size([32, 7])



## 3. 🪄 Design and train the Graph Neural Network

We have explored the data and created the Dataloaders, which will help us during the training. We are finally able to build the model!


```python
# Layers in PyG are very similar to PyTorch, e.g., this is a standard graph convolutional layer
gc = ptgeom.nn.GCNConv(mutag.num_features, 14)
```


```python
# Pay attention to the forward arguments
gc(batch.x, batch.edge_index).shape
```




    torch.Size([568, 14])




```python
# Different layers have different properties, see this "cheatsheet" from the documentation:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html
# For example, GCNConv accepts an additional "edge_weight" parameter to weight each edge.
```


```python
# If you are not used to PyTorch Lightning, see the 5-minutes intro from here:
# https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

train_losses = []
eval_accs = []

class MUTAGClassifier(ptlight.LightningModule):

  def __init__(self, hidden_features: int):
    super().__init__()
    self.gc1 = ptgeom.nn.GCNConv(mutag.num_features, hidden_features)
    self.gc2 = ptgeom.nn.GCNConv(hidden_features, hidden_features)      # two "hops" seems enough for these small graphs
    self.head = torch.nn.Linear(hidden_features, 1)                     # binary classification

  def forward(self, x, edge_index=None, batch=None, edge_weight=None):

    # unwrap the graph if the whole Data object was passed
    if edge_index is None:
      x, edge_index, batch = x.x, x.edge_index, x.batch

    # GNN layers
    x = self.gc1(x, edge_index, edge_weight)
    x = F.relu(x)
    x = self.gc2(x, edge_index, edge_weight)
    x = F.relu(x)

    x = ptgeom.nn.global_mean_pool(x, batch) # now it's (batch_size, embedding_dim)
    x = F.dropout(x, p=0.5)
    logits = self.head(x)

    return logits

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer


  def training_step(self, batch, _):

      logits = self.forward(batch.x, batch.edge_index, batch.batch)
      target = batch.y.unsqueeze(1)
      loss = F.binary_cross_entropy_with_logits(logits, target.float())

      self.log("train_loss", loss)
      self.log("train_accuracy", accuracy(logits, target, task='binary'), prog_bar=True, batch_size=32)
      train_losses.append(loss.detach())
      return loss

  def validation_step(self, batch, _):

    logits = self.forward(batch.x, batch.edge_index, batch.batch)
    target = batch.y.unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, target.float())

    self.log("eval_accuracy", accuracy(logits, target, task='binary'), prog_bar=True, batch_size=32)
    eval_accs.append(accuracy(logits, target, task='binary'))

```


```python
# print the model

model = MUTAGClassifier(hidden_features=256)
model
```




    MUTAGClassifier(
      (gc1): GCNConv(7, 256)
      (gc2): GCNConv(256, 256)
      (head): Linear(in_features=256, out_features=1, bias=True)
    )




```python
# forward one batch

batch_out = model(batch)
batch_out.shape
```




    torch.Size([32, 1])



### 3.1 Training the model




```python
# We save checkpoints every 50 epochs
# This is like taking 'snapshots' of the model every 50 epochs
# We will use this in the next notebook

checkpoint_callback = ptlight.callbacks.ModelCheckpoint(
    dirpath='./checkpoints/',
    filename='gnn-{epoch:02d}',
    every_n_epochs=50,
    save_top_k=-1)
```


```python
# define the trainer
trainer = ptlight.Trainer(max_epochs=80, callbacks=[checkpoint_callback])
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: False, used: False
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs



```python
# This is not a particularly well-designed model, we expect approximately 80% test accuracy
trainer.fit(model, train_loader, test_loader)
```

    INFO:pytorch_lightning.callbacks.model_summary:
      | Name | Type    | Params
    ---------------------------------
    0 | gc1  | GCNConv | 2.0 K 
    1 | gc2  | GCNConv | 65.8 K
    2 | head | Linear  | 257   
    ---------------------------------
    68.1 K    Trainable params
    0         Non-trainable params
    68.1 K    Total params
    0.272     Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=80` reached.



```python
# simple plots to visualize metrics

plt.figure(figsize=(5,4))
plt.plot(train_losses[::len(train_losses)//len(eval_accs)])
plt.plot(eval_accs)

plt.legend(['Loss', 'Accuracy'])
plt.show()
```


    
![png](_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_files/_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_54_0.png)
    



```python
# not working due to cookies settings in most cases
%reload_ext tensorboard
%tensorboard --logdir=/content/lightning_logs
```

## 4. 💪 Exercise time

Pytorch geometric contains a wide range of possibilities for Graph Convolutional layers. You can find them [here](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html).

1. Instead of the simple `GCNConv` we used, build a model making use of different layers, e.g. the GATConv. Train the model and compare the results with the ones we obtained. Are they better or worse?

2. (If we have time) Can we change the forward function of our model and also use edge weights. Is it beneficial for the training ?


```python
from torch_geometric.nn import GATConv

# Define the new model

train_losses = []
train_accs = []
eval_accs = []

class MyCoolGNN(ptlight.LightningModule):

  def __init__(self, hidden_features: int, heads: int):
    super().__init__()
    self.gat1 = GATConv(mutag.num_features, hidden_features, heads=heads)
    self.gat2 = GATConv(hidden_features * heads, hidden_features, heads=1, concat=False)
    self.head = torch.nn.Linear(hidden_features, 1)

  def forward(self, x, edge_index=None, batch=None, edge_weight=None):

    # unwrap the graph if the whole Data object was passed
    if edge_index is None:
      x, edge_index, batch = x.x, x.edge_index, x.batch

    x = F.relu(self.gat1(x, edge_index, edge_weight))
    x = F.relu(self.gat2(x, edge_index, edge_weight))

    x = ptgeom.nn.global_mean_pool(x, batch)
    x = F.dropout(x)
    logits = self.head(x)

    return logits

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
      return optimizer


  def training_step(self, batch, _):

      logits = self.forward(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
      target = batch.y.unsqueeze(1)

      loss = F.binary_cross_entropy_with_logits(logits, target.float())
      acc = accuracy(logits, target, 'binary')

      self.log('train loss', loss)
      self.log('train acc', acc.item(),  prog_bar=True, batch_size=32)
      train_losses.append(loss.detach().item())
      train_accs.append(acc.item())

      return loss

  def validation_step(self, batch, _):

      logits = self.forward(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
      target = batch.y.unsqueeze(1)
      acc = accuracy(logits, target, 'binary')

      self.log('eval acc', acc.item(), prog_bar=True, batch_size=32)
      eval_accs.append(acc)
      return acc

model = MyCoolGNN(256, 2)
```


```python
model(next(iter(train_loader))).shape
```




    torch.Size([32, 1])




```python
# Train (no callbacks needed this time)

trainer = ptlight.Trainer(max_epochs=100)

trainer.fit(model, train_loader, test_loader)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: False, used: False
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name | Type    | Params
    ---------------------------------
    0 | gat1 | GATConv | 5.1 K 
    1 | gat2 | GATConv | 131 K 
    2 | head | Linear  | 257   
    ---------------------------------
    137 K     Trainable params
    0         Non-trainable params
    137 K     Total params
    0.549     Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]


    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/loops/fit_loop.py:281: PossibleUserWarning: The number of training batches (5) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=100` reached.



```python
plt.figure(figsize=(5,4))
plt.plot(train_losses[::len(train_losses)//len(eval_accs)])
plt.plot(eval_accs)
plt.plot(train_accs[::len(train_losses)//len(eval_accs)])

plt.legend(['Loss', 'Eval Accuracy', 'Train Accuracy'])
plt.show()
```


    
![png](_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_files/_A_Primer_on_Graph_Neural_Networks_%28Liverpool%29_60_0.png)
    



```python

```
