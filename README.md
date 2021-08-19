# Adaptive Transfer Learning on Graph Neural Networks


We design an adaptive auxiliary loss weighting model to learn the weights of auxiliary tasks by quantifying the consistency between auxiliary tasks and the target task. In addition, we learn the weighting model through meta-learning. Our methods can be applied to various transfer learning approaches, it performs well not only in multi-task learning but also in pre-training and fine-tuning.


## DataSet
For **Open Academic Graph (OAG)**, we provide a heterogeneous graph containing highly-cited CS papers (8.1G) spanning from 1900-2020. You can download the preprocessed graph via [this link](https://drive.google.com/open?id=1a85skqsMBwnJ151QpurLFSa9o2ymc_rq). We split the data by their time: Pre-training ( t < 2014 ); Training ( 2014 <= t < 2017); Validation ( t = 2017 ); Testing ( 2018 <= t ). As we use the raw-text as attribute generation task for OAG, we provide a pre-trained word2vec model via [this link](https://drive.google.com/file/d/1ArdaMlPKVqdRGyiw4YSdUOV6CeFb2AmD/view?usp=sharing).

If you want to directly process from raw data, you can download via [this link](https://drive.google.com/open?id=1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz). After downloading it, run `preprocess_OAG.py` to extract features and store them in our data structure. 

For **Reddit**, we simply download the preprocessed graph using pyG.datasets API, and then turn it into our own data structure using `preprocess_reddit.py`. We randomly split the data into different sets.

## Setup

This implementation is based on pytorch_geometric. To run the code, you need the following dependencies:

- [Pytorch 1.3.0](https://pytorch.org/)
- [pytorch_geometric 1.3.2](https://pytorch-geometric.readthedocs.io/)
  - torch-cluster==1.4.5
  - torch-scatter==1.3.2
  - torch-sparse==0.4.3
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [tqdm](https://github.com/tqdm/tqdm)
- [dill](https://github.com/uqfoundation/dill)
- [pandas](https://github.com/pandas-dev/pandas)

You can simply run ```conda env create -f env.yaml``` to install all the necessary packages.

## Usage
We first introduce the arguments to control hyperparameters. There are mainly three types of arguments, for pre-training; for dataset; for model and optimization.

For pre-training, we provide arguments to control different modules for attribute and edge generation tasks:
```
  --attr_ratio                     FLOAT   The ratio (0~1) of attribute generation loss .       Default is 0.5.
  --attr_type                      STR     type of attribute decoder ['text' or 'vec']          Default is 'vec'
  --neg_samp_num                   INT     Whether to use layer-norm on the last layer.         Default is False.
  --queue_size                     INT     Max size of adaptive embedding queue.                Default is 256.
```  

For datasets, we provide arguments to control mini-batch sampling:
```
  --data_dir                       STR     The address of preprocessed graph.
  --pretrain_model_dir             STR     The address for storing the pre-trained models.
  --sample_depth                   INT     How many layers within a mini-batch subgraph         Default is 6.
  --sample_width                   INT     How many nodes to be sampled per layer per type      Default is 128.
```  

For both pre-training and fine-tuning, we provide arguments to control model and optimizer hyperparameters. We highlight some key arguments below:

```
  --conv_name                      STR     Name of GNN filter (model)                           Default is hgt.
  --scheduler                      STR     Name of learning rate scheduler                      Default is cycle (for pretrain) and cosine (for fine-tuning)
  --n_hid                          INT     Number of hidden dimension                           Default is 400.
  --n_layers                       INT     Number of GNN layers                                 Default is 3.
  --prev_norm                      BOOL    Whether to use layer-norm on previous layers.        Default is False.
  --last_norm                      BOOL    Whether to use layer-norm on the last layer.         Default is False.
  --max_lr                         FLOAT   Maximum learning rate.                               Default is 1e-3 (for pretrain) and 5e-4 (for fine-tuning).  
```

The following commands pretrain a 3-layer HGT over OAG-CS:
```bash
python pretrain_OAG.py --attr_type text --conv_name hgt --n_layers 3 --pretrain_model_dir /datadrive/models/gta_all_cs3
```


The following commands use the pre-trained model as initialization and finetune on the paper-field classification task(one sims):
```bash
python3 -u multi_OAG_PF_one.py --conv_name hgt --cuda 0 --n_epoch 500 --link_ratio 1 --attr_ratio 1 --w2v_dir <w2v_dir> --pretrain_model_dir <pretrain_model_dir> --data_dir <data_file> --model_dir <model_dir> --act_type sigmoid --n_batch 32 --use_pretrain
```

The following commands use the pre-trained model as initialization and finetune on Reddit dataset(one sims):
```bash
python3 -u multi_reddit_one.py --conv_name hgt --cuda 0 --n_epoch 200 --pretrain_model_dir <pretrain_model_dir> --data_dir <data_dir> --model_dir <model_dir> --batch_size 200 --use_pretrain
```


## Pre-trained Models

1. The 3-layer HGT model pre-trained over OAG-CS under Time-Transfer Setting via [this link](https://drive.google.com/file/d/1OyIRfpNXjaD0TiRF-_Upfl5hix3is5ca/view?usp=sharing)
2. The 3-layer HGT model pre-trained over Reddit via [this link](https://drive.google.com/file/d/1Ja4PJT2bkFH0qgoWXjGBjByIFPco4h-S/view?usp=sharing)


















### Citation

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{10.1145/3447548.3467450,
author = {Han, Xueting and Huang, Zhenhuan and An, Bang and Bai, Jing},
title = {Adaptive Transfer Learning on Graph Neural Networks},
year = {2021},
isbn = {9781450383325},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3447548.3467450},
doi = {10.1145/3447548.3467450},
abstract = {Graph neural networks (GNNs) is widely used to learn a powerful representation of
graph-structured data. Recent work demonstrates that transferring knowledge from self-supervised
tasks to downstream tasks could further improve graph representation. However, there
is an inherent gap between self-supervised tasks and downstream tasks in terms of
optimization objective and training data. Conventional pre-training methods may be
not effective enough on knowledge transfer since they do not make any adaptation for
downstream tasks. To solve such problems, we propose a new transfer learning paradigm
on GNNs which could effectively leverage self-supervised tasks as auxiliary tasks
to help the target task. Our methods would adaptively select and combine different
auxiliary tasks with the target task in the fine-tuning stage. We design an adaptive
auxiliary loss weighting model to learn the weights of auxiliary tasks by quantifying
the consistency between auxiliary tasks and the target task. In addition, we learn
the weighting model through meta-learning. Our methods can be applied to various transfer
learning approaches, it performs well not only in multi-task learning but also in
pre-training and fine-tuning. Comprehensive experiments on multiple downstream tasks
demonstrate that the proposed methods can effectively combine auxiliary tasks with
the target task and significantly improve the performance compared to state-of-the-art
methods.},
booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining},
pages = {565–574},
numpages = {10},
keywords = {graph neural networks, multi task learning, graph representation learning, transfer learning, GNN pre-training},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```


This implementation is mainly based on [pyHGT](https://github.com/acbull/pyHGT) API.
