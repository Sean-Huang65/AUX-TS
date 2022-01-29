# Adaptive Transfer Learning on Graph Neural Networks

This repository is the implementation of AUX-TS([arxiv](https://arxiv.org/abs/2107.08765)).

We design an adaptive auxiliary loss weighting model to learn the weights of auxiliary tasks by quantifying the consistency between auxiliary tasks and the target task. In addition, we learn the weighting model through meta-learning. Our methods can be applied to various transfer learning approaches, it performs well not only in multi-task learning but also in pre-training and fine-tuning.


## DataSet
We use datasets from [GPT-GNN](https://github.com/acbull/GPT-GNN).

## Setup

This implementation is based on pytorch_geometric. To run the code, you will need the following dependencies:

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
We first introduce the arguments to control hyperparameters. 

For pre-training, we provide arguments to control different modules for attribute and edge generation tasks:
```
  --n_fold                         INT     number of fold for cross validation                  Default is 1.
  --weight_emb_dim                 INT     embedding dimensions for weighting model             Default is 100.
  --wlr                            FLOAT   learning rate for weighting model                    Default is 1e-3.
  --lr                             FLOAT   learning rate for gnn model.                         Default is 1e-3.
  --act_type                       STR     activation type for weighting model                  Default is sigmoid
```  

The following commands pretrain a 3-layer HGT over OAG-CS:
```bash
python pretrain_OAG.py --attr_type text --conv_name hgt --n_layers 3 --pretrain_model_dir ./gta_all_cs3
```


The following commands use the pre-trained model as initialization and finetune on the paper-field classification task(one sim):
```bash
python3 -u multi_OAG_PF_one.py --conv_name hgt --cuda 0 --n_epoch 500 --link_ratio 1 --attr_ratio 1 --w2v_dir <w2v_dir> --pretrain_model_dir <pretrain_model_dir> --data_dir <data_file> --model_dir <model_dir> --act_type sigmoid --n_batch 32 --use_pretrain
```

The following commands use the pre-trained model as initialization and finetune on Reddit dataset(one sim):
```bash
python3 -u multi_reddit_one.py --conv_name hgt --cuda 0 --n_epoch 200 --pretrain_model_dir <pretrain_model_dir> --data_dir <data_dir> --model_dir <model_dir> --batch_size 200 --use_pretrain
```


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
booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining},
pages = {565–574},
numpages = {10},
keywords = {graph neural networks, multi task learning, graph representation learning, transfer learning, GNN pre-training},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```


This implementation is mainly based on [pyHGT](https://github.com/acbull/pyHGT) API.
