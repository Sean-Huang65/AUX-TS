from functools import reduce
from GPT_GNN.data import *
from GPT_GNN.model_meta import *
import torch.nn as nn
from warnings import filterwarnings
import numpy as np
from sklearn.metrics import f1_score
import pdb

filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(
    description='Multi-task method, the primary task is OAG Paper-Field (L2) classification task')

'''
   Multi-task arguments 
'''
parser.add_argument('--link_ratio', type=float, default=1.0,
                    help='Ratio of loss-loss against primary-loss, range: [0-1]')
parser.add_argument('--attr_ratio', type=float, default=1.0,
                    help='Ratio of attr-loss against primary-loss, range: [0-1]')
parser.add_argument('--attr_type', type=str, default='vec',
                    choices=['text', 'vec'],
                    help='The type of attribute decoder')
parser.add_argument('--neg_samp_num', type=int, default=255,
                    help='Maximum number of negative sample for each target node.')
parser.add_argument('--queue_size', type=int, default=256,
                    help='Max size of adaptive embedding queue.')
parser.add_argument('--w2v_dir', type=str, default='/data/data1/v-bangan/GPT/w2v_all',
                    help='The address of preprocessed graph.')
parser.add_argument('--use_pretrain', help='Whether to use pre-trained model', action='store_true')
parser.add_argument('--pretrain_model_dir', type=str, default='/datadrive/models/gpt_all_cs',
                    help='The address for pretrained model.')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='/data/data1/v-zhehuang/HGT',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='/data/data1/v-zhehuang/GPT/gpt_all_reddit',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--model_add_name', type=str, default='tmp',
                    help='Additional name.')
parser.add_argument('--task_name', type=str, default='reddit',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=str, default=2,
                    help='Avaiable GPU ID')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many layers within a mini-batch subgraph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers', action='store_true')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--temperature', type=int, default=2, help='weight adaptation')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--max_lr', type=float, default=1e-3,
                    help='Maximum learning rate.')
parser.add_argument('--scheduler', type=str, default='cosine',
                    help='Name of learning rate scheduler.', choices=['cycle', 'cosine'])
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=16,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--valid_batch', type=int, default=10,
                    help='Number of valid batch (sampled graphs) for each epoch')
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient Norm Clipping')
'''
meta-optimizing arguments
'''
parser.add_argument('--n_fold', type=int, default=1)
parser.add_argument('--weight_emb_dim', type=int, default=1000)
parser.add_argument('--wlr', type=float, default=0.005)
parser.add_argument('--act_type', type=str, default='sigmoid')
parser.add_argument('--seed', type=int, default=43,
                    help='Number of output nodes for training')  


args = parser.parse_args()
args_print(args)
CUDA_STR = "cuda:" + str(args.cuda)
np.random.seed(args.seed)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
print('Start Loading Graph Data...')
graph = dill.load(open(os.path.join(args.data_dir, 'graph_reddit.pk'), 'rb'))
print('Finish Loading Graph Data!')
if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)

# primary task pre-define
target_type = 'def'
rel_stop_list = ['self']
pre_target_nodes   = graph.pre_target_nodes
train_target_nodes = graph.train_target_nodes
valid_target_nodes = graph.valid_target_nodes
test_target_nodes  = graph.test_target_nodes
repeat_num = int(len(pre_target_nodes) / args.batch_size // args.n_batch)


pre_target_nodes_au = np.concatenate([pre_target_nodes, np.ones(len(pre_target_nodes))]).reshape(2, -1).transpose()
# train_target_nodes_au = np.concatenate([train_target_nodes, np.ones(len(train_target_nodes))]).reshape(2, -1).transpose()

def node_classification_sample(seed, nodes, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    samp_nodes = np.random.choice(nodes, args.batch_size, replace = False)
    feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                inp = {target_type: np.concatenate([samp_nodes, np.ones(args.batch_size)]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, feature_extractor = feature_reddit)
    
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)

    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]

def GPT_sample(seed, target_nodes, time_range, batch_size, feature_extractor):
    np.random.seed(seed)
    samp_target_nodes = target_nodes[np.random.choice(len(target_nodes), batch_size)]
    threshold   = 0.5
    feature, times, edge_list, _, attr = sample_subgraph(graph, time_range, \
                inp = {target_type: samp_target_nodes}, feature_extractor = feature_extractor, \
                    sampled_depth = args.sample_depth, sampled_number = args.sample_width)
    rem_edge_list = defaultdict(  #source_type
                        lambda: defaultdict(  #relation_type
                            lambda: [] # [target_id, source_id] 
                                ))
    
    ori_list = {}
    for source_type in edge_list[target_type]:
        ori_list[source_type] = {}
        for relation_type in edge_list[target_type][source_type]:
            ori_list[source_type][relation_type] = np.array(edge_list[target_type][source_type][relation_type])
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if target_ser < source_ser:
                    if relation_type not in rel_stop_list and target_ser < batch_size and \
                           np.random.random() > threshold:
                        rem_edge_list[source_type][relation_type] += [[target_ser, source_ser]]
                        continue
                    el += [[target_ser, source_ser]]
                    el += [[source_ser, target_ser]]
            el = np.array(el)
            edge_list[target_type][source_type][relation_type] = el
            
            if relation_type == 'self':
                continue
                
    '''
        Adding feature nodes:
    '''
    n_target_nodes = len(feature[target_type])
    feature[target_type] = np.concatenate((feature[target_type], np.zeros([batch_size, feature[target_type].shape[1]])))
    times[target_type]   = np.concatenate((times[target_type], times[target_type][:batch_size]))

    for source_type in edge_list[target_type]:
        for relation_type in edge_list[target_type][source_type]:
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if target_ser < batch_size:
                    if relation_type == 'self':
                        el += [[target_ser + n_target_nodes, target_ser + n_target_nodes]]
                    else:
                        el += [[target_ser + n_target_nodes, source_ser]]
            if len(el) > 0:
                edge_list[target_type][source_type][relation_type] = \
                    np.concatenate((edge_list[target_type][source_type][relation_type], el))


    rem_edge_lists = {}
    for source_type in rem_edge_list:
        rem_edge_lists[source_type] = {}
        for relation_type in rem_edge_list[source_type]:
            rem_edge_lists[source_type][relation_type] = np.array(rem_edge_list[source_type][relation_type])
    del rem_edge_list
          
    return to_torch(feature, times, edge_list, graph), rem_edge_lists, ori_list, \
            attr[:batch_size], (n_target_nodes, n_target_nodes + batch_size)

def prepare_data_pr(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for _ in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), train_target_nodes, {1: True}))
        jobs.append(p)
    for _ in range(args.valid_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), valid_target_nodes, {1: True}))
        jobs.append(p)
    return jobs

def prepare_data_au(pool):
    jobs = []
    for _ in np.arange(args.n_batch):
        jobs.append(pool.apply_async(GPT_sample, args=(randint(), pre_target_nodes_au, {1: True}, args.batch_size, feature_reddit)))
    # jobs.append(pool.apply_async(GPT_sample, args=(randint(), train_target_nodes, {1: True}, args.batch_size, feature_reddit)))
    return jobs


data, rem_edge_list, ori_edge_list, _, _ = GPT_sample(randint(), pre_target_nodes_au, {1: True}, args.batch_size, feature_reddit)
node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
types = graph.get_types()

# model
'''
    Initialize GNN (model is specified by conv_name)
'''
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]), n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = False)
gnn_meta = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]), n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = False)
if args.use_pretrain:
    gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir, map_location=CUDA_STR)), strict=False)
    print('Load Pre-trained Model from (%s)' % args.pretrain_model_dir)

'''
    Initialize model for the primary task
'''
classifier = Classifier(args.n_hid, graph.y.max().item() + 1)
classifier_meta = Classifier(args.n_hid, graph.y.max().item() + 1)
model_pr = nn.Sequential(gnn, classifier).to(device)
print('Primary Model:\n')
print(model_pr)
print('\n')
params = sum([p.numel() for p in model_pr.parameters()])
print('Parameters: {}'.format(params))

'''
    Initialize model for the auxiliary tasks
'''
if args.attr_type == 'text':
    from gensim.models import Word2Vec

    w2v_model = Word2Vec.load(args.w2v_dir)
    n_tokens = len(w2v_model.wv.vocab)
    attr_decoder = RNNModel(n_word=n_tokens, ninp=gnn.n_hid, \
                            nhid=w2v_model.vector_size, nlayers=2)
    attr_decoder.from_w2v(torch.FloatTensor(w2v_model.wv.vectors))
else:
    attr_decoder = Matcher(gnn.n_hid, gnn.in_dim)

gpt_gnn = GPT_GNN(gnn=gnn, rem_edge_list=rem_edge_list, attr_decoder=attr_decoder, \
                  neg_queue_size=0, types=types, neg_samp_num=args.neg_samp_num, device=device)
gpt_gnn_meta = GPT_GNN(gnn=gnn_meta, rem_edge_list=rem_edge_list, attr_decoder=attr_decoder, \
                  neg_queue_size=0, types=types, neg_samp_num=args.neg_samp_num, device=device)
# gpt_gnn.init_emb.data = node_feature[node_type == node_dict[target_type][1]].mean(dim=0).detach()
init_emb = node_feature[node_type == node_dict[target_type][1]].mean(dim=0).detach()
gpt_gnn = gpt_gnn.to(device)

'''
Initialize weight model for multi-tasks
'''
vnet = Weight(6, args.weight_emb_dim, 1, args.act_type).to(device)
optimizer_v = torch.optim.Adam(vnet.parameters(), lr=args.wlr)

# Optimizer
# params = list(gpt_gnn.parameters()) + list(model_pr.parameters())
params = list(gpt_gnn.parameters()) + list(classifier.parameters())
optimizer = torch.optim.AdamW(params, weight_decay = 1e-2, eps=1e-06, lr = args.max_lr)

params_meta = list(gpt_gnn_meta.parameters()) + list(classifier_meta.parameters())
optimizer_meta = torch.optim.AdamW(params_meta, weight_decay = 1e-2, eps=1e-06, lr = args.max_lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, repeat_num * args.n_batch, eta_min=1e-6)
scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta, repeat_num * args.n_batch, eta_min=1e-6)

# prepare training
best_val = 0
train_step = 0
train_step_meta = 0
stats = []
res = []


# prepare for cosine similarty
link_private_param_name = [p[0] for p in gpt_gnn.named_parameters() if 'params' in p[0] and p[1].requires_grad]
attr_private_param_name = [p[0] for p in gpt_gnn.named_parameters() if 'attr_decoder' in p[0] and p[1].requires_grad]
private_param_name = link_private_param_name + attr_private_param_name
share_param_name = []
for m in [gpt_gnn, classifier]:
    for n, p in m.named_parameters():
        if n not in private_param_name and p.requires_grad:
            share_param_name.append(n)

cos_ = nn.CosineSimilarity()
all_param_names = set(link_private_param_name + attr_private_param_name + share_param_name)
dummy_flag = False #激活一下梯度
dummy_flag_meta = False
link_cos_mean = 0
attr_cos_mean = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs_pr = prepare_data_pr(pool)
jobs_au = prepare_data_au(pool)
criterion = nn.NLLLoss(reduce=False)

def getCos(args, device, classifier, params, gpt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_link_mean, loss_attr_mean):
    optimizer.zero_grad()
    loss_pr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    pr_share_grad = record_grad([gpt_gnn, classifier], share_param_name)
    pr_grad_flat = torch.cat([p.clone().flatten().to(device) for n,p in pr_share_grad.items()])

    optimizer.zero_grad()
    loss_link_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    link_share_grad = record_grad([gpt_gnn, classifier], share_param_name)
    link_grad_flat = torch.cat([p.clone().flatten().to(device) for n, p in link_share_grad.items()])

    optimizer.zero_grad()
    loss_attr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    attr_share_grad = record_grad([gpt_gnn, classifier], share_param_name)
    attr_grad_flat = torch.cat([p.clone().flatten().to(device) for n, p in attr_share_grad.items()])

    pr_link_cos = cos_(pr_grad_flat.view(1, -1), link_grad_flat.view(1, -1))
    pr_attr_cos = cos_(pr_grad_flat.view(1, -1), attr_grad_flat.view(1, -1))

    return pr_link_cos, pr_attr_cos

# Training + validation + testing
print('Start training...')
for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data for the primary task
    '''
    train_data_pr = [job.get() for job in jobs_pr[:args.n_batch]]
    valid_data_pr = [job.get() for job in jobs_pr[args.n_batch:]]
    pool.close()
    pool.join()
    '''
            Prepare Training and Validation Data for the auxiliary tasks
    '''
    pool = mp.Pool(args.n_pool)
    train_data_au = [job.get() for job in jobs_au]
    pool.close()
    pool.join()

    '''
            After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs_pr = prepare_data_pr(pool)
    jobs_au = prepare_data_au(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    '''
            Prepare for Training
    '''
    train_losses = []
    train_pr_losses = []
    train_link_losses = []
    train_attr_losses = []
    gpt_gnn.neg_queue_size = args.queue_size * epoch // args.n_epoch
    gpt_gnn.train()
    model_pr.train()
    gpt_gnn_meta.train()
    gnn_meta.train()
    classifier_meta.train()

    pr_link_cos = 0
    pr_attr_cos = 0
    '''
            Train on primary task and auxiliary tasks (2014 <= time <= 2016)
    '''
    for i in range(args.n_batch):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = train_data_pr[i]
        data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = train_data_au[i]
        node_feature_au, node_type_au, edge_time_au, edge_index_au, edge_type_au, node_dict_au, edge_dict_au = data
        node_feature_au = node_feature_au.detach()
        node_feature_au[start_idx: end_idx] = init_emb
        
        loss_pr_meta = 0 
        for j in range(args.n_fold):
            # meta model
            gnn_meta.to(device)
            gnn_meta.load_state_dict(gnn.state_dict())
            gpt_gnn_meta.to(device)
            gpt_gnn_meta.load_state_dict(gpt_gnn.state_dict())
            classifier_meta.to(device)
            classifier_meta.load_state_dict(classifier.state_dict())


            # primary task
            node_rep = gnn_meta.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier_meta.forward(node_rep[x_ids])
            loss_pr = criterion(res, ylabel.to(device))   # loss per example

            # auxiliary task

            node_emb_au = gpt_gnn_meta.gnn(node_feature_au.to(device), node_type_au.to(device), edge_time_au.to(device), \
                                edge_index_au.to(device), edge_type_au.to(device))
            loss_link, _ = gpt_gnn_meta.link_loss(node_emb_au, rem_edge_list, ori_edge_list, node_dict_au, target_type,
                                            use_queue=True, update_queue=True)
            if args.attr_type == 'text':
                loss_attr = gpt_gnn_meta.text_loss(node_emb_au[start_idx: end_idx], attr, w2v_model, device)
            else:
                # pdb.set_trace()
                loss_attr = gpt_gnn_meta.feat_loss(node_emb_au[start_idx: end_idx], torch.FloatTensor(attr).to(device))
            # loss_link = 0
            # loss_attr = 0
            # for log
            loss_pr_mean = loss_pr.mean()
            loss_link_mean = loss_link.mean()
            loss_attr_mean = loss_attr.mean()
            if not dummy_flag_meta:
                optimizer_meta.zero_grad()
                loss = loss_pr_mean + loss_link_mean + loss_attr_mean
                loss.backward(retain_graph=True)
                clean_param_name([gpt_gnn_meta, classifier_meta], share_param_name)
                clean_param_name([gpt_gnn_meta, classifier_meta], link_private_param_name)
                clean_param_name([gpt_gnn_meta, classifier_meta], attr_private_param_name)
                dummy_flag_meta = True
            else:
                loss = 0

            pr_link_cos, pr_attr_cos = getCos(args, device, classifier_meta, params_meta, gpt_gnn_meta, optimizer_meta, share_param_name, cos_, loss_pr_mean, loss_link_mean, loss_attr_mean)

            # embeddings for v-net
            loss_pr_emb = torch.stack((loss_pr, \
                    torch.ones([len(loss_pr)]).to(device), \
                    torch.zeros([len(loss_pr)]).to(device), \
                    torch.zeros([len(loss_pr)]).to(device), \
                    torch.full([len(loss_pr)], pr_link_cos.item()).to(device), \
                    torch.full([len(loss_pr)], pr_attr_cos.item()).to(device)
                ))
            loss_link_emb = torch.stack((loss_link, \
                    torch.zeros([len(loss_link)]).to(device), \
                    torch.ones([len(loss_link)]).to(device), \
                    torch.zeros([len(loss_link)]).to(device), \
                    torch.full([len(loss_link)], pr_link_cos.item()).to(device), \
                    torch.full([len(loss_link)], pr_attr_cos.item()).to(device)
                ))
            loss_attr_emb = torch.stack((loss_attr, \
                    torch.zeros([len(loss_attr)]).to(device), \
                    torch.zeros([len(loss_attr)]).to(device), \
                    torch.ones([len(loss_attr)]).to(device), \
                    torch.full([len(loss_attr)], pr_link_cos.item()).to(device), \
                    torch.full([len(loss_attr)], pr_attr_cos.item()).to(device)
                ))

            loss_pr_emb = loss_pr_emb.transpose(1,0)
            loss_link_emb = loss_link_emb.transpose(1,0)
            loss_attr_emb = loss_attr_emb.transpose(1,0)

            # compute weight
            v_pr = vnet(loss_pr_emb)
            v_link = vnet(loss_link_emb)
            v_attr = vnet(loss_attr_emb)

            # compute loss
            loss_pr_avg = (loss_pr * v_pr).mean()
            loss_link_avg = (loss_link * v_link).mean()
            loss_attr_avg = (loss_attr * v_attr).mean()
            loss_meta = loss_pr_avg + loss_link_avg + loss_attr_avg

            # one step update of model parameter (fake) (Eq.6)
            optimizer_meta.zero_grad()
            loss_meta.backward()
            torch.nn.utils.clip_grad_norm_(params_meta, args.clip)
            optimizer_meta.step()
            train_step_meta += 1
            scheduler_meta.step(train_step_meta)

            # primary loss with updated parameter (Eq.7)
            node_rep = gnn_meta.forward(node_feature.to(device), node_type.to(device), edge_time.to(device),
                                        edge_index.to(device), edge_type.to(device))
            res = classifier_meta.forward(node_rep[x_ids])
            loss_pr_meta += criterion(res, ylabel.to(device)).mean() #scalar

        # backward and update v-net params (Eq.9)
        optimizer_v.zero_grad()
        loss_pr_meta.backward()
        optimizer_v.step()

        # with the updated weight, update model parameters (true) (Eq.8)
        # primary task
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = train_data_pr[i]
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), edge_time.to(device),
                               edge_index.to(device), edge_type.to(device))
        res = classifier.forward(node_rep[x_ids])
        loss_pr = criterion(res, ylabel.to(device))   # loss per example
        # auxiliary task
        data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = train_data_au[i]
        node_feature_au, node_type_au, edge_time_au, edge_index_au, edge_type_au, node_dict_au, edge_dict_au = data
        node_feature_au = node_feature_au.detach()
        node_feature_au[start_idx: end_idx] = init_emb
        node_emb_au = gpt_gnn.gnn(node_feature_au.to(device), node_type_au.to(device), edge_time_au.to(device), \
                                  edge_index_au.to(device), edge_type_au.to(device))
        loss_link, _ = gpt_gnn.link_loss(node_emb_au, rem_edge_list, ori_edge_list, node_dict_au, target_type,
                                         use_queue=True, update_queue=True)
        if args.attr_type == 'text':
            loss_attr = gpt_gnn.text_loss(node_emb_au[start_idx: end_idx], attr, w2v_model, device)
        else:
            loss_attr = gpt_gnn.feat_loss(node_emb_au[start_idx: end_idx], torch.FloatTensor(attr).to(device))
        # for log
        loss_pr_mean = loss_pr.mean()
        loss_link_mean = loss_link.mean()
        loss_attr_mean = loss_attr.mean()
        if not dummy_flag:
            optimizer.zero_grad()
            loss = loss_pr_mean + loss_link_mean + loss_attr_mean
            loss.backward(retain_graph=True)
            clean_param_name([gpt_gnn, classifier], share_param_name)
            clean_param_name([gpt_gnn, classifier], link_private_param_name)
            clean_param_name([gpt_gnn, classifier], attr_private_param_name)
            dummy_flag = True

        pr_link_cos, pr_attr_cos = getCos(args, device, classifier, params, gpt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_link_mean, loss_attr_mean)

        print('link cos: %f\tattr cos: %f'%(pr_link_cos.item(), pr_attr_cos.item()))

        # embeddings for v-net
        loss_pr_emb = torch.stack((loss_pr, \
                torch.ones([len(loss_pr)]).to(device), \
                torch.zeros([len(loss_pr)]).to(device), \
                torch.zeros([len(loss_pr)]).to(device), \
                torch.full([len(loss_pr)], pr_link_cos.item()).to(device), \
                torch.full([len(loss_pr)], pr_attr_cos.item()).to(device)
            ))
        loss_link_emb = torch.stack((loss_link, \
                torch.zeros([len(loss_link)]).to(device), \
                torch.ones([len(loss_link)]).to(device), \
                torch.zeros([len(loss_link)]).to(device), \
                torch.full([len(loss_link)], pr_link_cos.item()).to(device), \
                torch.full([len(loss_link)], pr_attr_cos.item()).to(device)
            ))
        loss_attr_emb = torch.stack((loss_attr, \
                torch.zeros([len(loss_attr)]).to(device), \
                torch.zeros([len(loss_attr)]).to(device), \
                torch.ones([len(loss_attr)]).to(device), \
                torch.full([len(loss_attr)], pr_link_cos.item()).to(device), \
                torch.full([len(loss_attr)], pr_attr_cos.item()).to(device)
            ))
        # embeddings for v-net
        loss_pr_emb = loss_pr_emb.transpose(1, 0)
        loss_link_emb = loss_link_emb.transpose(1, 0)
        loss_attr_emb = loss_attr_emb.transpose(1, 0)

        # compute weight
        with torch.no_grad():
            v_pr = vnet(loss_pr_emb)
            v_link = vnet(loss_link_emb)
            v_attr = vnet(loss_attr_emb)

        # compute loss
        loss_pr_avg = (loss_pr * v_pr).mean()
        loss_link_avg = (loss_link * v_link).mean()
        loss_attr_avg = (loss_attr * v_attr).mean()
        loss_pr_avg_weighted = loss_pr.mean()
        loss_link_avg_weighted = loss_link.mean()
        loss_attr_avg_weighted = loss_attr.mean()

        print((
                  "Epoch: %d  Batch: %d  Train Loss Pr: %.2f  Train Loss Link: %.2f  Train Loss Attr: %.2f  Pr_Weight_Mean: %.4f Link_Weight_Mean: %.4f Attr_Weight_Mean: %.4f Pr_Weight_Std: %.4f Link_Weight_Std: %.4f Attr_Weight_Std: %.4f ") %
              (epoch, i, loss_pr_avg_weighted, loss_link_avg_weighted, loss_attr_avg_weighted, v_pr.mean().item(), v_link.mean().item(),
               v_attr.mean().item(), v_pr.std().item(), v_link.std().item(), v_attr.std().item()))

        # total loss
        loss = loss_pr_avg + loss_link_avg + loss_attr_avg
        train_pr_losses += [loss_pr_avg_weighted.cpu().detach().tolist()]
        train_link_losses += [loss_link_avg_weighted.item()]
        train_attr_losses += [loss_attr_avg_weighted.item()]

        # optimize model parameters
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        del res, loss_pr, loss_attr, loss_link, loss_pr_emb, loss_link_emb, loss_attr_emb, v_pr, v_link, v_attr
        del loss_pr_mean, loss_attr_mean, loss_link_mean, loss_pr_avg_weighted, loss_link_avg_weighted, loss_attr_avg_weighted


    '''
        Valid with only primary task (2017 <= time <= 2017)
    '''
    model_pr.eval()
    with torch.no_grad():
        valid_loss_pr_mean = 0
        valid_loss_link_mean = 0
        valid_loss_attr_mean = 0
        valid_res = []
        loss = 0
        for i in range(args.valid_batch):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data_pr[i]
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res  = classifier.forward(node_rep[x_ids])
            loss += criterion(res, ylabel.to(device)).mean() / args.valid_batch #scalar
            # loss = criterion(res, ylabel.to(device))
        
            '''
                Calculate Valid F1. Update the best model based on highest F1 score.
            '''
            valid_res += [f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')]
        
        valid_f1 = np.average(valid_res)
        if valid_f1 > best_val:
            best_val = valid_f1
            torch.save(model_pr,
                    os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid F1: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_pr_losses), \
                    loss.cpu().detach().tolist(), valid_f1))
        del res, loss
    del train_data_pr, valid_data_pr, train_data_au
    '''
        Test 
    '''
    if epoch > 0 and epoch % 5 == 0:
        best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name), map_location=CUDA_STR).to(device)
        best_model.eval()
        gnn_best, classifier_best = best_model
        with torch.no_grad():
            test_res = []
            for _ in range(10):
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                            node_classification_sample(randint(), test_target_nodes, {1: True})
                paper_rep = gnn_best.forward(node_feature.to(device), node_type.to(device), \
                            edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
                res = classifier_best.forward(paper_rep)
                test_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
                test_res += [test_f1]
        print('Best Test F1: %.4f' % np.average(test_res))

'''
    Evaluate the trained model via test set (time >= 2018)
'''

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name), map_location=CUDA_STR).to(device)
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_target_nodes, {1: True})
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        test_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
        test_res += [test_f1]
    print('Best Test F1: %.4f' % np.average(test_res))
