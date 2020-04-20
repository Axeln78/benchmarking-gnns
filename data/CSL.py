import numpy as np, time, pickle, random, csv
import torch
from torch.utils.data import DataLoader, Dataset

import os
import pickle
import numpy as np

import dgl

from sklearn.model_selection import StratifiedKFold, train_test_split

random.seed(42)


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    
def format_dataset(dataset):  
    """
        Utility function to recover data,
        INTO-> dgl/pytorch compatible format 
    """
    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        #graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
        # adding edge features for Residual Gated ConvNet, if not there
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

    return DGLFormDataset(graphs, labels)


def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 3:1:1
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 5 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 5 fold have unique test set.
    """
    root_idx_dir = './data/CSL/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}
    
    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
        print("[!] Splitting the data into train/val/test ...")
        
        # Using 5-fold cross val as used in RP-GNN paper
        k_splits = 5

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []
        
        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)
            
        for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
            remain_index, test_index = indexes[0], indexes[1]    

            remain_set = format_dataset([dataset[index] for index in remain_index])

            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                    range(len(remain_set.graph_lists)),
                                                    test_size=0.25,
                                                    stratify=remain_set.graph_labels)

            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + dataset.name + '_train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + dataset.name + '_val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + dataset.name + '_test.index', 'a+'))
            
            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")
        
    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + dataset.name + '_'+ section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


class CSL(torch.utils.data.Dataset):
    """
        Circular Skip Link Graphs: 
        Source: https://github.com/PurdueMINDS/RelationalPooling/
    """
    
    def __init__(self, path="data/CSL/"):
        self.name = "CSL"
        self.adj_list = pickle.load(open(os.path.join(path, 'graphs_Kary_Deterministic_Graphs.pkl'), 'rb'))
        self.graph_labels = torch.load(os.path.join(path, 'y_Kary_Deterministic_Graphs.pt'))
        self.graph_lists = []
        
        self.n_samples = len(self.graph_labels)
        self.num_node_type = 1
        self.num_edge_type = 1
        self._prepare()
        
    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing Circular Skip Link Graphs ...")
        for sample in self.adj_list:
            _g = dgl.DGLGraph()
            _g.from_scipy_sparse_matrix(sample)
            g = dgl.transform.remove_self_loop(_g)
            g.ndata['feat'] = torch.zeros(g.number_of_nodes()).long()
                
            # adding edge features as generic requirement
            g.edata['feat'] = torch.zeros(g.number_of_edges()).long()
            
            # NOTE: come back here, to define edge features as distance between the indices of the edges
            ###################################################################
            # srcs, dsts = new_g.edges()
            # edge_feat = []
            # for edge in range(len(srcs)):
            #     a = srcs[edge].item()
            #     b = dsts[edge].item()
            #     edge_feat.append(abs(a-b))
            # g.edata['feat'] = torch.tensor(edge_feat, dtype=torch.int).long()
            ###################################################################
            
            self.graph_lists.append(g)
        print("[I] Finished preparation after {:.4f}s".format(time.time()-t0))
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
    
def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g
    
    
    
    
class CSLDataset(torch.utils.data.Dataset):
    def __init__(self, name='CSL'):
        t0 = time.time()
        self.name = name
        
        dataset = CSL()

        print("[!] Dataset: ", self.name)

        # this function splits data into train/val/test and returns the indices
        self.all_idx = get_all_split_idx(dataset)
        
        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(5)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(5)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(5)]
        
        print("Time taken: {:.4f}s".format(time.time()-t0))
    
    def format_dataset(self, dataset):  
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        return DGLFormDataset(graphs, labels)
    
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e
    
    
    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(5):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(5):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)