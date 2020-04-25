import os
import glob
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


def get_label_hierarchy(path):
    tree_labels_path = {}
    tree_label_full_path = {}
    tree_paths = set()
    p2t = {}
    with open(path,'r') as f:
        for line in f:
            label, l_path = line.split(',')[:2]
            full_path = l_path.strip()
            p2t[full_path] = int(label)
            tree_label_full_path[int(label)]=[int(l) for l in list(full_path)]
            path_labels ={}
            for k in range(1,1+len(full_path)):
                tree_paths.add(full_path[:k-1])
                path_labels[full_path[:k-1]] = int(full_path[k-1])
            tree_labels_path[int(label)] = path_labels
    path_inds = {k:i for i,k in enumerate(sorted(tree_paths,key=len))}
    tree_labels_path_indexed = {
        l:{path_inds[p]:p_l for p,p_l in path_dict.items()}
        for l, path_dict in tree_labels_path.items()
    }
    labels_hier_idx = {}
    for k, v in tree_labels_path_indexed.items():
        idx,labs = list(zip(*v.items()))
        labels_hier_idx[k] = (list(idx),list(labs))
    return labels_hier_idx, len(tree_paths), path_inds, p2t


class HierarchicalSoftmaxEnsemble(nn.Module):
    def __init__(self, input_dim, trees_path, device=None):
        super().__init__()
        self.num_paths = None
        self.path_indices = []
        self.path2label =  []
        self.num_hsfmx = 0
        self.labels2path_labels = []
        self.labels2path_labels_combined = {}
        self.trees_path = trees_path
        if device is None:
            self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.__device = device

        self.read_trees(trees_path)
        self.linear = nn.Linear(input_dim, self.num_paths * self.num_hsfmx)

    def read_trees(self, trees_path):
        for path in glob.glob(trees_path):
            labels_hier_idx, num_of_paths, path_idx, p2t = get_label_hierarchy(path)
            labels_hier_idx = {
                k:(torch.tensor(v[0]).long().to(self.__device),
                   torch.tensor(v[1]).float().to(self.__device))
                for k,v in labels_hier_idx.items()
            }
            self.labels2path_labels.append(labels_hier_idx)
            if self.num_paths is None:
                self.num_paths = num_of_paths
            else:
                assert self.num_paths == num_of_paths
            self.path_indices.append(path_idx)
            self.path2label.append(p2t)
        self.num_hsfmx = len(self.labels2path_labels)

        self.labels2path_labels_combined = {}

        for k in range(len(self.labels2path_labels[0])):
            comb_idx = torch.cat([
                h_idx[k][0] + self.num_paths*m
                for m, h_idx in enumerate(self.labels2path_labels)])
            comb_labels = torch.cat([h_idx[k][1] for h_idx in self.labels2path_labels])
            self.labels2path_labels_combined[k] = (comb_idx, comb_labels)

    def to(self, device):
        self.__device = device
        super().to(device)
        for i in range(len(self.labels2path_labels)):
            self.labels2path_labels[i] = {
                k:(torch.tensor(v[0]).long().to(self.__device),
                   torch.tensor(v[1]).float().to(self.__device))
                for k, v in self.labels2path_labels[i].items()
            }
        self.linear.to(device)
        return self

    def pred_label_single_hsfmx(self, pred, path_idx, p2t, start_ind=0):
        # TODO: implement beam search
        current_node=0
        current_path = []
        cur_node_path_idx = [0]
        while True:
            next_path_pred = pred[start_ind+cur_node_path_idx[-1]]
            current_path.append('1' if next_path_pred.item() >= 0 else '0')
            new_path = ''.join(current_path)
            if new_path in p2t:
                return p2t[new_path]
            cur_node_path_idx.append(path_idx[new_path])

    def get_labels(self, output, get_mode=True):
        pred = torch.Tensor([
            self.pred_label_single_hsfmx(row, path_idx, p2t, k * self.num_paths)
            for row in output
            for k, (path_idx, p2t) in enumerate(zip(self.path_indices, self.path2label))
        ]).long().to(self.__device).reshape(output.size(0), self.num_hsfmx)
        return pred.mode(dim=1)[0] if get_mode else pred

    def forward(self, x, target=None, collect_paths=True, pred_labels=False, pred_labels__get_mode=True):
        output = self.linear(x)
        if not collect_paths:
            return output

        y_hsfmx_idx = torch.cat([
            row * self.num_paths * self.num_hsfmx + self.labels2path_labels_combined[l][0]
            for row, l in enumerate(target.tolist())])
        target_hsfmx = torch.cat([
            self.labels2path_labels_combined[l][1]
            for row, l in enumerate(target.tolist())])
        output_hsfmx =  torch.gather(output.flatten(), 0, y_hsfmx_idx)

        if pred_labels:
            return output_hsfmx, target_hsfmx, self.get_labels(output, pred_labels__get_mode)
        return output_hsfmx, target_hsfmx
