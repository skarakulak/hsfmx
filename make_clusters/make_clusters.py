#!/usr/bin/env python
# coding: utf-8


import argparse
import os
import pickle
import random
import pandas as pd
import numpy as np
import torch
from gmm_tree import gmm_clustering

from transformers import AutoConfig, AutoModelWithLMHead

def main():
    cache_dir = './transformers_cache'
    model_name = 'bert-base-cased'
    model = AutoModelWithLMHead.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config= AutoConfig.from_pretrained(model_name, cache_dir=cache_dir),
        cache_dir=cache_dir
    )

    cor_pd = pd.DataFrame.corr(pd.DataFrame(model.cls.predictions.decoder.weight.cpu().detach().numpy().T))
    cor_pd.shape

    n_trees = 50
    n_partitions = 16
    cuts = np.linspace(0, cor_pd.shape[0], n_partitions + 1).astype(int)
    idx = np.arange(cor_pd.shape[0])
    idx_dict = {}
    for i in range(n_trees):
        np.random.shuffle(idx)
        idx_dict_tree = {}
        for j in range(n_partitions):
            idx_dict_tree[j] = idx[cuts[j]:cuts[j + 1]].copy()
            gmm_clustering(
                    cor_pd.iloc[idx_dict_tree[j], idx_dict_tree[j]].reset_index(drop=True),
                    f'tree_partitioned/tree{i}_partition{j}.txt'
                    )
            idx_dict[i] = idx_dict_tree

    with open('tree_partitioned/indices.pickle', 'wb') as handle:
        pickle.dump(idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == 'main':
    main()
