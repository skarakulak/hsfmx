#!/usr/bin/env python
# coding: utf-8


import argparse
import os
import pickle
import random
import multiprocessing
import pandas as pd
import numpy as np
import torch
from gmm_tree import gmm_clustering

from transformers import AutoConfig, AutoModelWithLMHead


def create_tree(
        i, n_partitions, cuts, idx, idx_dict, corr_path, save_dir, sample_cols):
    corr_df = get_corr_df(corr_path)
    idx = idx.copy()
    np.random.shuffle(idx)
    idx_dict_tree = {}
    for j in range(n_partitions):
        idx_dict_tree[j] = idx[cuts[j]:cuts[j + 1]].copy()
        gmm_clustering(
                corr_df.iloc[idx_dict_tree[j]].reset_index(drop=True)
                if not sample_cols else
                corr_df.iloc[idx_dict_tree[j], idx_dict_tree[j]].reset_index(drop=True),
                os.path.join(save_dir, f'tree{i}_partition{j}.txt')
                )
    idx_dict[i] = idx_dict_tree

def create_corr_df(corr_path, model):
    if not os.path.isfile(corr_path):
        #print('> building the correlation matrix')
        corr_df = pd.DataFrame.corr(pd.DataFrame(model.cls.predictions.decoder.weight.cpu().detach().numpy().T))
        with open(corr_path, 'wb') as handle:
            pickle.dump(corr_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        corr_df = get_corr_df(corr_path)
    return corr_df.shape[0]

def get_corr_df(corr_path):
    assert os.path.isfile(corr_path)
    with open(corr_path, 'rb') as handle:
        corr_df = pickle.load(handle)
    return corr_df

def main(save_dir, transformers_cache_dir, sample_cols, n_proc):
    model_name = 'bert-base-cased'
    model = AutoModelWithLMHead.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config= AutoConfig.from_pretrained(
            model_name, cache_dir=transformers_cache_dir),
        cache_dir=transformers_cache_dir
    )

    corr_path = os.path.join(save_dir, 'corr_matrix.pkl')
    num_features = create_corr_df(corr_path, model)

    print('> creating the trees')
    n_trees = 50
    n_partitions = 16
    cuts = np.linspace(0, num_features, n_partitions + 1).astype(int)
    idx = np.arange(num_features)
    idx_dict = {}
    create_tree_p = functools.partial(
            create_tree,
            n_partitions=n_partitions,
            cuts=cuts,
            idx=idx,
            idx_dict=idx_dict,
            corr_path=corr_path,
            save_dir=save_dir,
            sample_cols=sample_cols,
            )
    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.map(create_tree_p, range(n_trees))

    with open(os.path.join(save_dir, 'indices.pickle'), 'wb') as handle:
        pickle.dump(idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', type=str, default='trees_partitioned')
    parser.add_argument('-transformers_cache_dir', type=str, default='transformers_cache')
    parser.add_argument('-sample_cols', action='store_true')
    parser.add_argument('-n_proc', type=int, default=4)
    options = parser.parse_args()
    os.makedirs(options.save_dir, exist_ok=True)
    main(options.save_dir, options.transformers_cache_dir, options.sample_cols, options.n_proc)
