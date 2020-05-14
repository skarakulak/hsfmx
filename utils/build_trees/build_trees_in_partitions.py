#!/usr/bin/env python
# coding: utf-8


import argparse
import os
import pickle
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from gmm_tree import gmm_clustering

from transformers import AutoConfig, AutoModelWithLMHead

def main(save_dir, transformers_cache_dir, sample_cols):
    model_name = 'bert-base-cased'
    model = AutoModelWithLMHead.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config= AutoConfig.from_pretrained(
            model_name, cache_dir=transformers_cache_dir),
        cache_dir=transformers_cache_dir
    )

    corr_path = os.path.join(save_dir, 'corr_matrix.pkl')
    if os.path.isfile(corr_path):
        print('> loading the correlation matrix')
        with open(corr_path, 'rb') as handle:
            corr_df = pickle.load(handle)
    else:
        print('> building the correlation matrix')
        corr_df = pd.DataFrame.corr(pd.DataFrame(model.cls.predictions.decoder.weight.cpu().detach().numpy().T))
        with open(corr_path, 'wb') as handle:
            pickle.dump(corr_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(corr_df.shape)

    print('> creating the trees')
    n_trees = 50
    n_partitions = 16
    cuts = np.linspace(0, corr_df.shape[0], n_partitions + 1).astype(int)
    idx = np.arange(corr_df.shape[0])
    idx_dict = {}
    pbar = tqdm(range(n_trees))
    for i in pbar:
        np.random.shuffle(idx)
        idx_dict_tree = {}
        for j in range(n_partitions):
            pbar.set_description_str(f'(tree:{i} | partition:{j})')
            idx_dict_tree[j] = idx[cuts[j]:cuts[j + 1]].copy()
            gmm_clustering(
                    corr_df.iloc[idx_dict_tree[j]].reset_index(drop=True)
                    if not sample_cols else
                    corr_df.iloc[idx_dict_tree[j], idx_dict_tree[j]].reset_index(drop=True),
                    os.path.join(save_dir, f'tree{i}_partition{j}.txt')
                    )
        idx_dict[i] = idx_dict_tree
        pbar.update(1)

    with open(os.path.join(save_dir, 'indices.pickle'), 'wb') as handle:
        pickle.dump(idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', type=str, default='trees_partitioned')
    parser.add_argument('-transformers_cache_dir', type=str, default='transformers_cache')
    parser.add_argument('-sample_cols', action='store_true')
    options = parser.parse_args()
    dir_chr_idx = options.save_dir.rfind('/')
    if dir_chr_idx > 0:
        os.makedirs(options.save_dir[:dir_chr_idx], exist_ok=True)
    main(options.save_dir, options.transformers_cache_dir, options.sample_cols)
