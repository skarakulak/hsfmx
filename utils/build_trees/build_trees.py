#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
import os
import pickle
import pandas as pd
import pickle
import random
import torch
from tqdm import tqdm
from gmm_tree import gmm_clustering
from transformers import AutoConfig, AutoModelWithLMHead


def read_pkl(path):
    assert os.path.isfile(path), f'file does not exist: {path}'
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def get_weights_w_mapping(idx_mapping_path, weights):
    most_freq_idx_dict = read_pkl(idx_mapping_path)
    idx2new_idx = {e: i for i, e in most_freq_idx_dict.items() if e != 0}
    new_num_tokens = len(idx2new_idx) + 1

    w_mostfreq = weights.loc[[idx2new_idx[i] for i in range(1, new_num_tokens)]]
    w_unknown_tokens_mean = pd.DataFrame(weights[~weights.index.isin(
        [idx2new_idx[i] for i in range(1, new_num_tokens)]
    )].mean(axis=0)).T

    w_new = pd.concat([
        w_unknown_tokens_mean,
        w_mostfreq
    ]).reset_index(drop=True)
    return w_new


def main(save_dir, transformers_cache_dir, n_trees, col_sample_proportion, idx_mapping_path=None):
    model_name = 'bert-base-uncased'
    model = AutoModelWithLMHead.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config= AutoConfig.from_pretrained(
            model_name, cache_dir=transformers_cache_dir),
        cache_dir=transformers_cache_dir
    )

    weights = pd.DataFrame(model.cls.predictions.decoder.weight.cpu().detach().numpy())

    if idx_mapping_path is not None:
        weights = get_weights_w_mapping(idx_mapping_path, weights)

    print('> creating the trees')
    idx = np.arange(weights.shape[1])
    pbar = tqdm(range(n_trees))
    for i in pbar:
        np.random.shuffle(idx)
        pbar.set_description_str(f'(tree:{i})')
        cols = idx[: int(idx.shape[0] * col_sample_proportion)]
        gmm_clustering(
                weights.iloc[:, cols],
                os.path.join(save_dir, f'tree{i}.txt')
                )
        pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', type=str, default='hsfmx_trees')
    parser.add_argument('-transformers_cache_dir', type=str, default='transformers_cache')
    parser.add_argument('-n_trees', type=int, default=50)
    parser.add_argument('-col_sample_proportion', type=float, default=0.90)
    parser.add_argument('-idx_mapping_path', type=str, default=None)
    options = parser.parse_args()
    if not os.path.isdir(options.save_dir):
        os.makedirs(options.save_dir)
    main(
        options.save_dir,
        options.transformers_cache_dir,
        options.n_trees,
        options.col_sample_proportion,
        options.idx_mapping_path,
    )
