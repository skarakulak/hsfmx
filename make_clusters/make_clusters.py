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
)

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

    # split the data into parts due to memory constraints
    # this creates only one tree. we can select different subsets
    # in the future to create randomness in the generated tree hierarchies
    for i in range(15):
        gmm_clustering(
                cor_pd.iloc[i * 1812: (i + 1) * 1812].reset_index(drop=True),
                f'tree_partitioned/cl{i}.txt'
                )
    gmm_clustering(
            cor_pd.iloc[(i + 1) * 1812:].reset_index(drop=True),
            f'tree_partitioned/cl{i + 1}.txt'
            )


if __name__ == 'main':
    main()
