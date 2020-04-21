import os
import math
import glob
import re
import pickle
import argparse


def num_to_binary(num, max_num=15):
    num_digits = int(math.log2(max_num)) + 1
    l = [0] * num_digits
    for i in range(num_digits - 1, -1, -1):
        l[i] = '1' if num & 1 else '0'
        num >>= 1
    return ''.join(l)

def process_file(fname, output_fname, dict_idx, tree_idx, partition_idx, max_partition):
    path_prefix = num_to_binary(partition_idx, max_partition)
    with open(output_fname, 'a') as f_out:
        with open(fname, 'r') as f_in:
            for line in f_in:
                l_label, l_path = line.split(',')[:2]
                label = str(dict_idx[tree_idx][partition_idx][int(l_label)])
                l_path = path_prefix + l_path.strip()
                f_out.write(label + ',' + l_path + '\n')

def main(fpath, idx_fpath, output_fpath):
    re_str = fpath.replace('[0-9]*', '(\d+)')
    filenames = glob.glob(fpath)
    num_files = len(filenames)
    filenames = sorted([
        (tuple(int(k) for k in re.search(re_str, filename).group(1,2)),
        filename)
        for filename in filenames])
    with open(idx_fpath, 'rb') as handle:
        dict_idx = pickle.load(handle)
    n_trees, n_partitions = filenames[-1][0][0] + 1, filenames[-1][0][1] + 1

    for (tree_idx, partition_idx), filename in filenames:
        process_file(
            filename, output_fpath.format(tree_idx), dict_idx, tree_idx, partition_idx,n_partitions - 1
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_folder', type=str, default='trees_partitioned')
    parser.add_argument('-output_folder', type=str, default='hsfmx_trees')
    parser.add_argument('-input_file_pattern', type=str, default='tree[0-9]*_partition[0-9]*.txt')
    parser.add_argument('-idx_filename', type=str, default='indices.pickle')
    parser.add_argument('-output_file_pattern', type=str, default='tree{}.txt')
    options = parser.parse_args()
    os.makedirs(options.output_folder, exist_ok=True)
    main(
        os.path.join(options.input_folder, options.input_file_pattern),
        os.path.join(options.input_folder, options.idx_filename),
        os.path.join(options.output_folder, options.output_file_pattern)
    )
