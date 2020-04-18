import os
import math
import glob
import argparse

def num_to_binary(num, max_num=15):
    num_digits = int(math.log2(max_num)) + 1
    l = [0] * num_digits
    for i in range(num_digits - 1, -1, -1):
        l[i] = '1' if num & 1 else '0'
        num >>= 1
    return ''.join(l)

def process_file(fname, output_fname, file_idx, max_f_idx, label_start_idx):
    path_prefix = num_to_binary(file_idx, max_f_idx)
    num_labels = 0
    with open(output_fname, 'a') as f_out:
        with open(fname, 'r') as f_in:
            for line in f_in:
                label, l_path = line.split(',')[:2]
                label = str(label_start_idx + int(label))
                l_path = path_prefix + l_path.strip()
                f_out.write(label + ',' + l_path + '\n')
                num_labels += 1
    return label_start_idx + num_labels

def main(fpath, output_fpath):
    filenames = glob.glob(fpath)
    idx_left = fpath.rfind('[0-9]*')
    idx_right =  (idx_left + 6) - len(fpath)
    num_files = len(filenames)

    filenames = sorted([
        (int(filename[idx_left:idx_right]), filename )
        for filename in filenames])
    label_start_idx = 0
    for f_idx, filename in filenames:
        label_start_idx = process_file(
                filename, output_fpath, f_idx, len(filenames) - 1, label_start_idx)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fname', type=str, default='tree_partitioned/cl[0-9]*.txt')
    parser.add_argument('-output', type=str, default='hsfmx_tree/bert_clusters.txt')
    options = parser.parse_args()
    dir_chr_idx = options.output.rfind('/')
    if dir_chr_idx > 0:
        os.makedirs(options.output[:dir_chr_idx], exist_ok=True)
    main(options.fname, options.output)
