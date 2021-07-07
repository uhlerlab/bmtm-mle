'''
Show the results of sample and reconstruct over multiple files as TSNE embeddings
'''
import argparse
import math
import glob

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from scratch.util import fr_norm_sq, fr_norm_sq_one 
from scratch.tree import bhv_distance_owens, Tree
from scratch.gen_sample_series import add_reconstruct_args, reconstruct_file_tag

def in_tree(line):
    pre, vari, data, labels = (list(map(f, a.split(','))) 
        for a, f in zip(line.split('\t'), [int, float, float, int]))
    t = Tree()
    t.make_prefix(pre)
    t.set_var(vari)
    t.set_data(data)
    t.set_labels(labels)
    return t 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimators', nargs='+', type=str, default=['us', 'lineartreezero', 'gt'])
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--labels', type=int, default=0)
    parser.add_argument('--num_children', type=int, default=4)
    add_reconstruct_args(parser)
    args, unknown = parser.parse_known_args()

    recon_tag = reconstruct_file_tag(args)

    fmt = None

    def run(fn, gt=False):
        global fmt
        splits = []
        with open(fn, 'r') as f:
            lines = f.read().strip().split('\n')
            fmt = lines.pop(0)
            if fmt == 'tree':
                assert(len(lines) % 2 == 0)
                num_trials = len(lines)//2
                for ind in range(num_trials, len(lines)):
                    splits.append(in_tree(lines[ind]).get_splits())

            else:
                raise ValueError('format not supported')

        return splits

    all_splits = []
    for e in args.estimators:
        w = 'reconstruct_series-{}-{}-{}'.format('%05d' % args.num_children, e, recon_tag)
        label = w.split('-')[2]
        print('starting', label, w)
        result = run(w)
        all_splits.append((label, result))
    
    useful_splits = set()
    for _, splits in all_splits:
        for s in splits:
            useful_splits.update(s)
    useful_splits = list(useful_splits)

    def to_bin_vec(split_list, universe):
        return [int(u in split_list) for u in universe]

    all_bin_vec = np.zeros((sum(len(v) for _, v in all_splits), len(useful_splits))) 
    count = 0
    for k, v in all_splits:
        for s in v:
            all_bin_vec[count] = to_bin_vec(s, useful_splits)
            count += 1
    
    bin_vec_dict = []
    for k, v in all_splits:
        freq = {}
        for s in v:
            arr = tuple(to_bin_vec(s, useful_splits))
            if arr not in freq:
                freq[arr] = 1
            else:
                freq[arr] += 1
        bin_vec_dict.append((k, sorted(freq.keys()), freq))

    all_bin_vec = np.zeros((sum(len(v) for _, v, _ in bin_vec_dict), len(useful_splits))) 
    count = 0
    for k, v, _ in bin_vec_dict:
        for s in v:
            all_bin_vec[count] = s
            count += 1
    
    gt_bin = to_bin_vec(next(v for k, v in all_splits if k == 'gt')[0], useful_splits)


    
    print('splits')
    print(all_splits)
    print('bin_vec')
    print(all_bin_vec)
    print(bin_vec_dict)
    
    print('Starting TSNE...')
    embedded_bin_vec = TSNE(n_components=2).fit_transform(all_bin_vec)
    print('Done TSNE...')
    print(embedded_bin_vec)

    fig, ax = plt.subplots() 
    offset = 0
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'black', 'brown']
    for label_num, (k, v, lookup) in enumerate(bin_vec_dict):
        sizes = [100*(math.log(1+lookup[name])/math.log(10)) for name in v] 
        ax.scatter(embedded_bin_vec[offset:offset+len(v)][:,0], 
            embedded_bin_vec[offset:offset+len(v)][:,1], 
            sizes,
            alpha=(0.3 if k == 'gt' else 1),
            c=colors[label_num], label=k) 
        for i in range(len(v)):
            if all(a == b for a, b in zip(all_bin_vec[offset+i], gt_bin)):
                print('HIIT')
                ax.annotate('gt',
                    xy=(embedded_bin_vec[offset+i][0], 
                    embedded_bin_vec[offset+i][1]))
        if args.labels == 1:
            for i in range(len(v)):
                ax.annotate(''.join(map(str, map(int, all_bin_vec[offset+i]))),
                    xy=(embedded_bin_vec[offset+i][0], 
                    embedded_bin_vec[offset+i][1])) 
        offset += len(v)
    ax.legend()
    plt.title(args.title)
    plt.show()