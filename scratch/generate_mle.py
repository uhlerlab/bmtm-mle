'''
Precompute a bunch of MLEs for a specific tree structure
'''
import argparse
import json
import math

import numpy as np
from tqdm import tqdm

from scratch.tree import Tree
from util import max_var

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=1000)
    parser.add_argument('--method', type=str, default='dual_annealing')
    parser.add_argument('--structure', type=str, default='0-2-0-0')
    parser.add_argument('--tag', type=str, default='diff')
    args = parser.parse_args()

    tree = Tree()  
    prefix_arr = list(map(int, args.structure.split('-')))
    tree.make_prefix(prefix_arr)

    p = tree.num_leaf_nodes()
    print('Num leaf nodes', p, 'Num trials', args.num_trials, 'Structure', args.structure)
    data = np.random.normal(size=(args.num_trials, p)) 
    file_dict = {'structure': prefix_arr, 'results': []}
    for i in tqdm(range(args.num_trials)):
        print('Data', data[i])
        tree.set_data(data[i])
        print(data[i])

        def compute_mle(d):
            return tree.mle(method=args.method, 
                max_var=max(10, max_var(d)), 
                accept=-1000, maxiter=5000)

        mle = compute_mle(data[i])
        if any(math.isnan(m) or math.isinf(m) for m in mle):
            print('Redoing...')
            mle = compute_mle(data[i])

        file_name = '{}{}_res_file_{}_{}.json'.format(
            (args.tag+'_' if args.tag.strip() != '' else ''), 
            args.method,
            args.num_trials, args.structure)
        file_dict['results'].append([list(data[i]), list(mle)])
        print('file name', file_name)
        with open(file_name, 'w') as res_file:
            json.dump(file_dict, res_file)