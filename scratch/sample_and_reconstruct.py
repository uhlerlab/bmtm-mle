'''
Output MLE trees in various formats (e.g. newick, cov matrix, etc.)
'''
import argparse
import json
import math
import random
from scratch.util import operator_norm

import numpy as np
from tqdm import tqdm

from scratch.tree import Tree
from scratch.solver import Solver
from scratch.runtime_comparison import long_tree_structures
from util import max_var, fr_norm_sq_one

def sample_cov(data):
    return [[a*b for b in data] for a in data]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--true_structure', type=str, default='0-0-0')
    parser.add_argument('--guess_structure', type=str, default='0-0-0')
    parser.add_argument('--vars', type=str, default='0-100-100-100')
    parser.add_argument('--estimator', type=str, default='us')
    parser.add_argument('--format', type=str, default='covariance')
    parser.add_argument('--out_file', type=str, default='samples.txt')
    parser.add_argument('--normalize_fr', type=int, default=1)
    parser.add_argument('--randomize_vars', type=int, default=0)
    args = parser.parse_args()

    true_arr = list(map(int, args.true_structure.split('-')))
    guess_arr = list(map(int, args.guess_structure.split('-')))

    tree = Tree()  
    tree.make_prefix(true_arr)

    if args.randomize_vars == 0:
        first_vars = list(map(float, args.vars.split('-')))
    else:
        first_vars = [random.uniform(0, 10) for _ in range(len(args.vars.split('-')))]
    tree.set_var(first_vars)
    coef = operator_norm(tree.cov_matrix())
    tree.set_var([a / coef for a in first_vars])
    print(operator_norm(tree.cov_matrix()))
    assert(abs(operator_norm(tree.cov_matrix()) - 1) < 1e-6)

    
    print(tree.get_var())

    def format_tree(t):
        if args.format == 'newick':
            t.set_labels(list(range(t.num_leaf_nodes())))
            return t.sparse_newick()
        elif args.format == 'covariance':
            return t.cov_matrix()

    out_objs = [format_tree(tree)] 

    for n in range(args.num_trials):
        # sample data
        data = tree.sample_data()

        if args.estimator == 'us':
            reconstruct_tree = Tree()
            reconstruct_tree.make_prefix(guess_arr)
            reconstruct_tree.set_data(data)

            solver = Solver()
            solver.predict_mle(reconstruct_tree)
            out_objs.append(format_tree(reconstruct_tree))
        elif args.estimator == 'empcov':
            s_arr = sample_cov(data) 
            out_objs.append(s_arr)
        elif args.estimator == 'frbmtm':
            reconstruct_tree = Tree()
            reconstruct_tree.make_prefix(guess_arr)
            reconstruct_tree.set_data(data)
            reconstruct_tree.fr_proj()
            out_objs.append(reconstruct_tree.cov_matrix())

        elif args.estimator == 'lineartree':
            s_data = sorted(data+[0])
            #prefix = list(reversed(range(0, tree.num_leaf_nodes()-1)))
            prefix = long_tree_structures(tree.num_leaf_nodes())[-1]
            diffs = [(s_data[i]-s_data[i+1])**2 for i in range(len(s_data)-1)]
            variances = []
            for i in range(tree.num_leaf_nodes()):
                variances.append(diffs[i])
                variances.append(0)

            reconstruct_tree = Tree()
            reconstruct_tree.make_prefix(prefix)
            reconstruct_tree.set_var(variances)
            out_objs.append(reconstruct_tree.cov_matrix())
        elif args.estimator == 'fralltree':
            pass
        else:
            raise ValueError('No match in estimator!')


    with open(args.out_file, 'w') as f:
        out = None
        if args.format == 'newick':
            out = '\n'.join(out_objs)
        elif args.format == 'covariance':
            lines = [str(tree.num_leaf_nodes())]
            for cov_matrix in out_objs:
                lines.append(','.join(','.join(map(str, l)) for l in cov_matrix))
            out = '\n'.join(lines)

        f.write(args.format+'\n'+out)