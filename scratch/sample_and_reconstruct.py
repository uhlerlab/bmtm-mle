'''
Output MLE trees in newick format
'''
import argparse
import json
import math

import numpy as np
from tqdm import tqdm

from scratch.tree import Tree
from scratch.solver import Solver
from util import max_var

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--true_structure', type=str, default='0-0-0')
    parser.add_argument('--guess_structure', type=str, default='0-0-0')
    parser.add_argument('--out_file', type=str, default='samples.txt')
    args = parser.parse_args()

    true_arr = list(map(int, args.true_structure.split('-')))
    guess_arr = list(map(int, args.guess_structure.split('-')))

    tree = Tree()  
    tree.make_prefix(true_arr)
    vs = [100 for _ in range(tree.num())]
    vs[0] = 0
    tree.set_var(vs)
    
    print(tree.get_var())

    answers = [] 
    for n in range(args.num_trials):
        # sample data
        data = tree.sample_data()
        print(tree.get_var())
        print(data)

        reconstruct_tree = Tree()
        reconstruct_tree.make_prefix(guess_arr)
        reconstruct_tree.set_data(data)

        solver = Solver()
        solver.predict_mle(reconstruct_tree)
        print(reconstruct_tree.get_var())
        print(reconstruct_tree.zero_pattern())

        # put sparsity in readable format
        #reconstruct_tree.set_var([0, 1, 1, 1])
        reconstruct_tree.set_data(list(range(reconstruct_tree.num_leaf_nodes())))
        newick = reconstruct_tree.sparse_newick()
        print(newick)
        answers.append(newick)

    with open(args.out_file, 'w') as f:
        f.write('\n'.join(answers))
