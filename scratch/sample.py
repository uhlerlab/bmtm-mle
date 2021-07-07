'''
Output samples from a BMTM
'''
import argparse
import random
from itertools import chain

import numpy as np

from scratch.tree import Tree
from scratch.util import operator_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--structure', type=str, default='0-0-0')
    parser.add_argument('--vars', type=str, default='0-100-100-100')
    parser.add_argument('--out_file', type=str, default='samples.txt')
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--randomize_vars', type=int, default=1)
    parser.add_argument('--random_fo', type=int, default=0)
    parser.add_argument('--constant', type=int, default=0)
    parser.add_argument('--random_method', type=str, default='uniform')
    args = parser.parse_args()

    true_arr = list(map(int, args.structure.split('-')))

    ground_truths, datas = [], []

    assert(args.random_method != 'exponentialultrametric' or (args.random_fo == 0))

    for i in range(args.num_trials):
        if i == 0 or args.constant == 0:
            tree = Tree()
            tree.make_prefix(true_arr)

            if args.randomize_vars == 0:
                first_vars = list(map(float, args.vars.split('-')))
            else:
                if args.random_method == 'uniform':
                    first_vars = [random.uniform(0, 10) for _ in range(tree.num())]
                elif args.random_method == 'uniformaway':
                    first_vars = [random.uniform(10, 20) for _ in range(tree.num())]
                elif args.random_method == 'uniform_constant':
                    pass
                elif args.random_method == 'exponential' or args.random_method == 'exponentialultrametric':
                    first_vars = list(np.random.exponential(1, (tree.num(),)))
                elif args.random_method == 'symmetric' or args.random_method == 'symmetricnonormalize':
                    first_vars = [1 for _ in range(tree.num())]
                else:
                    raise ValueError('The given random_method is unknown')
                
            tree.set_var(first_vars)
            if args.random_fo == 1:
                fo_seed = [random.randrange(0, 20) for _ in range(tree.num())]
                tree.random_fo(fo_seed)

            #if args.random_method == 'exponentialultrametric':
            #    nodes = [tree.get_leaf(i) for i in range(tree.num_leaf_nodes())]
            #    vars = [a.covar(a) for a in nodes]
            #    max_var = max(vars)
            #    for i, n in enumerate(nodes):
            #        n.above_var += max_var - vars[i]
            #    
            #    tree.set_var([v/max_var for v in tree.get_var()])

            if args.random_method == 'exponentialultrametric':
                p = 0.01
                num = tree.num_leaf_nodes()
                nt = Tree()
                nt.above_var = 0.1
                while nt.num_leaf_nodes() < num:
                    leaves = [nt.get_leaf(i) 
                        for i in range(nt.num_leaf_nodes()) 
                            if random.uniform(0, 1) < p]
                    random.shuffle(leaves)
                    for leaf in leaves:
                        leaf.make_child()
                        leaf.make_child()
                        if nt.num_leaf_nodes() >= num:
                            break
                    for i in range(nt.num_leaf_nodes()):
                        leaf = nt.get_leaf(i)
                        leaf.above_var += 0.01
                tree = nt


            if args.normalize and args.random_method != 'symmetricnonormalize':
                coef = operator_norm(tree.cov_matrix())
                tree.set_var([a / coef for a in tree.get_var()])
                if args.random_fo == 1:
                    tree.random_fo(fo_seed)
                assert(abs(operator_norm(tree.cov_matrix()) - 1) < 1e-6)

        # assert(tree.get_var() == first_vars)

        ground_truths.append((tree.get_prefix(), tree.get_var()))
        datas.append(tree.sample_data())
    
    assert(len(ground_truths) == len(datas))

    with open(args.out_file, 'w') as f:
        lines = [str(len(ground_truths))]
        for structure, vari  in ground_truths:
            lines.append('-'.join(map(str, tree.get_prefix())) + '\t' + ','.join(map(str, vari)))
        for sample in datas:
            lines.append(','.join(map(str, sample)))
        f.write('\n'.join(lines))