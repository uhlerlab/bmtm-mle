
'''
Output samples from a BMTM
'''
import argparse
import math
import subprocess
import random
import os.path

from scratch.runtime_comparison import long_tree_structures, fat_tree_structures, stars
from scratch.tree import Tree

def pow_of_2(c):
    if c == 2:
        return True
    if c < 2:
        return False
    return c % 2 == 0 and pow_of_2(c//2)

def binary_tree(num_leaves):
    assert(pow_of_2(num_leaves))
    sqr = int(math.log(num_leaves)/math.log(2))
    return fat_tree_structures(sqr - 1)

def random_binary_tree(max_nodes):
    st = Tree()
    st.make_prefix([0, 0])
    while st.num_leaf_nodes() < max_nodes:
        pi = random.randrange(st.num_leaf_nodes())
        leaf = st.get_leaf(pi)
        leaf.make_child()
        leaf.make_child()

    return [st.get_prefix()]

TREE_TYPES = {'bin':binary_tree, 'long':long_tree_structures, 'star': stars, 'random_bin': random_binary_tree}

def add_reconstruct_args(parser):
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--format', type=str, default='covariance')
    parser.add_argument('--random_method', type=str, default='uniform')
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--tree_type', type=str, default='')
    parser.add_argument('--random_fo', type=int, default=0)
    parser.add_argument('--constant', type=int, default=0)

def reconstruct_file_tag(args):
    return "{}-{}-{}-{}-{}-{}-{}.txt".format(
                args.num_trials, args.format, args.random_method, args.normalize, args.tree_type, args.random_fo, args.constant)

def reconstruct_description(args):
    return "trials: {}; random method: {}, tree type: {},\n is normalized?: {}, is fo?: {}, is constant?: {}".format(
                args.num_trials, args.random_method, args.tree_type,
                args.normalize, args.random_fo, args.constant)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute_reconstruct', type=int, default=1)
    parser.add_argument('--recompute_sample', type=int, default=0)
    parser.add_argument('--start', type=int, default=3)
    parser.add_argument('--end', type=int, default=5)
    parser.add_argument('--estimators', nargs='+', type=str, 
        default=['us', 'empcov', 
        'lineartree', 'shrink', 'lineartreezero', 
        'invalidshrink', 'mxshrink'])
    add_reconstruct_args(parser)
    args = parser.parse_args()

    tree_method = TREE_TYPES[args.tree_type]

    sampled = set()

    for estimator in args.estimators:
        for i in range(args.start, args.end):
            if not pow_of_2(i):
                continue
            print("Running", i)
            istr = '%05d' % i
            structure = '-'.join(map(str, tree_method(i)[-1]))
            sample_out_file="sample_series-{}-{}-{}-{}-{}-{}-{}.txt".format(istr, args.num_trials, 
                args.random_method, args.normalize, args.tree_type, args.random_fo, args.constant)
            if sample_out_file not in sampled and (args.recompute_sample == 1 or (not os.path.exists(sample_out_file))):
                print("Not found. Sampling...")
                cmd = "python3 scratch/sample.py --num_trials {} --structure {} --out_file {} --random_method {} --random_fo {} --normalize {} --constant {}".format(
                    args.num_trials, structure, sample_out_file, args.random_method, args.random_fo, args.normalize, args.constant
                )
                print(cmd)
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print("Done...")
                sampled.add(sample_out_file)

            reconstruct_out_file = "reconstruct_series-{}-{}-{}".format(
                istr, estimator, reconstruct_file_tag(args))
            if args.recompute_reconstruct == 1 or (not os.path.exists(reconstruct_out_file)):
                cmd = "python3 scratch/reconstruct.py --format {} --in_file {} --out_file {} --estimator {}".format(
                    args.format, sample_out_file, reconstruct_out_file, estimator
                )
                print(cmd)
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()