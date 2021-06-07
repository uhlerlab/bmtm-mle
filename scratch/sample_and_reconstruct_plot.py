
'''
Plot the results of sample and reconstruct
'''
import argparse
import math

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='samples.txt')
    args = parser.parse_args()

    with open(args.in_file, 'r') as f:
        lines = f.read().strip().split('\n')
        fmt = lines.pop(0)
        if fmt == 'covariance':
            num_children = int(lines.pop(0))
            def in_matrix(l):
                symbols = l.split(',')
                assert(len(symbols) == num_children**2)
                return [list(map(float, symbols[i:i+num_children])) for i in range(0, len(symbols), num_children)]
            def lin(m):
                return [b for a in m for b in a]
            def fr_norm(one, two):
                return math.sqrt(sum((a-b)**2 for a, b in zip(lin(one), lin(two))))
            
            ground_truth = in_matrix(lines.pop(0))
            reconstruct = [in_matrix(l) for l in lines]
            differences = [fr_norm(ground_truth, a) for a in reconstruct] 

            plt.hist(differences, bins=40)
            plt.show()
        elif fmt == 'newick':
            ground_truth = lines.pop(0)
            weights = [1/float(len(lines))]*len(lines)
            plt.hist(lines, bins=2*len(set(lines)), weights=weights)
            plt.title('ground truth: {}'.format(ground_truth))
            plt.show()