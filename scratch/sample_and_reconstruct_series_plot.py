'''
Plot the results of sample and reconstruct over multiple files
'''
import argparse
import math
import glob

import matplotlib.pyplot as plt

from util import fr_norm_sq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file_wildcards', nargs='+', type = str)
    parser.add_argument('--out_file', type=str, default='series.txt')
    parser.add_argument('--title', type=str, default='')
    args = parser.parse_args()

    def run(wildcard):
        expectations = []
        num_childrens = []
        for i, fn in enumerate(sorted(glob.glob(wildcard))):
            with open(fn, 'r') as f:
                lines = f.read().strip().split('\n')
                fmt = lines.pop(0)
                if fmt == 'newick':
                    num_childrens.append(i)
                    ground_truth = lines.pop(0)
                    expectations.append(sum(int(l == ground_truth) for l in lines)/len(lines))
                elif fmt == 'covariance':
                    num_children = int(lines.pop(0))
                    num_childrens.append(num_children)
                    def in_matrix(l):
                        symbols = l.split(',')
                        assert(len(symbols) == num_children**2)
                        return [list(map(float, symbols[i:i+num_children])) for i in range(0, len(symbols), num_children)]
                    
                    ground_truth = in_matrix(lines.pop(0))
                    differences = [fr_norm_sq(ground_truth, in_matrix(l)) for l in lines]
                    expectations.append(sum(differences)/len(differences))
        return num_childrens, expectations

    for w in args.in_file_wildcards:
        print(w)
        num_childrens, expectations = run(w) 
        with open(args.out_file, 'w') as f:
            f.write('\n'.join(map(str, expectations)))
        plt.plot(num_childrens, expectations, label=w)
    #plt.plot(list(range(len(expectations))), [(1/2)*(i+1)**(-1/2) for i in range(len(expectations))])
    plt.legend(loc="upper left")
    plt.title(args.title)
    plt.show()