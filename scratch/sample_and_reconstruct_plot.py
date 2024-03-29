
'''
Plot the results of sample and reconstruct
'''
import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from scratch.util import fr_norm_sq_one, fr_norm_sq
from scratch.tree import bhv_distance_owens_list, in_tree, rf
from scratch.solver import Tree

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='samples.txt')
    parser.add_argument('--in_file_2', type=str, default=None)
    parser.add_argument('--file_label', type=str, default='')
    parser.add_argument('--file_label_2', type=str, default='')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--display', type=str, default='risk')
    parser.add_argument('--save_file', type=str, default=None)
    args = parser.parse_args()

    in_files = [a for a in [(args.in_file, args.file_label), (args.in_file_2, args.file_label_2)] if a[0] is not None]
    gl_labels, gl_differences = [], []

    for in_file, file_label in in_files:
        with open(in_file, 'r') as f:
            lines = f.read().strip().split('\n')
            fmt = lines.pop(0)
            if fmt == 'covariance':
                num_children = int(lines.pop(0))
                def in_matrix(l):
                    symbols = l.split(',')
                    assert(len(symbols) == num_children**2)
                    return [list(map(float, symbols[i:i+num_children])) for i in range(0, len(symbols), num_children)]
                
                num_trials = len(lines)//2
                if args.display == 'risk':
                    differences = [fr_norm_sq(in_matrix(lines[ind]), in_matrix(lines[num_trials+ind])) 
                        for ind in range(num_trials)]

                plt.hist(differences, bins=40)
            elif fmt == 'tree':
                assert(len(lines) % 2 == 0)
                num_trials = len(lines)//2

                #ns = []
                #for ind in range(len(lines)):
                #    ns.append(in_tree(lines[ind]).newick())
                #differences = bhv_distance_owens_list(ns)
                differences = [rf(in_tree(lines[ind]), in_tree(lines[num_trials+ind])) for ind in range(num_trials)]
                gl_differences.append(differences)
                gl_labels.append(file_label)

                print(file_label, sum(differences)/len(differences))

                bins = np.linspace(0, 16, 17)
                #plt.hist(differences, bins=bins, alpha=0.8, label=file_label,) #weights=np.ones(len(differences)) / len(differences))
                #plt.title(args.title)
            '''elif fmt == 'variances':
                num_children = int(lines.pop(0))
                guess_arr = list(map(int, lines.pop(0).split('-')))

                num_trials = len(lines)//2
                differences = []
                for ind in range(num_trials):
                    tree = Tree()
                    tree.make_prefix(guess_arr)
                    variances = list(map(float, lines[num_trials+ind].split(',')))
                    tree.set_var(variances)

                    if args.display == 'root_variance':
                        differences.append(tree.above_var)
                    elif args.display == 'other_variance':
                        differences.append(tree.children[0].above_var)
                    elif args.display == 'root_zero':
                        differences.append(int(tree.above_var == 0))
                    elif args.display == 'other_zero':
                        differences.append(int(tree.children[0].above_var == 0))
                    elif args.display == 'all_zero':
                        differences.append(int(any(c.above_var == 0 for c in tree.children) or tree.above_var == 0))
                    elif args.display == 'centroid':
                        if abs(tree.above_var) > 1e-7:
                            differences.append(math.sqrt(tree.above_var))
                            differences.append(-math.sqrt(tree.above_var))

                def p(d):
                    n,x,_ = plt.hist(d, bins=80)
                    bin_centers = 0.5*(x[1:]+x[:-1])
                    return bin_centers, n
                otherd = [(np.random.normal(0, 1)) for ind in range(2*num_trials)] 
                print(sum(differences)/len(differences))
                print(sum(otherd)/len(otherd))
                a, b = p(differences)
                c, d = p(otherd)
                plt.clf()
                plt.plot(a, b)
                #plt.plot(c, d)
            
            elif fmt == 'newick':
                ground_truth = lines.pop(0)
                weights = [1/float(len(lines))]*len(lines)
                plt.hist(lines, bins=2*len(set(lines)), weights=weights)
                plt.title('ground truth: {}'.format(ground_truth))
            '''
    
    #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    if len(gl_differences) > 0:
        print(gl_differences)
        bins = np.linspace(-1, 15, 9)
        print(bins)
        plt.hist(gl_differences, bins=bins, alpha=1.0, label=gl_labels) #weights=np.ones(len(differences)) / len(differences))
        plt.xticks(np.linspace(0, 16, 9))
        plt.title(args.title)

    plt.legend(loc="upper left")
    plt.xlabel('Robinson-Foulds Distance')
    plt.ylabel('PMF')
    if args.save_file is None:
        plt.show()
    else:
        plt.savefig(args.save_file)