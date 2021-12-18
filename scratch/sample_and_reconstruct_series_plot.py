'''
Plot the results of sample and reconstruct over multiple files
'''
import argparse
import random
import math
import glob

import matplotlib.pyplot as plt
import numpy as np

from scratch.util import fr_norm_sq, fr_norm_sq_one
from scratch.tree import bhv_distance_owens, Tree, in_tree, bhv_distance_owens_list, rf
from scratch.reconstruct import diff
from scratch.solver import Solver, var_prediction_star, var_prediction_star_novel

f1, f2 = plt.rcParamsDefault["figure.figsize"]
scale = 1.5
plt.rcParams["figure.figsize"] = (f1*scale, f2*scale) 

def format_errors(differences, mean=False, median=False):
    differences.sort()
    if mean:
        avg = sum(differences)/len(differences)
        std = sum(abs(d - avg) for d in differences)/len(differences)
        #return (avg, std, std)
    if median:
        avg = sorted(differences)[len(differences)//2]
        low = avg - differences[len(differences)//10]
        up = differences[(9*len(differences))//10] - avg
        return (avg, low, up)
    
    avg = sum(differences)/len(differences)
    low = avg - differences[len(differences)//10]
    up = differences[(9*len(differences))//10] - avg
    return (avg, low, up)

if __name__ == '__main__':

    plt.rcParams.update({'font.size': 14})

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file_wildcards', nargs='+', type=str)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--display', type=str, default='risk')
    parser.add_argument('--desc', type=int, default=0)
    parser.add_argument('--y_log', type=int, default=0)
    parser.add_argument('--jitter', type=float, default=0.2)
    parser.add_argument('--errorbar', type=int, default=1)
    args = parser.parse_args()

    fmt = None

    def run(wildcard):
        global fmt
        expectations = []
        num_childrens = []
        print('wild', wildcard)
        for i, fn in enumerate(sorted(glob.glob(wildcard))):
            with open(fn, 'r') as f:
                lines = f.read().strip().split('\n')
                fmt = lines.pop(0)
                if fmt == 'newick':
                    raise ValueError('not supported right now')
                    num_childrens.append(i)
                    ground_truth = lines.pop(0)
                    expectations.append(sum(int(l == ground_truth) for l in lines)/len(lines))
                elif fmt == 'tree':
                    assert(len(lines) % 2 == 0)
                    num_trials = len(lines)//2

                    num_childrens.append(in_tree(lines[0]).num_leaf_nodes())

                    if args.display == 'risk':
                        #differences = [bhv_distance_owens(in_tree(lines[ind]), in_tree(lines[ind+num_trials]))
                        #    for ind in range(num_trials)]
                        ns = []
                        for ind in range(len(lines)):
                            ns.append(in_tree(lines[ind]).newick())
                        differences = bhv_distance_owens_list(ns)
                    elif args.display == 'guess_norm':
                        def l2_norm(vec):
                            return sum(a**2 for a in vec)
                        differences = [l2_norm(in_tree(lines[num_trials+ind]).get_var())/(2*num_childrens[-1]-1)
                            for ind in range(num_trials)]
                    elif args.display == 'guess_variance':
                        def l2_norm(v1, v2):
                            return sum((a - b)**2 for a, b in zip(v1, v2))
                        sumt = [0 for _ in in_tree(lines[num_trials]).get_var()]
                        for ind in range(num_trials):
                            sumt = [s + a for s, a in zip(sumt, in_tree(lines[num_trials+ind]).get_var())]
                        sumt = [s/num_trials for s in sumt]

                        differences = [sum(l2_norm(sumt, in_tree(lines[num_trials+ind]).get_var()) for ind in range(num_trials)) / num_trials]
                    elif args.display == 'guess_bias':
                        sumt = [0 for _ in in_tree(lines[num_trials]).get_var()]
                        for ind in range(num_trials):
                            sumt = [s + a for s, a in zip(sumt, in_tree(lines[num_trials+ind]).get_var())]
                        sumt = [s/num_trials for s in sumt]

                        differences = [sum(abs(a-b) for a, b in zip(in_tree(lines[0]).get_var(), sumt))]
                    elif args.display == 'rf_distance':
                        differences = [rf(in_tree(lines[ind]), in_tree(lines[num_trials+ind])) for ind in range(num_trials)]
                    else:
                        raise ValueError('Could not find display')

                    worst_diff, worst_i = max((dif, i) for i, dif in enumerate(differences))
                    gt = in_tree(lines[worst_i])
                    guess = in_tree(lines[num_trials+worst_i])
                    nt = Tree()
                    nt.make_prefix(gt.get_prefix())
                    nt.set_data(gt.get_data())
                    Solver().predict_mle(nt)

                    expectations.append(format_errors(differences))
                elif fmt == 'covariance':
                    num_children = int(lines.pop(0))
                    num_childrens.append(num_children)
                    def in_matrix(l):
                        symbols = l.split(',')
                        assert(len(symbols) == num_children**2)
                        return [list(map(float, symbols[i:i+num_children])) for i in range(0, len(symbols), num_children)]
                    
                    assert(len(lines) % 2 == 0)
                    num_trials = len(lines)//2
                    if args.display == 'risk':
                        differences = [fr_norm_sq(in_matrix(lines[ind]), in_matrix(lines[num_trials+ind])) 
                            for ind in range(num_trials)]
                    elif args.display == 'guess_norm':
                        differences = [fr_norm_sq_one(in_matrix(lines[num_trials+ind])) 
                            for ind in range(num_trials)]
                    elif args.display == 'guess_eig':
                        differences = [np.var(np.linalg.eigvals(np.array(in_matrix(lines[num_trials+ind]))))
                            for ind in range(num_trials)]
                    elif args.display == 'true_norm':
                        differences = [fr_norm_sq_one(in_matrix(lines[ind])) 
                            for ind in range(num_trials)]
                    elif args.display == 'root_value':
                        differences = [in_matrix(lines[num_trials+ind])[0][1]
                            for ind in range(num_trials)]
                    elif args.display == 'first_value':
                        differences = [in_matrix(lines[num_trials+ind])[1][1]
                            for ind in range(num_trials)]
                    elif args.display == 'guess_variance':
                        mean_matrix = in_matrix(lines[num_trials+0])
                        for ind in range(1, num_trials):
                            m = in_matrix(lines[num_trials+ind])
                            mean_matrix = [[mean_matrix[ii][jj] + m[ii][jj] for jj in range(len(m[0]))] 
                                for ii in range(len(m))]
                        mean_matrix = [[mean_matrix[ii][jj]/num_trials for jj in range(len(mean_matrix[0]))] 
                            for ii in range(len(mean_matrix))]
                        differences = [fr_norm_sq(in_matrix(lines[num_trials+ind]), mean_matrix) 
                            for ind in range(num_trials)]
                    elif args.display == 'guess_bias':
                        mean_matrix = in_matrix(lines[num_trials+0])
                        for ind in range(1, num_trials):
                            m = in_matrix(lines[num_trials+ind])
                            mean_matrix = [[mean_matrix[ii][jj] + m[ii][jj] for jj in range(len(m[0]))] 
                                for ii in range(len(m))]
                        mean_matrix = [[mean_matrix[ii][jj]/num_trials for jj in range(len(mean_matrix[0]))] 
                            for ii in range(len(mean_matrix))]
                        differences = [fr_norm_sq(in_matrix(lines[ind]), mean_matrix) 
                            for ind in range(num_trials)]

                    _, worst_i = max((dif, i) for i, dif in enumerate(differences))

                    expectations.append(format_errors(differences))
        return num_childrens, expectations

    SUB = ['us', 'lineartreezero', 'shrink', 'foshrink', 'upgma', 'nj', 'ls', 'ledoitwolfvalidshrink', 'mxshrink']
    SUB = ['-'+s+'-' for s in SUB]

    x_values = set()
    for i, w in enumerate(sorted(args.in_file_wildcards, 
        key=lambda x: SUB.index(next(s for s in SUB if s in x)) if any(s in x for s in SUB) else x)):
        print(w)
        num_childrens, expectations = run(w) 
        mid, low, up = zip(*expectations)
        print('---hey---')
        print('outs', num_childrens, mid, low, up)
        print('-----')
        #with open(args.out_file, 'w') as f:
        #    f.write('\n'.join(map(str, expectations)))
        label = w.split('-')[2]
        LABELS = {'us': 'bmtm mle', 'lineartreezero': 'ddgm mle', 
            'shrink': 'shrink-to-ddm', 'foshrink': 'one-third-shrink',
            'nj': 'neighbor-joining', 'ls': 'least-squares', 'ledoitwolfvalidshrink': 'linear-shrink'}
        if label in LABELS:
            label = LABELS[label]

        num_childrens = [round(math.log(a)/math.log(2)) for a in num_childrens]
        x_values.update(num_childrens)
        newx = [args.jitter*(2*(i/(len(args.in_file_wildcards)-1))-1)+a for a in num_childrens]
        if len(num_childrens) != 0:
            if args.errorbar == 1:
                plt.errorbar(newx, 
                    mid, yerr=[low, up], 
                    label=label, capsize=10, fmt='o')
            else:
                plt.errorbar(newx, 
                    mid, yerr=[low, up], 
                    label=label, capsize=0, fmt='o')
    #plt.plot(list(range(len(expectations))), [(1/2)*(i+1)**(-1/2) for i in range(len(expectations))])
    #plt.legend(loc="upper left")
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #      ncol=3, fancybox=True, shadow=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    plt.title(args.title)
    if args.desc:
        plt.figtext(.5, .9, args.description, wrap=True, ha='center', fontsize=7)
    plt.xlabel('# of leaves')
    risk = '(' + ('frobenius squared)' if fmt != 'tree' else 'bhv l2)')
    DISPLAY_LABELS = {'guess_bias': 'estimator bias',
        'guess_variance': 'estimator variance',
        'guess_norm': 'estimator norm',
        'risk': 'estimator risk'}
    plt.ylabel('{} {}'.format(DISPLAY_LABELS.get(args.display, args.display), risk))
    if args.y_log:
        plt.yscale('log')
        #plt.ylabel(plt.gca().yaxis.get_label().get_text())

    x_val_list = list(sorted(x_values))
    plt.xticks(x_val_list, list(map(str, [2**a for a in x_val_list])))
    
    #plt.subplots_adjust(left=0.45, bottom=0.15, right=0.4)

    if args.save_file is None:
        plt.show()
    else:
        plt.savefig(args.save_file, bbox_inches='tight')