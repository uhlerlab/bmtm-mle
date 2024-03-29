'''
Reconstruct covariance matrix given samples
'''
import argparse
import random
import math
from itertools import chain

import numpy as np
from numpy.lib.arraysetops import intersect1d
import cvxpy as cp

from scratch.tree import Tree, gaussian_likelihood
from scratch.solver import Solver, var_prediction_star_novel
from scratch.runtime_comparison import long_tree_with_root
from scratch.util import fr_norm_sq_one, max_var, operator_norm

def compute_ls(data, guess_arr):
    tree = Tree()
    tree.make_prefix(guess_arr)
    tree.set_data(data)

    _, sample_cov, _ = tree._build_matrices()
    sample_cov = np.array(sample_cov)

    # Problem data.
    p = tree.num_leaf_nodes()
    np.random.seed(1)

    # Construct the problem.
    A = cp.Variable((p, p))
    objective = cp.Minimize(cp.sum_squares(A - sample_cov))
    constraints = [A[i][j] - A[j][i] == 0 for i in range(p) for j in range(i+1, p)]
    #constraints.extend([A[i][j] >= 0 for i in range(p) for j in range(i+1, p)])
    #constraints = [A >> 0]
    #constraints.append(A >> 0)
    constraints.append(A >= 0)

    eqfunnel = {}
    for i, a in enumerate(tree.leaves()):
        for j, b in enumerate(tree.leaves()):
            if j >= i:
                continue
            parent = a.lca(b)
            if parent.num_leaf_nodes() > 2:
                eqfunnel[parent] = (i, j)

    for i, a in enumerate(tree.leaves()):
        for j, b in enumerate(tree.leaves()):
            if j >= i:
                continue
            parent = a.lca(b)
            if parent in eqfunnel:
                k, l = eqfunnel[parent]
                if (k, l) != (i, j):
                    print('eq', (k, l), (i, j))
                    constraints.append(A[i][j] - A[k][l] == 0)

    for i, a in enumerate(tree.leaves()):
        a.ind = i
    
    def m_ind(a):
        if len(a.children) == 0:
            return a.ind, a.ind
        else:
            return a.children[0].get_leaf(0).ind, a.children[1].get_leaf(0).ind
    for a in tree.nodes():
        if a.parent is not None:
            ac1, ac2 = m_ind(a)
            pc1, pc2 = m_ind(parent)
            constraints.append(A[pc1][pc2] - A[ac1][ac2] <= 0)

    '''done = set()
    for i, a in enumerate(tree.leaves()):
        for j, b in enumerate(tree.leaves()):
            for k, c in enumerate(tree.leaves()):
                for l, d in enumerate(tree.leaves()):
                    n1 = a.lca(b)
                    n2 = c.lca(d)
                    nparent = n1.lca(n2)

                    if n1 != n2 and (n1, n2) not in done and (n2, n1) not in done:
                        if nparent == n1:
                            print(n1.hash() == n2.hash(), a.hash() == b.hash())
                            print(n1.hash(), n2.hash(), a.hash(), b.hash())
                            print('ineq', (i, j), (k, l))
                            constraints.append(A[i][j] - A[k][l] <= 0)
                        elif nparent == n2:
                            print('ineq', (k, l), (i, j))
                            constraints.append(A[i][j] - A[k][l] >= 0)
                        done.add((n1, n2))
    '''
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    #print(constraints[0].dual_value)
    for a in tree.nodes():
        ac1, ac2 = m_ind(a)
        a.tv = A.value[ac1][ac2]
    for a in tree.nodes():
        a.above_var = a.tv - (a.parent.tv if a.parent is not None else 0)
    
    return tree

def eigshrink(mle):
    othereigs = np.linalg.eigvals(np.array(compute_linear_tree(data, 0)))
    othermu = sum(othereigs)/len(othereigs)

    eigs, vectors = np.linalg.eig(mle)
    mu = sum(eigs)/len(eigs)
    
    def disp(m, e):
        return sum(abs(a-m)**2 for a in e)
    
    coef = math.sqrt(disp(othermu, othereigs)/(disp(mu, eigs)))

    newdiag = np.diag([mu + (eigs[i]-mu)*coef for i in range(len(eigs))])
    return [[float(a) for a in b] for b in vectors @ newdiag @ np.transpose(vectors)]

def mxshrink(mle):
    values, vectors = np.linalg.eig(mle)
    other = vectors @ np.diag(values) @ np.transpose(vectors)
    assert(np.allclose(mle, other))
    
    p = mle.shape[0]
    n = (1 if p % 2 == 1 else 1.5)
    #print(n, p, len(data))
    newdiag = np.diag([(n/(n+p+1-2*i))*values[i] for i in range(len(values))])
    #print('values', values, newdiag)
    #newdiag = np.diag([(n/(n+p+1-2*i))*values[i] for i in range(len(values))])
    return [[float(a) for a in b] for b in vectors @ newdiag @ np.transpose(vectors)]

def make_cluster_tree(cl):
    if type(cl[0]) != tuple:
        t = Tree()
        t.data = cl[0]
        return t

    t = Tree()
    for c in cl: 
        t.children.append(make_cluster_tree(c))
        t.children[-1].parent = t
    return t

def make_cluster_var(cl, dists):
    if type(cl[0]) != tuple:
        return []

    vars = []
    for c in cl: 
        vars.append(dists[(c, cl)])
        vars.extend(make_cluster_var(c, dists))
    return vars

def neighbor_joining(data):
    clusters = [(d,) for d in data]
    dists = {(a, b):abs(a[0]-b[0]) for a in clusters for b in clusters}
    def ad(piv):
        return sum(dists[(piv, k)] for k in clusters)
    def n():
        return len(clusters) + 1
    while len(clusters) > 1:
        qdists = {(a, b):((n()-2)*dists[(a, b)]
            - ad(a) - ad(b))
        for a in clusters for b in clusters}
        mv, indi, indj = min((qdists[clusters[a], clusters[b]], a, b) 
            for a in range(len(clusters)) 
                for b in range(len(clusters)) if a != b)

        ci = clusters[indi]
        cj = clusters[indj]

        nc = (ci,cj)
        # check cluster len
        dists[(ci, nc)] = ((1/2)*dists[(ci, cj)] 
            + (1/(2*(n()-2)))*(ad(ci) - ad(cj)))
        dists[(cj, nc)] = dists[(ci, cj)] - dists[(ci, nc)]
        dists[(nc, ci)] = dists[(ci, nc)]
        dists[(nc, cj)] = dists[(cj, nc)]

        clusters.remove(ci)
        clusters.remove(cj)

        for a in clusters:
            nd = (dists[(ci, a)]+dists[(cj, a)] - dists[(ci, cj)])/2
            dists[(nc, a)] = nd 
            dists[(a, nc)] = nd 

        dists[(nc, nc)] = 0
        clusters.append(nc) 

    nj_tree = make_cluster_tree(clusters[0])
    vars = [0]+make_cluster_var(clusters[0], dists)
    nj_tree.set_var([v**2 for v in vars])
    #nj_tree.set_var(vars)
    return nj_tree

def upgma(data):
    clusters = [(d,) for d in data]
    dists = {(a, b):abs(a[0]-b[0]) for a in clusters for b in clusters}
    while len(clusters) > 1:
        mv, indi, indj = min((dists[clusters[a], clusters[b]], a, b) 
            for a in range(len(clusters)) 
                for b in range(len(clusters)) if a != b)

        ci = clusters[indi]
        cj = clusters[indj]

        nc = (ci,cj)
        for a in clusters:
            nd = (dists[(ci, a)]+dists[(cj, a)])/2
            dists[(nc, a)] = nd 
            dists[(a, nc)] = nd 
        dists[(nc, nc)] = 0

        clusters.remove(ci)
        clusters.remove(cj)
        clusters.append(nc) 
    upgma_tree = make_cluster_tree(clusters[0])
    vars = [0]+make_cluster_var(clusters[0], dists)
    #upgma_tree.set_var([v for v in vars])
    upgma_tree.set_var(vars)
    return upgma_tree
 
def sample_cov(data):
    return [[a*b for b in data] for a in data]

def compute_zero_linear_tree(data):
    pos = sorted([d for d in data + [0] if d >= 0])
    neg = list(reversed(sorted([d for d in data + [0] if d <= 0])))
    pos_diff = [(pos[i]-pos[i+1])**2 for i in range(len(pos)-1)]
    neg_diff = [(neg[i]-neg[i+1])**2 for i in range(len(neg)-1)]


    tree = Tree()
    structure = long_tree_with_root(len(neg)-1) + long_tree_with_root(len(pos) - 1)
    tree.make_prefix(structure)
    fmt_data = [p for p in neg if p != 0]+[p for p in pos if p != 0]
    tree.set_data(fmt_data)
    def intersperse(diff):
        if len(diff) == 0:
            return []
        return list(map(float, (',0,'.join(map(str, diff))).split(',')))
    var = [0]+intersperse(neg_diff) + intersperse(pos_diff)
    tree.set_var(var)

    return tree

def compute_linear_tree(data, pivot, order=None):
    pos = sorted([d for d in data + [0] if d >= pivot])
    neg = list(reversed(sorted([pivot]+[d for d in data + [0] if d < pivot])))
    pos_diff = [(pos[i]-pos[i+1])**2 for i in range(len(pos)-1)]
    neg_diff = [(neg[i]-neg[i+1])**2 for i in range(len(neg)-1)]
    def get_tv(i, j):
        if (data[i] >= pivot) != (data[j] >= pivot):
            return 0
        arr, arr_diff = neg, neg_diff
        if data[i] >= pivot:
            arr, arr_diff = pos, pos_diff
            
        indi = arr.index(data[i])
        indj = arr.index(data[j])
        return sum(arr_diff[:min(indi, indj)])
    
    if order is None:
        order = range(len(data))
    return [[get_tv(i, j) for j in order] for i in order]

def our_mle(data, guess_ar):
    reconstruct_tree = Tree()
    reconstruct_tree.make_prefix(guess_arr)
    reconstruct_tree.set_data(data)

    solver = Solver()
    solver.predict_mle(reconstruct_tree)

    return reconstruct_tree

def fill_fo(ot):
    def fill_up(leave):
        if leave.above_var == 0:
            assert(leave.parent is not None)
            assert(leave.parent.data is None)
            leave.parent.data = leave.data
            fill_up(leave.parent)
    def fill_down(leave):
        if leave.above_var == 0:
            assert(leave.data is None)
            leave.data = 0
            for c in leave.children:
                fill_down(leave)

    for l in ot.leaves():
        fill_up(l)
    fill_down(ot)
    for l in ot.nodes():
        assert(l.data is not None)

def foknowstruct(data, guess_arr, gt_vars):
    ot = Tree()
    ot.make_prefix(guess_arr)
    ot.set_var(gt_vars)
    ot.set_data(data)

    fill_fo(ot)

    def assign(node):
        if node.above_var != 0:
            node.above_var = (node.data - (0 if node.parent is None else node.parent.data))**2
        
        for c in node.children:
            assign(c)
    
    assign(ot)
    return ot

def inner(a, b):
    return np.trace(np.matmul(a, np.transpose(b)))/a.shape[0]
def diff(a, b):
    d = a - b
    return inner(d, d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', type=str, default='us')
    parser.add_argument('--format', type=str, default='covariance')
    parser.add_argument('--in_file', type=str, default='samples.txt')
    parser.add_argument('--out_file', type=str, default='reconstructs.txt')
    args = parser.parse_args()

    def format_tree(t, data_to_label):
        if args.format == 'newick':
            t.set_labels(list(range(t.num_leaf_nodes())))
            return t.sparse_newick()
        elif args.format == 'covariance':
            return t.cov_matrix()
        elif args.format == 'variances':
            return t.get_var()
        elif args.format == 'tree':
            datas = t.get_data()
            print([data_to_label[d] for d in datas])
            return (t.get_prefix(), t.get_var(), datas, 
                [data_to_label[d] for d in datas])

    ground_truths, guesses = [], []

    with open(args.in_file, 'r') as f:
        lines = f.read().strip().split('\n')

        num_trials = int(lines.pop(0))

        assert(len(lines) == 2*num_trials)

        for i in range(num_trials):
            guess_arr = list(map(int, lines[i].split('\t')[0].split('-')))

            data = list(map(float, lines[num_trials+i].split(',')))

            ground_truth_tree = Tree()
            ground_truth_tree.make_prefix(guess_arr)
            ground_truth_tree.set_var(list(map(float, lines[i].split('\t')[1].split(','))))
            ground_truth_tree.set_data(data)
            data_to_label = {d:i for i, d in enumerate(data)}
            ground_truths.append(format_tree(ground_truth_tree, data_to_label))

            num_leaf_nodes = ground_truth_tree.num_leaf_nodes()

            if args.estimator == 'gt':
                guesses.append(ground_truths[-1])
            elif args.estimator == 'us':
                guesses.append(format_tree(our_mle(data, guess_arr), data_to_label))
            elif args.estimator == 'correctedges':
                ot = our_mle(data, guess_arr)
                ot.set_var([1 if ((a != 0) and (b != 0)) else 0 
                    for a, b in zip(ot.get_var(), ground_truth_tree.get_var())])
                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'foshrink2':
                ot = our_mle(data, guess_arr)
                ot.set_var([(1/3.2)*a if a != 0 else a for a in ot.get_var()])
                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'foshrink':
                ot = our_mle(data, guess_arr)
                ot.set_var([(1/3)*a if a != 0 else a for a in ot.get_var()])
                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'foknowstruct':
                ot = foknowstruct(data, guess_arr, ground_truth_tree.get_var())
                ot.set_var([(1/3)*a if a != 0 else a for a in ot.get_var()])
                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'foknowstructoracle':
                ot = foknowstruct(data, guess_arr, ground_truth_tree.get_var())
                ot.set_var([(b**2/(2*b**2 + b))*a if b != 0 else a for a, b in zip(ot.get_var(), ground_truth_tree.get_var())])
                assert(all((a == 0) == (b == 0) for a, b in zip(ot.get_var(), ground_truth_tree.get_var())))

                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'upgma':
                guesses.append(format_tree(upgma(data), data_to_label))
            elif args.estimator == 'nj':
                guesses.append(format_tree(neighbor_joining(data), data_to_label))
            elif args.estimator == 'indep':
                indep_tree = Tree()
                indep_tree.make_prefix([0]*num_leaf_nodes)
                indep_tree.set_data(data)
                indep_tree.set_var([0]+[a**2 for a in data])
                guesses.append(format_tree(indep_tree, data_to_label))

            elif args.estimator == 'ridge':
                nt = Tree()
                nt.make_prefix(guess_arr)
                nt.set_data(data)
                nt.mle(method='differential_evolution', 
                    max_var=max(10, max_var(data)), 
                    accept=-1000, maxiter=5000, lam=1)
                print(nt.likelihood())
                
                guesses.append(format_tree(nt, data_to_label))
            elif args.estimator == 'lowersumsq':
                nt = Tree()
                nt.make_prefix(guess_arr)
                nt.set_data(data)
                nt.set_var(var_prediction_star_novel(data))
                guesses.append(format_tree(nt, data_to_label))

            elif args.estimator == 'lowersumsqshrink':
                nt = Tree()
                nt.make_prefix(guess_arr)
                nt.set_data(data)
                nt.set_var(var_prediction_star_novel(data))

                mle = np.array(nt.cov_matrix())
                identity = (1/math.sqrt(num_leaf_nodes))*np.identity(mle.shape[0])
                norm = fr_norm_sq_one(compute_linear_tree(data, 0))
                coef = math.sqrt(norm/fr_norm_sq_one(mle))
                nt.set_var([coef*a for a in nt.get_var()])
                guesses.append(format_tree(nt, data_to_label))
            elif args.estimator == 'ussqrt':
                ot = our_mle(data, guess_arr)
                ot.set_var([math.sqrt(a) for a in ot.get_var()])
                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'shrink':
                ot = our_mle(data, guess_arr)
                mle = np.array(ot.cov_matrix())
                identity = (1/math.sqrt(num_leaf_nodes))*np.identity(mle.shape[0])

                norm = fr_norm_sq_one(compute_linear_tree(data, 0))

                #final = (1/(1+4))*(mle + 4*identity)
                coef = math.sqrt(norm/fr_norm_sq_one(mle))
                ot.set_var([coef*a for a in ot.get_var()])
                assert(abs(fr_norm_sq_one(ot.cov_matrix())-norm) < 1e-6)

                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'shrink2':
                ot = our_mle(data, guess_arr)
                lt = compute_zero_linear_tree(data)
                def l2(vec):
                    return math.sqrt(sum(v**2 for v in vec))
                coef = l2(lt.get_var()) / l2(ot.get_var())
                ot.set_var([coef*a for a in ot.get_var()])
                assert(abs(l2(lt.get_var()) - l2(ot.get_var())) < 1e-7)

                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'varshrink':
                mle_tree = our_mle(data, guess_arr)
                vars = mle_tree.get_var()
                mu = sum(vars)/len(vars)
                p = mle_tree.num_leaf_nodes()
                coef = p/(p+1)
                newvars = [mu + (v-mu)*coef for v in vars]
                mle_tree.set_var(newvars)
                guesses.append(mle_tree.cov_matrix())
            elif args.estimator == 'empcov':
                s_arr = sample_cov(data) 
                guesses.append(s_arr)
            elif args.estimator == 'frbmtm':
                reconstruct_tree = Tree()
                reconstruct_tree.make_prefix(guess_arr)
                reconstruct_tree.set_data(data)
                reconstruct_tree.fr_proj()
                guesses.append(reconstruct_tree.cov_matrix())
            elif args.estimator == 'ls':
                guesses.append(format_tree(compute_ls(data, guess_arr), data_to_label))

            elif args.estimator == 'lineartree':
                guesses.append(compute_linear_tree(data, min(data+[0])))
            elif args.estimator == 'lineartreezero':
    
                new_tree = compute_zero_linear_tree(data)
                new_data = new_tree.get_data()
                order = [data.index(a) for a in new_data]
                old_matrix = compute_linear_tree(data, 0, order=order)

                assert(old_matrix == new_tree.cov_matrix())
                assert(compute_linear_tree(data, 0, order=list(reversed(order)))
                    != new_tree.cov_matrix())

                guesses.append(format_tree(new_tree, data_to_label))
                #print("cov matrix", guesses[-1])

            elif args.estimator == 'zero':
                zero_tree = Tree()
                zero_tree.make_prefix(guess_arr)
                zero_tree.set_data(data)
                zero_tree.set_var([0]*zero_tree.num())
                guesses.append(format_tree(zero_tree, data_to_label))

            elif args.estimator == 'zeroshrink':
                our_tree = our_mle(data, guess_arr)
                new_vars = [math.log(1+v) for v in our_tree.get_var()]
                our_tree.set_var(new_vars)
                guesses.append(format_tree(our_tree, data_to_label))

            elif args.estimator == 'ledoitwolfvalidshrink':
                mle_tree = our_mle(data, guess_arr)
                mle_matrix = np.array(mle_tree.cov_matrix())
                #sample_cov_matrix = np.array(sample_cov(data))
                identity_matrix = np.identity(mle_matrix.shape[0])
                ground_truth_matrix = np.array(ground_truth_tree.cov_matrix())

                def diff(a, b):
                    d = a - b
                    return np.trace(np.matmul(d, np.transpose(d)))
                
                mu = np.trace(ground_truth_matrix)/mle_matrix.shape[0]
                a_sq = diff(ground_truth_matrix, mu*identity_matrix)
                b_sq = 0
                trials = 20
                for i in range(trials):
                    b_sq += diff(ground_truth_matrix, 
                        np.array(our_mle(ground_truth_tree.sample_data(), guess_arr).cov_matrix()))
                b_sq /= trials
                delta_sq = a_sq + b_sq
                
                #final_matrix = (b_sq/delta_sq)*mu*identity_matrix + (a_sq/delta_sq)*mle_matrix
                #guesses.append(list(final_matrix))
                #guesses.append(list((4/5)*mu*identity + (1/5)*mle))
                for l in mle_tree.nodes():
                    l.above_var = (a_sq/delta_sq)*l.above_var
                for l in mle_tree.leaves():
                    l.above_var += (b_sq/delta_sq)*mu
                guesses.append(format_tree(mle_tree, data_to_label))
            
            elif args.estimator == 'invalidshrink':
                #sigma = np.array(ground_truth_tree.cov_matrix())
                mle = np.array(our_mle(data, guess_arr).cov_matrix())
                mle = np.array(sample_cov(data))
                identity = np.identity(mle.shape[0])

                def inner(a, b):
                    return np.trace(np.matmul(a, np.transpose(b)))/a.shape[0]
                def diff(a, b):
                    d = a - b
                    return inner(d, d)
                
                mu = inner(mle, identity)
                alpha_sq = diff(sigma, mu*identity)
                #beta_sq = diff(sigma, mle)
                #delta_sq = diff (mle, mu*identity)

                print(alpha_sq, beta_sq, delta_sq, alpha_sq+beta_sq)

                guesses.append(list((beta_sq/delta_sq)*mu*identity + (alpha_sq/delta_sq)*mle))
                #guesses.append(list((4/5)*mu*identity + (1/5)*mle))

            elif args.estimator == 'eigshrink':
                mle = np.array(our_mle(data, guess_arr).cov_matrix())
                guesses.append(eigshrink(mle))
            elif args.estimator == 'random_fo':
                fo_seed = [random.randrange(0, 20) for _ in range(ground_truth_tree.num())]
                ot = Tree()
                ot.make_prefix(ground_truth_tree.get_prefix())
                ot.set_var([1 for _ in range(ground_truth_tree.num())])
                ot.random_fo(fo_seed)
                ot.set_data(data)
                guesses.append(format_tree(ot, data_to_label))
            elif args.estimator == 'eigshrinkempcov':
                mle = np.array(sample_cov(data))
                guesses.append(eigshrink(mle))
            elif args.estimator == 'mxshrink':
                mle = np.array(our_mle(data, guess_arr).cov_matrix())
                guesses.append(mxshrink(mle))
            elif args.estimator == 'mxshrinkempcov':
                mle = np.array(sample_cov(data))
                guesses.append(mxshrink(mle))
            elif args.estimator == 'fralltree':
                pass
            else:
                print(args.estimator)
                print(args.estimator == 'us')
                raise ValueError('No match in estimator!')

    assert(len(ground_truths) == len(guesses))

    with open(args.out_file, 'w') as f:
        out = None
        if args.format == 'newick':
            out = '\n'.join(chain(ground_truths, guesses))
        elif args.format == 'covariance':
            lines = [str(ground_truth_tree.num_leaf_nodes())]
            for cov_matrix in chain(ground_truths, guesses):
                lines.append(','.join(','.join(map(str, l)) for l in cov_matrix))
            out = '\n'.join(lines)
        elif args.format == 'variances':
            lines = [str(ground_truth_tree.num_leaf_nodes()), '-'.join(map(str, guess_arr))]
            for variances in chain(ground_truths, guesses):
                lines.append(','.join(map(str, variances)))
            out = '\n'.join(lines)
        elif args.format == 'tree':
            lines = []
            for tup in chain(ground_truths, guesses):
                assert(type(tup) == tuple)
                lines.append('\t'.join([','.join(map(str, a)) for a in tup]))
            out = '\n'.join(lines)

        f.write(args.format+'\n'+out)