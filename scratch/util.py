import numpy as np

'''Utility functions for the command line scripts'''
def max_var(data):
    new_l = list(data) + [0]
    print('newl', new_l)
    return 3*(max(abs(a-b)**2 for a in new_l for b in new_l)**2)

def lin(m):
    return [b for a in m for b in a]

def fr_norm_sq(one, two):
    return sum((a-b)**2 for a, b in zip(lin(one), lin(two)))
    
def fr_norm_sq_one(one):
    return sum((a)**2 for a in lin(one))

def operator_norm(arr):
    return np.linalg.norm(np.array(arr), 2)

def is_data_invalid(data):
    return any(abs(a-b) < .005 for i, a in enumerate(data) for b in data[i+1:]) or any(abs(a) < .005 for a in data)

def fat_tree_structures(levels):
    trees = [[0, 0]]
    for i in range(levels):
        half = [len(trees[-1])] + trees[-1]
        trees.append(half + half)
    return trees

def stars(max_nodes):
    trees = []
    for i in range(2, max_nodes+1):
        trees.append([0]*i)
    return trees

def long_tree_structures(max_nodes):
    if max_nodes == 0:
        return [[]]
    if max_nodes == 1:
        return [[0]]
    trees = [[0, 0]]
    for i in range(max_nodes-2):
        trees.append([0, (i*2)+2] + trees[-1])
    return trees

def long_tree_with_root(max_nodes):
    start = long_tree_structures(max_nodes)[-1]
    if max_nodes > 1:
        return [len(start)] + start
    return start

def gen_cmds(vs, ad):
    if len(vs) == 0:
        return [ad]
    k, vals = vs[0]
    ret = []
    for v in vals:
        ret.extend(gen_cmds(vs[1:], ad+['--'+k, str(v)]))
    return ret
