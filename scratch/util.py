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