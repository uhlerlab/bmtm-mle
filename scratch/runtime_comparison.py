'''
Compare the runtime of our solver to others
Dumps CSV of runtime numbers
'''
import random 
import math 
import os 
import os.path
import time 
import signal
from contextlib import contextmanager

from scratch.tree import Tree
from scratch.solver import Solver
from scratch.util import *

# From stack overflow
class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def dump_experiment(times, title=None):
    if title is None:
        title = str(random.uniform(0, 10000000))

    path = '{}.csv'.format(title)
    if os.path.exists(path):
        path = '{}.csv'.format(title + str(random.uniform(0, 10000000)))
        print('File already exists. Changing name')
        print('Path is', path)
    with open(path, 'w') as out_file:
        ordered_keys = sorted(list(times[list(times.keys())[0]].keys()))
        out_file.write('p,'+','.join(ordered_keys)+'\n')
        for p in times:
            out_file.write(str(p)+','+','.join(str(times[p][k]) for k in ordered_keys)+'\n')

def compute_experiments(structures, solvers, title=None, upper_limit=60*5, tol=1):
    canonical_solver = Solver()
    times = {}
    for struct in structures:
        tree = Tree()
        tree.make_prefix(struct)
        tree.set_random_data()

        p = tree.num_leaf_nodes()
        times[p] = {}

        try:
            with time_limit(upper_limit):
                st = time.time()
                canonical = canonical_solver.predict_mle(tree)
                canonical_time = time.time() - st
        except TimeoutException as e:
            print("Couldn't solve canonical in time")
            return times

        canonical_likelihood = tree.likelihood() 
        times[p]['canonical'] = canonical_time

        for name, solver in solvers: 
            iter_param = 0
            while True:
                try:
                    with time_limit(upper_limit):
                        st = time.time()
                        found_likelihood = solver(tree, iter_param=iter_param)
                        total_time = time.time() - st
                except TimeoutException as e:
                    print("Hit time limit")
                    times[p][name] = upper_limit
                    break
                if found_likelihood >= canonical_likelihood - tol:
                    times[p][name] = total_time
                    break
                else:
                    iter_param += 1
            print(canonical_likelihood, found_likelihood)

    return times

if __name__ == '__main__':
    structures = long_tree_structures(8)
    TITLE = 'long_trees'
    #structures = stars(10)
    #TITLE = 'stars'
    #structures = fat_tree_structures(3)
    #TITLE = 'fat_tree'

    print(structures)
    def da_solver(tree, iter_param):
        tree.mle(method='dual_annealing', accept=(-1e4)+1, maxiter=int(math.exp(iter_param)*50))
        return tree.likelihood()
    def de_solver(tree, iter_param):
        tree.mle(method='differential_evolution', maxiter=int(math.exp(iter_param)*50))
        return tree.likelihood()

    solvers = [('differential_evolution', de_solver), ('dual_annealing', da_solver)]
    times = compute_experiments(structures, solvers)
    dump_experiment(times, title=TITLE)
    