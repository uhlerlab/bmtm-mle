''' 
Test the general MLE solver contained in solver.py
Run against unit tests and compared to MLEs discovered by generate_mle.py
'''
import argparse
import json
import math

import numpy as np
from tqdm import tqdm

from scratch.tree import Tree
from util import * 
from solver import * 

def test_solver():
    solver = Solver()

    for p in range(2, 10):
        print("Trying on {} node star".format(p))
        tree = Tree()
        tree.make_prefix([0]*p)
        data = list(np.random.normal(size=(p)) )
        tree.set_data(data)

        solver.predict_mle(tree)
        solve_likelihood = tree.likelihood()
        solve_vars = tree.get_var()

        star_vars = var_prediction_star(data)
        tree.set_var(star_vars)
        star_likelihood = tree.likelihood()

        if star_likelihood > solve_likelihood:
            print('MLE of solve then star', solve_vars, star_vars)
            print('Likelihood of solve then star', solve_likelihood, star_likelihood)
            raise ValueError('Star function finds better likelihood')

    tree = Tree()
    tree.make_prefix([0, 2, 0, 0])
    tree.set_data([-1, 5, 2])
    solver.predict_mle(tree)
    assert(tuple(tree.get_var()) == (0, 1, 4, 9, 0))

    print("Pass")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='res_file_1000_0-2-0-0.json')
    args = parser.parse_args()

    print('Running tests')
    test_solver()

    mle_solver = Solver()

    with open(args.in_file, 'r') as res_file:
        file_dict = json.load(res_file)
        tree = Tree()  
        assert(a == 0 for a in file_dict['structure']) # only stars for now
        tree.make_prefix(file_dict['structure'])

        buckets = {}
        for i, res in enumerate(tqdm(file_dict['results'])):
            data, mle = res
            tree.set_data(data)

            mle_solver.predict_mle(tree)
            var_pred = tree.get_var()
            assert(all(a >= 0 for a in var_pred))
            pred = tree.likelihood()
            pred_struct = tree.zero_pattern()

            tree.mle(method='trust-constr')
            other_l = tree.likelihood()

            tree.set_var(mle)
            found = tree.likelihood()
            found_struct = tree.zero_pattern()

            if found-FLOATING_POINT_EPS > pred or other_l-FLOATING_POINT_EPS > pred:
                print('Fail', pred, found, other_l)
                print(var_pred, mle)
                raise ValueError('Did not predict MLE')
            else:
                print('Pass', pred, found, pred_struct == found_struct, pred_struct, found_struct)