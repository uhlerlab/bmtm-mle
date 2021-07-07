import argparse
import math
import subprocess
import os.path

from scratch.runtime_comparison import long_tree_structures, fat_tree_structures, stars
from scratch.util import gen_cmds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    known, unknown = parser.parse_known_args()

    VARS = [('normalize', [0, 1]), ('random_fo', [0, 1]), 
        ('random_method', ['exponential', 'uniform']),
        ('format', ['tree', 'covariance']),
        ('tree_type', ['star', 'bin', 'long'])]

    '''VARS = [('normalize', [0, 1]), ('random_fo', [0, 1]), 
        ('random_method', ['exponentialultrametric']),
        ('format', ['tree', 'covariance']),
        ('tree_type', ['random_bin'])]
    '''


    for ad in gen_cmds(VARS, []):
        print(ad)
        cmd = 'python3 ./scratch/gen_sample_series.py'.split() + ad + unknown
        print(' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output, error = process.communicate()