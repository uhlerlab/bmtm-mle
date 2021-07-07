import argparse
import math
import subprocess
import os.path

from scratch.runtime_comparison import long_tree_structures, fat_tree_structures, stars
from scratch.util import gen_cmds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    known, unknown = parser.parse_known_args()
    print(unknown)

    VARS = [('normalize', [0, 1]), ('random_fo', [0, 1]), 
        ('random_method', ['exponential', 'uniform']),
        ('format', ['tree', 'covariance']),
        ('tree_type', ['star', 'bin', 'long'])]
    VARS = [('normalize', [0, 1]), ('random_fo', [0]), 
        ('random_method', ['exponentialultrametric']),
        ('format', ['tree', 'covariance']),
        ('tree_type', ['random_bin'])]
    for ad in gen_cmds(VARS, []):
        title = '-'.join(ad[i-1][2:]+'_'+ad[i] if ad[i].isdigit() else ad[i] for i in range(1, len(ad), 2))
        cmd = 'python3 ./scratch/run_plot.py --title {} --out_file {}.png '.format(
            title,
            title,
            ).split() + ad + unknown
        print(' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)