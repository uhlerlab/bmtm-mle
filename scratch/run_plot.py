'''
Show results from risk by # of nodes across estimators
'''
import argparse
import glob
import subprocess
import os.path

from scratch.gen_sample_series import add_reconstruct_args, reconstruct_file_tag, reconstruct_description

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_reconstruct_args(parser)
    parser.add_argument('--estimators', nargs='+', type=str, default=None)

    args, unknown = parser.parse_known_args()

    ft = reconstruct_file_tag(args)
    wildcard = '*-{}'.format(ft)
    print(wildcard)

    estimators = set(fn.split('-')[2] for i, fn in enumerate(sorted(glob.glob(wildcard))))
    print(estimators)
    if args.estimators is not None:
        estimators = estimators.intersection(set(args.estimators))
    files = ['reconstruct_series-*-{}-{}'.format(e, ft) for e in estimators]
    print(files)

    cmd = ('python3 scratch/sample_and_reconstruct_series_plot.py --in_file_wildcard'.split() 
        + files + unknown + ['--description', reconstruct_description(args)])
    print('cmd', ' '.join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode())