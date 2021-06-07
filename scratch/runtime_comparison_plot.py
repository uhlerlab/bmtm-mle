'''
Plot the results of runtime comparison
'''
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='stars.csv')
    parser.add_argument('--log', type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.in_file)
    df = pd.melt(df.reset_index(), id_vars=['p'], 
        value_vars=list(set(df.keys()).difference(['p'])), 
        var_name='label', 
        value_name='runtime')
    #df = df.stack().to_frame(name=['name', 'name2'])

    #fit = np.polyfit(df['p'], df['canonical'], 2)
    #equation = np.poly1d(fit)
    #print ("The fit coefficients are a = {0:.4f}, b = {1:.4f} c = {2:.4f}".format(*fit))
    #print (equation)

    '''xmesh = np.linspace(min(df['p']), max(df['p']), len(df['p']))
    print(xmesh)
    df['other'] = equation(xmesh)
    '''

    g = sns.lineplot(data = df, x = 'p', y = 'runtime', hue='label')
    if args.log != 0:
        g.set_yscale('log')
    plt.show()
