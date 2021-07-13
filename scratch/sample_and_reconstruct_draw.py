'''
Draw pictures based on the results of sample and reconstruct
'''
import argparse
import math
import itertools

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import average
from tkinter import *

from scratch.tree import average_tree_percent_zero, in_tree, average_tree
from scratch.solver import Tree

def get_canvas(width=1200, height=800):
    win= Tk()
    win.geometry("{}x{}".format(width, height))
    c= Canvas(win,width=width, height=height)
    c.pack()
    return win, c, width, height

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='samples.txt')
    parser.add_argument('--title', type=str, default='')
    #parser.add_argument('--display', type=str, default='risk')
    args = parser.parse_args()

    with open(args.in_file, 'r') as f:
        lines = f.read().strip().split('\n')
        fmt = lines.pop(0)
        if fmt == 'tree':
            assert(len(lines) % 2 == 0)
            num_trials = len(lines)//2
            
            t1, t2 = itertools.tee((in_tree(lines[num_trials+ind]) for ind in range(num_trials)))
            a_tree = average_tree(t1)
            c_tree = average_tree_percent_zero(t2)
            gt_tree = in_tree(lines[0])
            assert(gt_tree.get_var() == in_tree(lines[num_trials-1]).get_var())

            bias_tree = in_tree(lines[0])
            bias_tree.set_var([a-b for a, b in zip(a_tree.get_var(), gt_tree.get_var())])

            win, can, width, height = get_canvas()
            can.create_text(50, 20, anchor=W, text='Ground Truth Tree')
            gt_tree.draw(can, 0, 100)

            can.create_text(width/3+50, 20, anchor=W, text='Average BMTM MLE Tree')
            a_tree.draw(can, width/3, 100)

            #can.create_text(2*width/3+50, 20, anchor=W, text='Bias Tree')
            #bias_tree.draw(can, 2*width/3, 100)

            can.create_text(2*width/3+50, 20, anchor=W, text='% Zeroed Out for BMTM MLE')
            c_tree.draw(can, 2*width/3, 100)

            #can.create_text(50, height/2, anchor=W, text='% Zeroed Out for BMTM MLE')
            #c_tree.draw(can, 0, height/2+100)
            #can.create_text(width/2+50, height/2, anchor=W, text='Bias Tree')
            #bias_tree.draw(can, width/2, height/2+100)
            win.mainloop()
        else:
            raise ValueError('Format not supported')