'''
Draw picture
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

def get_canvas(width=1000, height=800):
    win= Tk()
    win.geometry("{}x{}".format(width, height))
    c= Canvas(win,width=width, height=height)
    c.pack()
    return win, c, width, height

if __name__ == '__main__':
    data = [7.995279617997176,  8.030069685088913, -10.487949480851258, -2.8155541964710675]
    var1 = [12.468430995407857, 16.93159747971192, 16.20508699552593, 18.453967565643346, 12.449657732934805]
    var2 = [63.924496169961074, 0.0, 0.0012103487682475415, 341.6297579205175, 116.87412776404997]
    prefix = [0, 0, 0, 0]

    t1 = Tree()
    t1.make_prefix(prefix)
    print('children',t1.num_leaf_nodes())
    t1.set_data(data)
    t1.set_var(var1)

    t2 = Tree()
    t2.make_prefix(prefix)
    t2.set_data(data)
    t2.set_var(var2)

    win, can, width, height = get_canvas()
    can.create_text(50, 20, anchor=W, text='Ground Truth Tree')
    t1.draw(can, 0, 100, xspace=100, round_number=3)
    
    can.create_text(width/2+50, 20, anchor=W, text='BMTM MLE Tree')
    t2.draw(can, width/2, 100, xspace=100, round_number=3)

    win.mainloop()