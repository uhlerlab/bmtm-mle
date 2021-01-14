'''Tree datastructure'''
import math
import random

import numpy as np
from scipy import optimize

EPSILON = 1e-6

class Tree:  
    ''' Recusive BMTM datastructure

    Allows getting/setting of variance and observed data
    Methods for computing likelihood'''

    def __init__(self, parent=None):
        self.children = []
        self.above_var = 0
        self.data = None
        self.parent = parent
    
    def hash(self):
        if self.parent is None:
            return (0,)
        
        return self.parent.hash() + (self.parent.children.index(self),)

    def make_child(self):
        self.children.append(Tree(self))

    def num(self):
        count = 1
        for c in self.children:
            count += c.num() 
        return count

    def num_leaf_nodes(self):
        if len(self.children) == 0:
            return 1

        count = 0
        for c in self.children:
            count += c.num_leaf_nodes() 
        return count

    def set_var(self, args):
        self.above_var = args[0]
        count = 1
        for c in self.children:
            count += c.set_var(args[count:]) 

        return count

    def get_var(self):
        ans = [self.above_var]
        for c in self.children:
            ans.extend(c.get_var())
        return ans

    def set_data(self, data):
        if len(self.children) == 0:
            self.data = data[0]
            return 1 

        count = 0
        for c in self.children:
            count += c.set_data(data[count:]) 
        return count
    
    def var(self):
        return self.above_var + (0 if self.parent is None else self.parent.var())

    def _build_matrices(self):
        nodes = [self.get_leaf(i) for i in range(self.num_leaf_nodes())]
        m_arr = [[a.covar(b) for b in nodes] for a in nodes]
        s_arr = [[a.data*b.data for b in nodes] for a in nodes]
        x_arr = [a.data for a in nodes]

        assert(all(m_arr[i][j] == m_arr[j][i] for i in range(len(m_arr)) for j in range(len(m_arr))))
        assert(all(s_arr[i][j] == s_arr[j][i] for i in range(len(s_arr)) for j in range(len(s_arr))))
        
        return m_arr, s_arr, x_arr

    def _is_singular(self, m):
        return False if np.linalg.matrix_rank(m) == m.shape[0] else True

    def likelihood(self, debug=False):
        m_arr, s_arr, x_arr = self._build_matrices()
        m = np.array(m_arr)
        #m = np.reshape(np.array(params), (self.num_leaf_nodes(), -1))
        s = np.array(s_arr)
        x = np.array(x_arr)
        if debug:
            print("Computing likelihood...")
            print(m)
            print(s)

        if self._is_singular(m):
            return -99999999

        inv = np.linalg.inv(m)
        if debug:
            print(inv)

        detlog = np.log(np.linalg.det(inv)) 
        tr = - np.matmul(np.matmul(np.transpose(x), inv), x)
        # This should be equivalent? tr = - np.trace(np.matmul(s, inv)))
        if debug:
            print('DL', detlog, ', Tr', tr)
        score = detlog + tr
        if math.isnan(score) or math.isinf(score):
            return -9999999
        return score

    def is_singular(self):
        m_arr, s_arr, x_arr = self._build_matrices()
        m = np.array(m_arr)
        return -2*int(self._is_singular(m)) + 1

    def get_leaf(self, ind):
        if len(self.children) == 0:
            if ind == 0:
                return self
            else:
                raise ValueError('Reach leaf node with non zero ind')

        count = 0
        for c in self.children:
            toadd =  c.num_leaf_nodes()
            if count + toadd > ind:
                return c.get_leaf(ind - count)
            count += toadd

        raise ValueError("Couldnt find child with index")
    
    def covar(self, other):
        me = self 
        them = other
        while me is not None and them != me:
            while them is not None and them != me:
                them = them.parent

            if them == me:
                break

            them = other 
            me = me.parent

        if them is None or me is None or them is not me:
            raise ValueError('Couldnt find LCA')
        
        return them.var()

    def make_prefix(self, l):
        i = 0
        while i < len(l):
            n = l[i]
            self.make_child()
            if n > 0:
                self.children[-1].make_prefix(l[i+1:i+n+1])
            i += 1 + n
    
    def mle(self, method='trust-constr', max_var=30, accept=-100, maxiter=500):
        def is_singular(args):
            self.set_var(args)
            return int(self.is_singular())

        def opt(args):
            self.set_var(args)
            ll = self.likelihood()
            return -1*ll # min to max

        size = self.num_leaf_nodes()**2
        #print(size)
        size = self.num()
        starting = [random.uniform(0, 1) for _ in range(size)]
        bnds = tuple((0, None) for _ in range(size))
        cons = ({'type': 'ineq', 'fun': is_singular},)
        print(bnds, cons)

        # Various optimization methods
        global_param_space = tuple((0, max_var) for _ in range(size))
        if method == 'dual_annealing':
            res = optimize.dual_annealing(opt, global_param_space, accept=accept, maxiter=maxiter, initial_temp=10000)
        elif method == 'differential_evolution':
            res = optimize.differential_evolution(opt, global_param_space)
        else:
            res = optimize.minimize(opt, starting, method=method, bounds=bnds, constraints=cons)

        self.set_var(res.x)
        return res.x
    
    def is_observed_node(self):
        if len(self.children) == 0:
            return True
        if self.above_var < EPSILON:
            return all(c.is_observed_node() for c in self.children) 

        return any(c.above_var < EPSILON and c.is_observed_node() for c in self.children) 
    
    def is_fully_observed(self):
        return self.is_observed_node() and all(c.is_fully_observed() for c in self.children)
    

    def zero_pattern(self):
        p = (self.above_var < EPSILON,)
        for c in self.children:
            p += c.zero_pattern()
        return p


if __name__ == '__main__':
    tree = Tree()
    tree.make_prefix([0, 0, 0])
    tree.set_data([1, 2, 3])
    tree.set_var([0, 1, 4, 9])
    print(tree.likelihood())