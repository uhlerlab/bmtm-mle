import math
import random
from scipy import optimize
import numpy as np

class Tree:  
    ''' Recusive BMTM datastructure

    Allows getting/setting of variance and observed data
    Methods for computing likelihood'''

    def __init__(self, parent=None):
        self.children = []
        self.above_var = 0
        self.data = None
        self.parent = parent

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

    def set(self, args):
        self.above_var = args[0]
        count = 1
        for c in self.children:
            count += c.set(args[count:]) 

        return count

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
        return detlog + tr

    def is_singular(self):
        m_arr, s_arr = self._build_matrices()
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


if __name__ == '__main__':
    #Use scipy's optimize module to estimate MLE

    t = Tree()
    t.make_prefix([0, 0])
    t.set_data([2, 5])

    def is_singular(args):
        t.set(args)
        return int(t.is_singular())

    def opt(args):
        t.set(args)
        ll = t.likelihood()
        return -1*ll # min to max

    size = t.num()
    starting = [random.uniform(0, 1) for _ in range(size)]
    bnds = tuple((0, None) for _ in range(size))
    cons = ({'type': 'ineq', 'fun': is_singular},)

    # Various optimization methods
    #res = optimize.minimize(opt, starting, method='trust-constr', bounds=bnds, constraints=cons)
    #res = optimize.minimize(opt, starting, method='SLSQP', bounds=bnds, constraints=cons)
    #res = optimize.dual_annealing(opt, tuple((0, 30) for _ in range(size)))
    res = optimize.differential_evolution(opt, tuple((0, 100) for _ in range(size)))
    #res = optimize.minimize(opt, starting, bounds=bnds)

    print('---Result---')
    print(res)
    print('------------')
    print("Compute current liklihood:")
    print(t.likelihood(True))
