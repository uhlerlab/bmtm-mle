'''Tree datastructure'''
import math
import random
import re
import subprocess

import numpy as np

from scipy import optimize
from util import fr_norm_sq

EPSILON = 1e-6

LOW_NUMBER = -999999

def rf(a, b):
    ass = a.get_splits()
    bss = b.get_splits()

    return sum(int(s not in bss) for s in ass) + sum(int(s not in ass) for s in bss)

def in_tree(line):
    pre, vari, data, labels = (list(map(f, a.split(','))) 
        for a, f in zip(line.split('\t'), [int, float, float, int]))
    t = Tree()
    t.make_prefix(pre)
    t.set_var(vari)
    t.set_data(data)
    t.set_labels(labels)
    return t 

def average_tree_percent_zero(gen):
    num_trees = 1
    sumt = [1 if a < 1e-7 else 0 for a in next(gen).get_var()]
    for t in gen:
        sumt = [s+(1 if a < 1e-7 else 0) for s, a in zip(sumt, t.get_var())]
        num_trees += 1
    nt = Tree()
    nt.make_prefix(t.get_prefix())
    nt.set_var([s/num_trees for s in sumt])
    return nt

def average_tree(gen):
    num_trees = 1
    sumt = next(gen).get_var()
    for t in gen:
        sumt = [s+a for s, a in zip(sumt, t.get_var())]
        num_trees += 1
    nt = Tree()
    nt.make_prefix(t.get_prefix())
    nt.set_var([s/num_trees for s in sumt])
    return nt

def box(x, y, width):
    return (x-width/2, y, x+width/2, y+width)

def bhv_distance_owens(t1, t2, fh='tmpbhv.txt'):
    input = t1.newick() + '\n' + t2.newick()
    #print('input')
    #print(input)
    with open(fh, 'w') as f:
        f.write(input)
    cmd = 'java -jar jj.jar -o /dev/stdout {}'.format(fh)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    comps = output.decode().strip().split()
    print(comps)

    return float(comps[-1])

def bhv_distance_owens_list(ts, fh='tmpbhv.txt'):
    input = '\n'.join(ts)
    with open(fh, 'w') as f:
        f.write(input)
    cmd = 'java -jar jj.jar -o /dev/stdout {}'.format(fh)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    comps = [line.strip().split() for line in output.decode().strip().split('\n')]
    return [float(c[-1]) for c in comps]


def gaussian_likelihood(m, s):
    #print(m, s)
    inv = np.linalg.inv(m)
    detlog = np.log(np.linalg.det(inv)) 
    tr = -np.trace(np.matmul(s, inv))
    score = detlog + tr
    return score

class Tree:  
    ''' Recusive BMTM datastructure

    Allows getting/setting of variance and observed data
    Methods for computing likelihood'''

    def __init__(self, parent=None):
        self.children = []
        self.above_var = 0
        self.data = None
        self.label = None
        self.parent = parent
    
    def hash(self):
        if self.parent is None:
            return (1, 0,)
        
        return self.parent.hash() + (self.parent.children.index(self),)

    # ONLY USE FOR NODES IN THE SAME TREE
    # A dictionary of nodes from multiple trees will have unexpected behavior
    def __hash__(self):
        return int(''.join(map(str, self.hash())))

    def make_child(self):
        self.children.append(type(self)(parent=self))

    def num(self):
        count = 1
        for c in self.children:
            count += c.num() 
        return count

    def num_depth(self):
        if len(self.children) == 0:
            return 0
        return 1+max(c.num_depth() for c in self.children)

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
    
    def random_fo(self, seed):
        count = 1
        if len(self.children) > 0:
            zero_index = seed[0] % len(self.children)
            self.children[zero_index].above_var = 0
            for c in self.children:
                count += c.random_fo(seed[count:]) 

        return count

    def sample_data(self, mean=0):
        newmean = np.random.normal(loc=mean, scale=math.sqrt(self.above_var))
        #newmean = np.random.uniform(low=mean-math.sqrt(self.above_var), high=mean+math.sqrt(self.above_var))
        if len(self.children) == 0:
            return [newmean]
        out = []
        for c in self.children:
            out.extend(c.sample_data(newmean))
        return out

    def set_random_data(self):
        p = self.num_leaf_nodes()
        return self.set_data(list(np.random.normal(size=(p,))))

    def newick(self, root=True):
        if len(self.children) == 0:
            return '{}:{}'.format(self.data, self.above_var)
        
        body = ','.join(c.newick(False) for c in self.children)
        if not root:
            return '({}):{}'.format(body, self.above_var)
        return '({},0:{});'.format(body, self.above_var)

    def sparse_newick(self):
        if len(self.children) == 0:
            return str(self.label)

        direct_child_newicks = []
        newicks = []
        for c in self.children: 
            n = c.sparse_newick()
            if c.above_var == 0:
                direct_child_newicks.append(n)
            else:
                newicks.append(n)
        candidate_parents = [n for n in direct_child_newicks if n[-1] != ')']
        if len(candidate_parents) > 1:
            raise ValueError("Two candidate parents")
        
        def get_inside_parens(s):
            inside_parens = re.compile('\((.*)\)')
            try:
                return inside_parens.search(s).group(1)
            except:
                return None
        direct_child_newicks_inside = [get_inside_parens(n) for n in direct_child_newicks]
        direct_child_newicks_inside = [n for n in direct_child_newicks_inside if n is not None]
        new = '(' + ', '.join(newicks+direct_child_newicks_inside) + ')'
        if len(candidate_parents) > 0:
            i = candidate_parents[0].rfind(')')
            new += candidate_parents[0][i+1:]
        return new

    def _set_at_leaves(self, key, arr):
        if len(self.children) == 0:
            setattr(self, key, arr[0])
            return 1 

        count = 0
        for c in self.children:
            count += c._set_at_leaves(key, arr[count:]) 
        return count

    def _get_at_leaves(self, key):
        if len(self.children) == 0:
            return [getattr(self, key)]
        ans = []
        for c in self.children:
            ans.extend(c._get_at_leaves(key))
        return ans

    def get_labels(self):
        return self._get_at_leaves('label')

    def set_labels(self, labels):
        return self._set_at_leaves('label', labels)

    def get_data(self):
        return self._get_at_leaves('data')

    def set_data(self, data):
        return self._set_at_leaves('data', data)
    
    def var(self):
        return self.above_var + (0 if self.parent is None else self.parent.var())
    
    def cov_matrix(self):
        #nodes = [self.get_leaf(i) for i in range(self.num_leaf_nodes())]
        nodes = list(self.leaves() )
        return [[a.covar(b) for b in nodes] for a in nodes]

    def _build_matrices(self):
        #nodes = [self.get_leaf(i) for i in range(self.num_leaf_nodes())]
        nodes = list(self.leaves())
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
        n = len(x_arr)
        m = np.array(m_arr)
        #m = np.reshape(np.array(params), (self.num_leaf_nodes(), -1))
        s = np.array(s_arr)
        x = np.array(x_arr)
        if debug:
            print("Computing likelihood...")
            #print(m)
            #print(s)

        if self._is_singular(m):
            return LOW_NUMBER 

        score = gaussian_likelihood(m, s)
        if math.isnan(score) or math.isinf(score):
            return LOW_NUMBER 
        return (1/2)*score

    def is_singular(self):
        m_arr, s_arr, x_arr = self._build_matrices()
        m = np.array(m_arr)
        return -2*int(self._is_singular(m)) + 1

    def nodes(self):
        yield self
        for c in self.children:
            yield from c.nodes()

    def latents(self):
        if len(self.children) > 0:
            yield self
            for c in self.children:
                yield from c.latents()

    def leaves(self):
        if len(self.children) == 0:
            yield self
        else:
            for c in self.children:
                yield from c.leaves()
    
    def draw(self, can, x, y, width=20, yspace=50, xspace=60, scale=1, round_number=2):
        width *= scale
        yspace *= scale
        xspace *= scale

        spacer_x = (width+xspace)
        next_y = y + width + yspace
        bot_width = spacer_x*(self.num_leaf_nodes())
        next_x = x 

        my_center_x = x+bot_width/2
        my_center_y = y+width/2
        can.create_oval(*box(my_center_x, y, width), fill='black')

        if self.parent is None:
            can.create_line(my_center_x, my_center_y, 
                my_center_x, my_center_y-yspace)
            can.create_text(my_center_x+2, my_center_y-(3/4)*yspace, font="Arial",
                text="{}".format(round(self.above_var, round_number)))

        for c in self.children:
            win = spacer_x*c.num_leaf_nodes()
            line = (my_center_x, my_center_y, 
                next_x+win/2, next_y+width/2)
            can.create_line(*line)
            can.create_text((line[0]+line[2])/2, (line[1]+line[3])/2, font="Arial",
                text="{}".format(round(c.above_var, round_number)))
            c.draw(can, next_x, next_y, width, yspace, xspace, 1, round_number)
            next_x += win

        if len(self.children) > 0:
            assert(abs(next_x - (x + bot_width)) < 1e-6)
        elif self.data is not None:
            can.create_text(my_center_x, my_center_y+width, font="Arial",
                text="{}".format(round(self.data, round_number)))

    def _splits_below(self):
        if len(self.children) == 0:
            return [((self.label,), self.above_var)]
        
        splits = []
        below = set()
        for c in self.children:
            new = c._splits_below()
            splits.extend(new)
            for split, av in new:
                below.update(split)
        return splits + [(tuple(sorted(below)), self.above_var)]

    def get_splits(self):
        raw_labels = self.get_labels()
        if any(a is None for a in raw_labels):
            raise ValueError('Splits operates on labels. One of your labels is empty.')
        if any(not isinstance(a, (int, float)) for a in raw_labels):
            raise ValueError('For splits to work, your labels need to be numeric.')
        labels = set(raw_labels)
        labels.add(-100)
        below_splits = self._splits_below()

        return [tuple(sorted((s, tuple(sorted(labels - set(s)))))) for s, av in below_splits if av > 1e-8]

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
    
    def lca(self, other):
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
        
        return them

    def covar(self, other):
        return self.lca(other).var()

    def make_prefix(self, l):
        i = 0
        while i < len(l):
            n = l[i]
            self.make_child()
            if n > 0:
                self.children[-1].make_prefix(l[i+1:i+n+1])
            i += 1 + n

    def get_prefix(self, root=True):
        if len(self.children) == 0:
            return [0]
        ans = []
        for c in self.children:
            ans.extend(c.get_prefix(False))
        if root:
            return ans
        return [len(ans)]+ans
    
    def mle_size(self):
        return self.num()
    
    def mle_set(self, var):
        return self.set_var(var)
    
    def mle(self, method='trust-constr', max_var=30, accept=-100, maxiter=500, lam=0, global_param_space=None):
        def is_singular(args):
            #args = [self.top]+list(args)[1:] # del

            self.mle_set(args)
            return int(self.is_singular())

        def opt(args):
            #args = [self.top]+list(args)[1:] # del
            
            if any(a < 0 for a in args):
                return -LOW_NUMBER 

            self.mle_set(args)
            ll = self.likelihood()
            if lam != 0:
                #print(lam*sum(a**2 for a in self.get_var()), ll)
                ll -= lam*sum(a**2 for a in self.get_var())
            return -1*ll # min to max

        size = self.mle_size()
        starting = [random.uniform(-1, 1) for _ in range(size)]
        bnds = tuple((0, None) for _ in range(size))
        #bnds = tuple((None, None) for _ in range(size))
        cons = ({'type': 'ineq', 'fun': is_singular},)
        #print(bnds, cons)

        # Various optimization methods
        if global_param_space is None:
            global_param_space = tuple((0, max_var) for _ in range(size))
        if method == 'dual_annealing':
            res = optimize.dual_annealing(opt, global_param_space, accept=accept, maxiter=maxiter, initial_temp=10000)
        elif method == 'differential_evolution':
            res = optimize.differential_evolution(opt, global_param_space)
        elif method == 'basinhopping':
            res = optimize.basinhopping(opt, starting)
        else:
            res = optimize.minimize(opt, starting, method=method, bounds=bnds, constraints=cons)

        #res2 = [self.top]+list(res.x)[1:] # del
        #self.set_var(res2)
        self.mle_set(res.x)
        return res.x
        #return res2

    def fr_proj(self, method='trust-constr', max_var=30, accept=-100, maxiter=500):
        def is_singular(args):
            self.set_var(args)
            return int(self.is_singular())

        def opt(args):
            if any(a < 0 for a in args):
                return -LOW_NUMBER 
            self.set_var(args)
            m_arr, s_arr, x_arr = self._build_matrices()
            return fr_norm_sq(m_arr, s_arr)

        size = self.num()
        starting = [random.uniform(-1, 1) for _ in range(size)]
        bnds = tuple((0, None) for _ in range(size))
        cons = ({'type': 'ineq', 'fun': is_singular},)

        # Various optimization methods
        global_param_space = tuple((0, max_var) for _ in range(size))
        if method == 'dual_annealing':
            res = optimize.dual_annealing(opt, global_param_space, accept=accept, maxiter=maxiter, initial_temp=10000)
        elif method == 'differential_evolution':
            res = optimize.differential_evolution(opt, global_param_space)
        elif method == 'basinhopping':
            res = optimize.basinhopping(opt, starting)
        else:
            res = optimize.minimize(opt, starting, method=method, bounds=bnds, constraints=cons)

        self.set_var(res.x)
        return res.x
    
    def is_observed_node(self, eps=EPSILON):
        if len(self.children) == 0:
            return True
        if self.above_var < eps:
            return all(c.is_observed_node() for c in self.children) 

        return any(c.above_var < eps and c.is_observed_node() for c in self.children) 
    
    def is_fully_observed(self, eps=EPSILON):
        return self.is_observed_node(eps) and all(c.is_fully_observed() for c in self.children)
    

    def zero_pattern(self):
        p = (self.above_var < EPSILON,)
        for c in self.children:
            p += c.zero_pattern()
        return p

    def fo(list_nodes):
        if len(list_nodes) == 0:
            return []
        return ([a + b
            for a in list_nodes[0].gen_r()
                for b in fo(list_nodes[1:])]
        + [a + b
            for a in list_nodes[0].gen_u()
                for b in fo(list_nodes[1:])])
    
    def gen_t(self):
        return self.gen_r() + self.gen_u()

    def gen_r(self):
        if len(self.children) == 0:
            return ((False,),)
        return [b + a + c
            for i, c in enumerate(self.children)
                for a in c.gen_r()
                    for b in fo(self.children[:i]) 
                        for c in fo(self.children[i+1:])]

    def gen_u(self):
        if len(self.children) == 0:
            return tuple()
        return fo(self.children)

import time

def s(xs, vs):
    print(xs, vs)
    n = 1#len(xs)
    print(n)
    sig = 1/(sum(1/v for v in vs))
    '''return np.log(
        1/((2*math.pi)**((n-1)/2))
        *math.sqrt(sig)/math.prod(math.sqrt(v) for v in vs)
        *math.exp(
            (1/2)*sig*((sum(x/v for x, v in zip(xs, vs)))**2) 
            - (1/2)*sum((x**2)/v for x, v in zip(xs, vs))))
    '''
    '''return (np.log(1) - ((n-1)/2)*np.log(((2*math.pi)))
        +(1/2)*np.log(sig)
        -(1/2)*np.log(math.prod(vs))
        + (1/2)*sig*((sum(x/v for x, v in zip(xs, vs)))**2)
        - (1/2)*sum((x**2)/v for x, v in zip(xs, vs)))
    '''
    mu = (sum(x/v for x, v in zip(xs, vs)))*sig
    print('ok',
            sum((x**2)/v for x, v in zip(xs, vs))
            - mu**2/sig
        )
    return (((1-n)/2)*np.log(((2*math.pi)))
        +(1/2)*np.log(sig)
        -(1/2)*np.log(math.prod(vs))
        - (1/2)*(
            sum((x**2)/v for x, v in zip(xs, vs))
            - mu**2/sig)
        )

if __name__ == '__main__':
    tree = Tree()
    tree.top = None
    tree.make_prefix([2, 0, 0, 2, 0, 0])
    tree.set_data([1, 2, 5, 4])
    assert(tree.is_fully_observed())
    print(tree.mle(method='differential_evolution'))
    print(tree.likelihood())
    print(tree.get_var())
    assert(False)

    for i in range(0, 10):
        tree = Tree()
        tree.top = i
        tree.make_prefix([2, 0, 0, 2, 0, 0])
        tree.set_data([1, 2, 0.1, 4])
        assert(tree.is_fully_observed())
        print(tree.mle(method='differential_evolution'))
        print('MADE IT')

    tree = Tree()
    tree.top = 0
    tree.make_prefix([0, 0])
    tree.set_data([1, 2])
    print(tree.mle(method='differential_evolution'))
    assert(False)

    tree = Tree()
    p = 3
    tree.make_prefix([0]*p)
    d = list(np.random.normal(size=(p,)))
    tree.set_data(d)
    v = [0.00001]+[abs(a) for a in np.random.normal(size=(p+1,))][1:]
    tree.set_var(v)
    print(tree.likelihood())
    print(s([0]+d, v))
    print(s([0]+d+[2], v+[2])/s([0]+d, v))
    assert(False)
    print(tree.mle(method='differential_evolution'))
    tree = Tree()
    tree.make_prefix([0, 0, 0, 0])
    tree.set_data([0.95, 1, 2, 4])
    print(tree.mle(method='differential_evolution'))
    assert(False)
    #data = [-0.5, 3, 4]

    '''for i in range(100):
        data = list(np.random.normal(size=(3,)))
        tree.set_data(data)

        tree.top = abs(list(np.random.normal(size=(1,)))[0])
        tree.mle(method='differential_evolution')
        sl = tree.likelihood()
        print(sl, tree.get_var())

        var = tree.get_var()
        if 0 in list(var):
            continue
        var[0] = 0
        tree.set_var(var)
        nl = tree.likelihood()
        print(nl, nl > sl, tree.get_var())
        if nl <= sl:
            raise ValueError('Did not improve')
    '''


    data = [-1, 1]
    tree.set_data(data)
    tree.set_var([0.5, 0.5, 0.5])
    #tree.set_var([0, 1, 1])
    print(tree.likelihood())
    
    #var = [0.5190013868567017, 0.07556557282991373, 0.4716822586090975, 0.7244937043066909] 
    tree.top = 0.5
    tree.mle(method='differential_evolution')
    print(tree.get_var())
    var = [0.5, 1, 1.5**2]
    tree.set_var(var)
    print(tree.likelihood())
    sl = tree.likelihood()
    print('none', sl)
    for i in range(len(var)):
        var2 = list(var)
        var2[i] = 0
        tree.set_var(var2)
        print(i, tree.likelihood())
    for i in range(len(var)):
        var2 = list(var)
        var2[i] += .1
        tree.set_var(var2)
        nl = tree.likelihood()
        print(i, nl, nl > sl)
    for i in range(len(data)):
        var2 = [data[i]**2] + [(data[j]-data[i])**2 for j in range(len(data))]
        tree.set_var(var2)
        print(i, tree.likelihood())

    '''top = 0.5190013868567017
    #tree.mle(method='differential_evolution')
    #print(tree.likelihood(), tree.get_var())
    #var = [top] + [(math.sqrt(top)-d)**2 for d in data]
    #var = [0.5190013868567017, 7.400671421698864, 5.196496334593458, 10.755661317172377]
    var = [0.5190013868567017, 4.714107400797064, 9.021789737166483, 15.883258486743134]
    print(var)
    #var = [0.5190013868567017, 0.07556557282991373, 0.4716822586090975, 0.7244937043066909] 
    tree.set_var(var)
    print(tree.likelihood())
    #var = [abs(a) for a in np.random.normal(size=(4,))]
    print(var)
    tree.set_var(var)
    sl = tree.likelihood()
    print('none', sl)
    for i in range(4):
        var2 = list(var)
        var2[i] = 0
        tree.set_var(var2)
        print(i, tree.likelihood())
    for i in range(4):
        var2 = list(var)
        var2[i] -= .1
        tree.set_var(var2)
        nl = tree.likelihood()
        print(i, nl, nl > sl)
    '''
