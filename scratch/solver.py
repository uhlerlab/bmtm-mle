'''
General solver algorithm for 1-sample MLE for BMTM
Also includes a star-tree-specific algorithm
'''
import math

from scratch.tree import Tree

# margin of error for floating point equality
FLOATING_POINT_EPS = 1e-10

# Empirically proven star solution 
# Special case of general solver
# O(n^2)
def var_prediction_star(data):
    line = sorted(data+[0])
    scores = sorted([(math.prod(1/math.sqrt(abs(a-b)) for b in line if a != b), a) for a in line])
    parent = scores[-1][1]
    return [(parent)**2] + [(a-parent)**2 for a in data]

# High level pseudo code
# for given latent node, let's store the scores over all possible children
# for a new latent node, we can recompute the scores as the multiplication of two products
# first product: for each unaffected tree, either:
#     1) loop over all center values and find the min when multiplied with new center
#     2) ``remove'' current root and let children directly connect with parent -- recursively find best value 
# second product: for affected tree, return the center value of root latent with current center
# all observed nodes start at one

# We call our number of observed nodes n

# A note on tree structures:
# We restrict ourselves to trees where all latent nodes have more than one child (either latent or observed)
# The removal of any such latent-one-child nodes results in an equivalent tree
# Thus, we have that there are O(n) latent nodes, and so, O(n) total nodes in our tree

# Initial Analysis:
# We at each depth, we consider O(n) centers. For each center, we require O(n) time for each subdepth to check against subcenters.
# This process is O(sd) where s is the depth away from the bottom of the tree. Thus, this process is O(dn) = O(n^2).
# Thus, overall, we spend O(n^3) at each layer, leading us to a runtime of O(dn^3) = O(n^4).

class Solver:
    def __init__(self):
        # indexed by latent node and then parent/center value
        self.mem = {}

    def get_stored(self, node):
        c_hash = node.hash()
        if c_hash in self.mem:
            return self.mem[c_hash]
        if len(node.children) == 0:
            self.mem[c_hash] = {node.data: 1}
        else:
            self.solve(node)
        assert(c_hash in self.mem)
        return self.mem[c_hash]

    def best_subtree_value(self, tree, top_center):
        candidate = min(abs(top_center-lower_center)*value for lower_center, value in self.get_stored(tree).items())
        if len(tree.children) != 0: # try removing current node
            other = math.prod(self.best_subtree_value(c, top_center) for c in tree.children)
            return min(other, candidate)
        return candidate

    def set_optimal_vars(self, tree, top_center):
        best_val = self.best_subtree_value(tree, top_center)
        candidate = min((abs(top_center-lower_center)*value, lower_center) for lower_center, value in self.get_stored(tree).items())

        #print(top_center, best_val, candidate)

        if candidate[0]-FLOATING_POINT_EPS > best_val:
            tree.above_var = 0
            for c in tree.children:
                self.set_optimal_vars(c, top_center)
        else:
            #print('var steppin', (top_center - candidate[1])**2)
            tree.above_var = (top_center - candidate[1])**2
            for c in tree.children:
                self.set_optimal_vars(c, candidate[1])

    def solve(self, tree):
        current_hash = tree.hash()
        self.mem[current_hash] = {}
        for c in tree.children: 
            this_tree_data = self.get_stored(c)
            for i in range(c.num_leaf_nodes()):
                l = c.get_leaf(i)
                d = l.data
                self.mem[current_hash][d] = this_tree_data[d] # first product
                for other in tree.children:
                    if other is not c:
                        self.mem[current_hash][d] *= self.best_subtree_value(other, d)

    def predict_mle(self, tree):
        self.mem = {}
        self.solve(tree)

        # Believe unnecessary except for debugging
        top_hash = tree.hash()
        self.mem[tuple()] = {0: self.best_subtree_value(tree, 0)}

        self.set_optimal_vars(tree, 0)