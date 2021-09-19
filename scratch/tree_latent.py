from scratch.tree import *


class LTree(Tree):
    def set_other_var(self, vars):
        return self._set_at_leaves('other_var', vars)
    def get_other_var(self):
        return self._get_at_leaves('other_var')

    def mle_size(self):
        return self.num()+self.num_leaf_nodes()-1
    
    def mle_set(self, v):
        var = list(v[:])
        #var[2] = 0.99
        count = self.set_var([0]+var)
        count2 = self.set_other_var(var[count-1:])

        '''if all(a < 1 and a > 0 for a in var):
            matrix = np.linalg.inv(tree._build_matrices()[0])
            if any(matrix[i][j] > 0 for i in range(len(matrix)) for j in range(i+1, len(matrix))):
                print(var, matrix)
                raise ValueError('Matrix problem')
            if any(sum(matrix[i][j] for i in range(len(matrix))) < 0 for j in range(len(matrix))):
                print(var, matrix)
                raise ValueError('Not a diagonally dominant problem')'''
        return count+count2-1
    
    def mult_up(self, up):
        prod = 1
        track = self
        while track != up and track is not None:
            prod *= track.above_var
            track = track.parent
        return prod

    def covar(self, other):
        lca = self.lca(other)
        return self.mult_up(lca)*other.mult_up(lca)
    
    def _build_matrices(self):
        num_leaf = self.num_leaf_nodes()

        s_arr = [[a.data*b.data for b in self.leaves()] for a in self.leaves()]
        x_arr = [a.data for a in self.leaves()]

        k_arr = np.array([[a.covar(b) for b in self.leaves()] for a in self.leaves()])
        try:
            others = [math.sqrt(a.other_var) for a in self.leaves()]
        except:
            print([a.other_var for a in self.leaves()])
            raise ValueError('error')
        others_diag_arr = np.array([[0 if i != j else others[i] for j in range(num_leaf)] 
            for i in range(num_leaf)])
        m_arr = np.matmul(np.matmul(others_diag_arr, k_arr), others_diag_arr)

        print(others)
        print(k_arr)
        print(m_arr)

        assert(all(abs(m_arr[i][j]-m_arr[j][i]) < 1e-8 for i in range(len(m_arr)) for j in range(len(m_arr))))
        assert(all(s_arr[i][j] == s_arr[j][i] for i in range(len(s_arr)) for j in range(len(s_arr))))
        
        return m_arr, s_arr, x_arr

if __name__ == '__main__':
    tree = LTree()
    tree.make_prefix([0, 0, 0, 0])
    tree.set_data([1, 4, -2, -3])

    '''tree.mle_set([1.44259281e+01, 1.22347954e+00, 8.79944968e-01, 1.84144197e+00,
        1.16979181e-03, 2.28014920e-06, 1.99997229e+01])
    print(np.linalg.inv(tree._build_matrices()[0]))
    print(tree.likelihood())

    tree.mle_set([6.77409675e+00, 4.67341986e+00, 7.22465752e-01, 4.87753059e+00,
        3.74111075e-05, 5.05802221e-04, 1.99785501e+01])
    print(np.linalg.inv(tree._build_matrices()[0]))
    print(tree.likelihood())

    tree.mle_set([1.44259281e+01, 1.22347954e+00, 8.79944968e-01, 1.84144197e+00,
        1.16979181e-03, 2.28014920e-11, 1.99997229e+01])
    print(tree.likelihood())

    tree.mle_set([1.05032405e+00, 8.87207773e+00, 5.71110359e+00, 7.14760127e-01,
        7.16310578e+00, 1.81234814e+01, 6.47711000e-03])
    print(tree.likelihood())'''

    #tree.mle_set([0.76869849, 0.01424754, 0.67041767, 1.49150532, 2.49451865, 5.70094544])
    #print(np.linalg.inv(tree._build_matrices()[0]))

    #tree.mle_set([(1e-15), 1-(1e-15), 1, 4])
    #tree.mle_set([1, 0, 1, 4])
    tree.mle_set([0.999, 0.999, 0, 0, 1, 16, 4, 9])
    print(tree._build_matrices()[0])
    print(tree.likelihood())

    '''print(tree.mle(method='dual_annealing', 
        max_var=20, 
        accept=-1000, maxiter=5000,
        global_param_space=([(0, 1)]*2+[(0, 20)]*2)))
    print(tree.likelihood())
    print(np.linalg.inv(tree._build_matrices()[0]))
    '''