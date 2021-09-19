from random import sample
from scratch.tree import bhv_distance_owens, Tree
from scratch.reconstruct import compute_ls, sample_cov, compute_linear_tree, compute_zero_linear_tree
from scratch.util import fr_norm_sq

if __name__ == '__main__':
    print('Running tests...')

    pre = [4, 2, 0, 0, 0, 2, 0, 0]
    tree = Tree()
    tree.make_prefix(pre)
    print(tree.get_prefix())
    assert(pre == tree.get_prefix())

    var = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tree.set_var(var)
    print(tree.get_var())
    assert(var == tree.get_var())

    data = [1, 2, 3, 4, 5]
    tree.set_data(data)
    print(tree.get_data())
    assert(data == tree.get_data())

    labels = [1, 2, 3, 4, 5]
    tree.set_labels(labels)
    print(tree.get_labels())
    assert(labels == tree.get_labels())

    data = [1, 2, 3]
    var = [1, 2, 3, 4]
    t1 = Tree()
    t1.make_prefix([0, 0, 0])
    t1.set_labels(data)
    t1.set_data(data)
    t1.set_var(var)

    t2 = Tree()
    t2.make_prefix([0, 0, 0])
    t2.set_labels(data)
    t2.set_data(data)
    t2.set_var([0]+var[1:])


    print(t1.newick(), t2.newick())

    assert(bhv_distance_owens(t1, t2) == 1)

    ground_truth = {((-100, 2, 3), (1,)), ((-100, 1, 3), (2,)), ((-100, 1, 2), (3,)), ((-100,), (1, 2, 3))}
    assert(set(t1.get_splits()) == ground_truth)
    assert(set(t2.get_splits()) == (ground_truth - {((-100,), (1, 2, 3))}))
    assert(set(tree.get_splits()) == {((-100, 2, 3, 4, 5), (1,)), 
        ((-100, 1, 3, 4, 5), (2,)), ((-100, 3, 4, 5), (1, 2)), 
        ((-100, 1, 2, 4, 5), (3,)), ((-100, 1, 2, 3), (4, 5)), 
        ((-100, 1, 2, 3, 5), (4,)), ((-100, 1, 2, 3, 4), (5,)), 
        ((-100, 4, 5), (1, 2, 3)), ((-100,), (1, 2, 3, 4, 5))})

    def test_ls(data):
        cvx_cov = compute_ls(data, [0, 0, 0]).cov_matrix()

        lstree = Tree()
        lstree.make_prefix([0, 0, 0])
        lstree.set_data(data)
        lstree.fr_proj()
        opt_cov = lstree.cov_matrix()

        sample_cov = lstree._build_matrices()[1]
        print('opt', opt_cov)
        print('cvx', cvx_cov)
        print('sample', sample_cov)
        print('opt, cvx errors', fr_norm_sq(opt_cov, sample_cov), fr_norm_sq(cvx_cov, sample_cov))

        assert(all(abs(cvx_cov[i][j] - opt_cov[i][j]) < 1e-3
            for i in range(len(opt_cov)) 
                for j in range(len(opt_cov))))
    
    test_ls([-3, 1, 5, -2])
    test_ls([1, 2, 3, 4])
    test_ls([1, 6, 3, 4])

    import numpy as np
    data = [-4, 5, 6]
    min_cov_matrix = np.array(compute_linear_tree(data, min(data+[0])))
    zero_tree = compute_zero_linear_tree(data)
    zero_cov_matrix = np.array(zero_tree.cov_matrix())
    print('min', min_cov_matrix)
    print('zero', zero_cov_matrix)
    print('min, zero rank', 
        np.linalg.matrix_rank(min_cov_matrix),
        np.linalg.matrix_rank(zero_cov_matrix))

    print("Passed all!")