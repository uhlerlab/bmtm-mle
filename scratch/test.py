from scratch.tree import bhv_distance_owens, Tree

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

    labels = list(map(str, [1, 2, 3, 4, 5]))
    tree.set_labels(labels)
    print(tree.get_labels())
    assert(labels == tree.get_labels())

    data = [1, 2, 3]
    var = [1, 2, 3, 4]
    t1 = Tree()
    t1.make_prefix([0, 0, 0])
    t1.set_data(data)
    t1.set_var(var)

    t2 = Tree()
    t2.make_prefix([0, 0, 0])
    t2.set_data(data)
    t2.set_var([0]+var[1:])

    print(t1.newick(), t2.newick())

    assert(bhv_distance_owens(t1, t2) == 1)

    ground_truth = {((1,), (2, 3)), ((1, 3), (2,)), ((1, 2), (3,)), (tuple(), (1, 2, 3))}
    assert(set(t1.get_splits()) == ground_truth)
    assert(set(t2.get_splits()) == (ground_truth - {(tuple(), (1, 2, 3))}))
    assert(set(tree.get_splits()) == {((1,), (2, 3, 4, 5)), 
        ((1, 3, 4, 5), (2,)), ((1, 2), (3, 4, 5)), 
        ((1, 2, 4, 5), (3,)), ((1, 2, 3), (4, 5)), 
        ((1, 2, 3, 5), (4,)), ((1, 2, 3, 4), (5,)), 
        ((1, 2, 3), (4, 5)), ((), (1, 2, 3, 4, 5))})
