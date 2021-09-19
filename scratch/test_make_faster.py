from scratch.tree import bhv_distance_owens, Tree
from scratch.solver import Solver

if __name__ == '__main__':
    print('Running tests...')

    tree = Tree()
    tree.make_prefix([2, 0, 0, 2, 0, 0, 8, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0])
    tree.set_data([1, 2, 3, 4, -5, -4, -10, 20, 13, 15, 16, 2, 22, 45, -3])

    solver = Solver()
    solver.predict_mle(tree)

    res = solver.mem[tree.hash()]
    for k, v in sorted(res.items()):
        print(k, v*abs(k)/1000)
