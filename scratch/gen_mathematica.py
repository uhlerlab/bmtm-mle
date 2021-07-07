import argparse
import random
import subprocess

from scratch.tree import Tree

def write_to_clipboard(output):
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))


def math_format(pijs, rindices, nt):
    keys, values = zip(*sorted(pijs.items()))
    matrix = [['-p{}{}'.format(*sorted((i, j))) for j in range(1, nt)]
        for i in range(1, nt)]
    for i in range(1, nt):
        matrix[i-1][i-1] = ' + '.join(['p{}{}'.format(*sorted((i, j))) 
            for j in range(nt) if i != j])
    
    lines = ['data = {{%s}}' % ', '.join(str(random.randrange(-10, 10)) for _ in range(nt-1))]
    lines.append('SM = Transpose[data] . data')
    lines.append('likematrix[K_] := Log[Det[K]] - Tr[SM . K]')

    pkeys = ', '.join(k+'_' for k in keys)
    pmatrix = '{%s}' % ', '.join('{%s}' % ', '.join(row) for row in matrix)
    lines.append('likep[{}] := likematrix[{}]'.format(
        pkeys,
        pmatrix
    ))

    vs = set()
    for vals in pijs.values():
        vs.update(vals)
    tkeys = ', '.join(v+'_' for v in sorted(vs))
    tvals = ', '.join('*'.join(v) for v in values)
    lines.append('like[{}] := likep[{}]'.format(
        tkeys,
        tvals
    ))

    solvevars = ', '.join('u{}'.format(i) for i in range(len(vs)))
    solvevarsdefn = ', '.join('u{}_'.format(i) for i in range(len(vs)))
    lines.append('Print["start intermediate"]')
    lines.append('intermediate = FullSimplify[Reduce[Grad[like[%s], {%s}] == 0, {%s}]]' % (
        solvevars,
        solvevars,
        solvevars
    ))
    lines.append('Print["finish intermediate"]')

    lines.append('')
    lines.append('')
    for i in range(len(rindices)):
        rlow, rhigh = rindices[i]

        low = 0 if rlow is None else 'Inverse[{}][[{}, {}]]' .format(pmatrix, rlow[0]+1, rlow[1]+1)
        high = 'Inverse[{}][[{}, {}]]' .format(pmatrix, rhigh[0]+1, rhigh[1]+1)
        lines.append('mt{}[{}] := {} - {}'.format(
            i, pkeys, high, low
        ))
        lines.append('tt{}[{}] := mt{}[{}]'.format(i, tkeys, i, tvals))

    lines.append('isgood[{}] := ({})'.format(
        solvevarsdefn,
        ' && '.join('tt{}[{}] >= 0'.format(i, solvevars) for i in range(len(rindices)))))

    lines.append('')
    lines.append('')

    lines.append('solvels = Values[N[Solve[intermediate, {%s}, Reals]]]' % solvevars)
    lines.append('Sum[isgood @@ solvels[[i]], {i, 1, Length[solvels]}]')

    return '\n'.join(lines)


if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--structure', type=str, default='0-0-0')
    parser.add_argument('--eliminate', type=int, default=1)
    parser.add_argument('--restrictions', type=int, default=1)
    args = parser.parse_args()

    tree = Tree()  
    structure = list(map(int, args.structure.split('-')))
    tree.make_prefix(structure)

    count = tree.num_leaf_nodes()+1
    for l in tree.latents():
        l.indentity = count
        count += 1

    pijs = {'p{}{}'.format(i+1, j+1) : {
        't{}'.format(i+1),
        't{}'.format(j+1),
        't{}'.format(li.lca(lj).indentity)
        } 
        for i, li in enumerate(tree.leaves())
            for j, lj in enumerate(tree.leaves()) if i < j}
    
    for i in range(tree.num_leaf_nodes()):
        pijs['p0{}'.format(i+1)] = {'t{}'.format(i+1)}

    if args.eliminate == 1:
        vs = set()
        for vals in pijs.values():
            vs.update(vals)
        
        for v in vs:
            candidates = vs - {v}
            for vals in pijs.values():
                if v in vals:
                    candidates = candidates & vals
                if len(candidates) == 0:
                    break
            if len(candidates) != 0:
                for k in pijs:
                    if v in pijs[k]:
                        pijs[k] = pijs[k] - candidates
    
    for i, li in enumerate(tree.leaves()):
        for j, lj in enumerate(tree.leaves()):
            if i < j:
                li.lca(lj).index = (i, j)
    
    for i, li in enumerate(tree.leaves()):
        li.index = (i, i)
    
    rindices = []
    if args.restrictions == 1:
        rindices = [(None if node.parent is None else node.parent.index, node.index) for node in tree.nodes()]


    #print(pijs)
    out = math_format(pijs, rindices, tree.num_leaf_nodes()+1)
    print(out)
    write_to_clipboard(out)
    
