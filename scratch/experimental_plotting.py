import math

import matplotlib.pyplot as plt

def ev(xs, x):
    return math.prod(abs(a-x) for a in xs if a != x)

xs = [0, 10, 20]
p = 0
record = []
xaxis = list(range(-1000, 1000)) 
for i in xaxis:
    xp = i / 100
    nx = xs[:] + [xp]
    record.append(ev(xs, xp)/ev(xs, 0))

plt.plot(xaxis, record)
plt.plot(xaxis, [1]*len(xaxis))
plt.show()