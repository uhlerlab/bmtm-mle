import numpy as np

import causaldag as cd

variances = np.random.uniform(size=4)
g = cd.GaussDAG(
    nodes=[0, 1, 2, 3], 
    arcs={(0, 1), (0, 2), (0, 3)}, 
    variances=variances
)
cov = g.covariance
cov_obs = cov[np.ix_([1, 2, 3], [1, 2, 3])]
prec = np.linalg.inv(cov)
prec_obs = np.linalg.inv(cov_obs)