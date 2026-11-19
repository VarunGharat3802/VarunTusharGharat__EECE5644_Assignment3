import numpy as np
from scipy.stats import multivariate_normal as mvn
rng=np.random.default_rng(0)
V=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
covs=[np.array([[1.0,0.3,0.1],[0.3,0.8,0.0],[0.1,0.0,0.6]]),
      np.array([[0.7,-0.2,0.0],[-0.2,1.1,0.25],[0.0,0.25,0.9]]),
      np.array([[1.2,0.0,-0.3],[0.0,0.6,0.1],[-0.3,0.1,0.9]]),
      np.array([[0.8,0.25,0.25],[0.25,0.8,0.25],[0.25,0.25,0.8]])]
for a in [0.8,0.9,1.0,1.1,1.2,1.3]:
    M=a*V; N=100000
    y=rng.integers(0,4,N); X=np.empty((N,3))
    for c in range(4): X[y==c]=rng.multivariate_normal(M[c],covs[c],(y==c).sum())
    L=np.column_stack([mvn(M[c],covs[c]).logpdf(X) for c in range(4)])
    print(a,(L.argmax(1)!=y).mean())
