import numpy as np

def sampling(X , Y , m ,seed =42):
    np.random.seed(seed)
    indices = np.random.choice(len(X), size=m, replace=False)
    return X[indices], Y[indices]
