from scipy.stats import norm
import os
import pickle
import numpy as np


if __name__ == "__main__":
    xs = []
    cache_file = "label_frac_histogram.cache"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            Xs = pickle.load(fp)
            
    mus = []
    for i in Xs.keys():
        (mu, sigma) = norm.fit(Xs[i])
        mus.append(mu)
    print(np.mean(mus))
    

