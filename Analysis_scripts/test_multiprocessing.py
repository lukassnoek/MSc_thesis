__author__ = 'lukas'

def cube(x):
    return x**3

import multiprocessing as mp
pool = mp.Pool(processes=4)

pool = mp.Pool(processes=4)
results = pool.map(cube, range(1,7))
print(results)