import multiprocessing as mp
import random as r
import numpy as np

def f(_): return np.random.randint(1000), r.randint(0,999)

r.seed("1234")

if __name__ == '__main__':
    with mp.Pool(4) as p:
        for i, (np_v, py_v) in enumerate(p.map(f, range(50))):
            if not i & 0b11: print('-' * 23)
            print(f'{i:2}: {np_v=:3} | {py_v=:3}')

# it looks like it is alright to use random with multiprocessing even when seeded
