import time

import numpy as np

from sugc import count_npoint


def benchmark():
    n_gal = 100000
    box_size = 500.0
    coords = np.random.uniform(0, box_size, (n_gal, 3)).astype(np.float64)
    partition_ids = np.random.randint(0, 8, n_gal).astype(np.int32)
    r_bins = np.logspace(-1, 1, 11).astype(np.float64) # 0.1 to 10 Mpc/h
    
    print(f"Benchmarking count_npoint with {n_gal} galaxies, box_size={box_size}")
    
    for n in [2, 3, 4]:
        start = time.perf_counter()
        t_by_s, t_total = count_npoint(coords, partition_ids, r_bins, box_size, n)
        end = time.perf_counter()
        print(f"N={n}: {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark()
