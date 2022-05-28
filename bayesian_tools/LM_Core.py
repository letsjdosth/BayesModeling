import csv
import time
from random import seed

import numpy as np

class LM_base():
    def __init__(self, response_vec, design_matrix, rnd_seed=None) -> None:
        self.x = design_matrix
        self.y = response_vec

        self.num_data = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1]

        self.MC_sample = []

        if rnd_seed:
            seed(rnd_seed)
        self.np_rng = np.random.default_rng()

        self.xtx = np.transpose(self.x) @ self.x
        self.xty = np.transpose(self.x) @ self.y
    
    def deep_copier(self, x_iterable) -> list:
        rep_x_list = []
        for x in x_iterable:
            try:
                _ = iter(x)
                rep_x_list.append(self.deep_copier(x))
            except TypeError:
                rep_x_list.append(x)
        return rep_x_list

    def sampler(self, **kwargs):
        pass

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler(iter_idx=i)
            
            if i==100 and verbose:
                elap_time_head_iter = time.time()-start_time
                estimated_time = (num_samples/100)*elap_time_head_iter
                print("estimated running time: ", estimated_time//60, "min ", estimated_time%60, "sec")

            if i%print_iter_cycle == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%print_iter_cycle == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        
        if pid is not None and verbose:
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")

    def write_samples(self, filename: str):
        with open(filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample
                writer.writerow(csv_row)
