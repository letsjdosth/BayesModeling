import time
import csv

class MCMC_Gibbs:
    def __init__(self, initial):
        self.MC_sample = [initial]

    def gibbs_sampler(self):
        last = self.MC_sample[-1]
        new = [x for x in last] #[nu, theta]
        #update new

        self.MC_sample.append(new)

    
    def full_conditional_sampler(self, last_param):
        new_sample = [x for x in last_param]
        #update new

        return new_sample


    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        start_time = time.time()
        for i in range(1, num_samples):
            self.gibbs_sampler()
            
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


