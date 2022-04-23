from random import seed
import takehome1_sampler as tk1
from bayesian_tools.MCMC_Core import MCMC_Diag

import matplotlib.pyplot as plt

seed(20220423)
diag_inst_list = []

for i in range(10):
    print("chain:",i)
    initial1 = [0.1*(i+1)-0.05 for _ in range(38)]
    gibbs_inst1 = tk1.MCMC_Gibbs_HModel_TH1(initial1, 100, 100) #vague hyperparam
    gibbs_inst1.generate_samples(30000, print_iter_cycle=5000, pid=i)
    diag_inst1=MCMC_Diag()
    diag_inst1.set_mc_sample_from_MCMC_instance(gibbs_inst1)
    diag_inst1.set_variable_names(["mu"+str(i) for i in range(1,38)]+["d"])
    diag_inst1.burnin(5000)
    diag_inst1.thinning(20)
    
    diag_inst_list.append(diag_inst1)

for diag_inst in diag_inst_list:
    ten_th_mu = diag_inst.get_specific_dim_samples(9) #10-th region
    plt.hist(ten_th_mu, histtype="step", bins=30)
plt.show()

for diag_inst in diag_inst_list:
    d = diag_inst.get_specific_dim_samples(37) #d
    plt.hist(d, histtype="step", bins=30)
plt.show()