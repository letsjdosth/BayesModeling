from math import exp, log, lgamma
from random import seed, normalvariate
from functools import partial

import numpy as np

from bayesian_tools.MCMC_Core import MCMC_MH, MCMC_Diag
from special_dist_sampler.sampler_gamma import Sampler_univariate_Gamma


#(t,y)
data = [(94.3, 5), (15.7, 1), (62.9, 5), (126, 14), (5.24, 3),
        (31.4, 19), (1.05, 1), (1.05, 1), (2.1, 4), (10.5, 22)]

def log_proposal_pdf(from_smpl, to_smpl):
    #symmetric
    return 0 

def proposal_sampler(from_smpl, sigma2_alpha, sigma2_beta):
    proposal_alpha = normalvariate(from_smpl[0], sigma2_alpha**0.5)
    proposal_beta = normalvariate(from_smpl[0], sigma2_beta**0.5)
    return np.array([proposal_alpha, proposal_beta])

def log_target_pdf(eval_pt):
    alpha_bar = eval_pt[0]
    beta_bar = eval_pt[1]

    log_target_pdf_val = beta_bar*(-0.9) - exp(alpha_bar) - exp(beta_bar) + alpha_bar + beta_bar
    for (t,y) in data:
        term = lgamma(exp(alpha_bar)+y) - lgamma(exp(alpha_bar)) + beta_bar*exp(alpha_bar) - (y+exp(alpha_bar))*log(exp(beta_bar)+t)
        log_target_pdf_val +=term
    return log_target_pdf_val

proposal_sampler_with_sigma2 = partial(proposal_sampler, sigma2_alpha=0.1, sigma2_beta=0.1)

if __name__=="__main__":
    seed(20220425)
    initial = [0, 0]
    mc_mh_inst1 = MCMC_MH(log_target_pdf, log_proposal_pdf, proposal_sampler_with_sigma2, initial)
    mc_mh_inst1.generate_samples(100000, print_iter_cycle=25000)
    alpha_beta_vec = [[exp(sample[0]),exp(sample[1])] for sample in mc_mh_inst1.MC_sample]

    diag_inst1 = MCMC_Diag()
    diag_inst1.set_mc_samples_from_list(alpha_beta_vec)
    diag_inst1.set_variable_names(["alpha","beta"])
    diag_inst1.burnin(5000)
    diag_inst1.thinning(10)
    diag_inst1.show_traceplot((1,2))
    diag_inst1.show_acf(30, (1,2))
    diag_inst1.show_hist((1,2))
    diag_inst1.print_summaries(5)
    diag_inst1.show_scatterplot(0, 1)

    gamma_sampler_inst = Sampler_univariate_Gamma()
    alpha_beta_theta1_vec = [[sample[0], sample[1], gamma_sampler_inst.sampler(sample[0]+data[0][1], sample[1]+data[0][1])] for sample in alpha_beta_vec]
    diag_inst2 = MCMC_Diag()
    diag_inst2.set_mc_samples_from_list(alpha_beta_theta1_vec)
    diag_inst2.set_variable_names(["alpha","beta","theta1"])    
    diag_inst2.show_scatterplot(2,0)
    diag_inst2.show_scatterplot(2,1)
