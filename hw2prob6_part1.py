from math import exp, log, lgamma
from random import seed, normalvariate
from functools import partial

import numpy as np

from special_dist_sampler.sampler_gamma import Sampler_univariate_Gamma
from bayesian_tools.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag

#(t,y)
data = [(94.3, 5), (15.7, 1), (62.9, 5), (126, 14), (5.24, 3),
        (31.4, 19), (1.05, 1), (1.05, 1), (2.1, 4), (10.5, 22)]

class Gibbs_withoutMarginalize_HW2P6(MCMC_Gibbs):
    def __init__(self, initial):
        #param: [alpha, beta, [theta1,...,theta10]]
        super().__init__(initial)
        self.gamma_sampler = Sampler_univariate_Gamma()
    
    
    def full_conditional_sampler_alpha_beta(self, last_param):
        new = [last_param[i] for i in range(2)] + [[x for x in last_param[2]]]
        #param: [alpha, beta, [theta1,...,theta10]]
        #update new
        last_alpha_bar = log(new[0])
        last_beta_bar = log(new[1])

        def log_proposal_pdf(from_smpl, to_smpl):
            #symmetric
            return 0 

        def proposal_sampler(from_smpl, sigma2_alpha, sigma2_beta):
            proposal_alpha_bar = normalvariate(from_smpl[0], sigma2_alpha**0.5)
            proposal_beta_bar = normalvariate(from_smpl[0], sigma2_beta**0.5)
            return np.array([proposal_alpha_bar, proposal_beta_bar])

        def log_target_pdf(eval_pt, theta_vec):
            alpha_bar = eval_pt[0]
            beta_bar = eval_pt[1]
            log_target_pdf_val = beta_bar*(10*exp(alpha_bar)-0.9) - 10*lgamma(exp(alpha_bar))-exp(alpha_bar)-exp(beta_bar)+alpha_bar+beta_bar
            for theta in theta_vec:
                log_target_pdf_val += ((exp(alpha_bar)-1)*log(theta) - exp(beta_bar)*theta)
            return log_target_pdf_val

        proposal_sampler_with_sigma2 = partial(proposal_sampler, sigma2_alpha=0.1, sigma2_beta=0.1)
        log_target_pdf_with_theta = partial(log_target_pdf, theta_vec=new[2])
        mc_mh_inst = MCMC_MH(log_target_pdf_with_theta, log_proposal_pdf, proposal_sampler_with_sigma2, [last_alpha_bar, last_beta_bar])
        mc_mh_inst.generate_samples(2, verbose=False)
        new_alpha_bar_beta__bar_vec = mc_mh_inst.MC_sample[-1]
        new[0] = exp(new_alpha_bar_beta__bar_vec[0])
        new[1] = exp(new_alpha_bar_beta__bar_vec[1])
        return new

    def full_conditional_sampler_thetas(self, last_param):
        new = [last_param[i] for i in range(2)] + [[x for x in last_param[2]]]
        alpha = new[0]
        beta = new[1]
        theta_vec = []
        for (t,y) in data:
            theta_vec.append(self.gamma_sampler.sampler(alpha+y, beta+t))
        new[2] = theta_vec
        return new

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = [last[i] for i in range(2)] + [[x for x in last[2]]]
        #param: [alpha, beta, [theta1,...,theta10]]
        #update new
        new = self.full_conditional_sampler_alpha_beta(new)
        new = self.full_conditional_sampler_thetas(new)
        self.MC_sample.append(new)

if __name__=="__main__":
    seed(20220425)
    initial = [1, 1, [1 for _ in range(10)]]
    wo_marginal_gibbs_inst1 = Gibbs_withoutMarginalize_HW2P6(initial)
    wo_marginal_gibbs_inst1.generate_samples(100000, print_iter_cycle=25000)
    wo_marginal_alpha_beta_vec1 = [sample[:-1] for sample in wo_marginal_gibbs_inst1.MC_sample]
    wo_marginal_alpha_beta_theta0_vec1 = [sample[:-1]+[sample[2][0]] for sample in wo_marginal_gibbs_inst1.MC_sample]
    wo_marginal_theta_vec1 = [sample[-1] for sample in wo_marginal_gibbs_inst1.MC_sample]

    wo_marginal_diag_alphabeta_inst1 = MCMC_Diag()
    wo_marginal_diag_alphabeta_inst1.set_mc_samples_from_list(wo_marginal_alpha_beta_vec1)
    wo_marginal_diag_alphabeta_inst1.set_variable_names(["alpha","beta"])
    wo_marginal_diag_alphabeta_inst1.burnin(5000)
    wo_marginal_diag_alphabeta_inst1.thinning(10)
    wo_marginal_diag_alphabeta_inst1.show_traceplot((1,2), [0,1])
    wo_marginal_diag_alphabeta_inst1.show_acf(30, (1,2), [0,1])
    wo_marginal_diag_alphabeta_inst1.show_hist((1,2), [0,1])
    wo_marginal_diag_alphabeta_inst1.print_summaries(5)

    wo_marginal_diag_alphabetatheta1_inst1 = MCMC_Diag()
    wo_marginal_diag_alphabetatheta1_inst1.set_mc_samples_from_list(wo_marginal_alpha_beta_theta0_vec1)
    wo_marginal_diag_alphabetatheta1_inst1.set_variable_names(["alpha","beta", "theta1"])
    wo_marginal_diag_alphabetatheta1_inst1.burnin(5000)
    wo_marginal_diag_alphabetatheta1_inst1.thinning(10)
    wo_marginal_diag_alphabetatheta1_inst1.show_scatterplot(0,1)
    wo_marginal_diag_alphabetatheta1_inst1.show_scatterplot(2,0)
    wo_marginal_diag_alphabetatheta1_inst1.show_scatterplot(2,1)
    
    wo_marginal_diag_thetas_inst1 = MCMC_Diag()
    wo_marginal_diag_thetas_inst1.set_mc_samples_from_list(wo_marginal_theta_vec1)
    wo_marginal_diag_thetas_inst1.set_variable_names(["theta"+str(i) for i in range(1,11)])
    wo_marginal_diag_thetas_inst1.burnin(5000)
    wo_marginal_diag_thetas_inst1.thinning(10)
    wo_marginal_diag_thetas_inst1.show_traceplot((3,4))
    wo_marginal_diag_thetas_inst1.show_acf(30, (3,4))
    wo_marginal_diag_thetas_inst1.show_hist((3,4))