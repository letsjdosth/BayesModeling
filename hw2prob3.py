from random import seed, betavariate, choices
from statistics import mean
import numpy as np
from scipy.special import loggamma

from bayesian_tools.MCMC_Core import MCMC_base, MCMC_Diag
from bayesian_tools.grid_approximation import GridApprox2D

class CondSampler_hw2p3(MCMC_base):
    def __init__(self, random_seed):
        self.observations = [(16,58), (9,90), (10,48), (13,57), (19,103), (20,57), (18,86), (17,112), (35,273), (55,64)]
        self.y_for_each_j = [obs[0] for obs in self.observations]
        self.n_for_each_j = [sum(obs) for obs in self.observations]

        self.MC_sample = []

        self.random_seed = random_seed
        seed(random_seed)
        
    def joint_posterior_alpha_beta_sampler(self, num_sample):
        grid_inst = GridApprox2D(0.001, 10, 2000, 0.001, 40, 2000, self.random_seed)
        def posterior_joint_alpha_beta(alpha_meshgrid, beta_meshgrid):
            term1 = 10 * (loggamma(alpha_meshgrid + beta_meshgrid) - loggamma(alpha_meshgrid) - loggamma(beta_meshgrid))
            term2 = -5/2 * np.log(alpha_meshgrid + beta_meshgrid)
            term3 = np.zeros(alpha_meshgrid.shape)
            for y, n in zip(self.y_for_each_j, self.n_for_each_j):
                term3 += (loggamma(alpha_meshgrid+y)+loggamma(beta_meshgrid+(n-y))-loggamma(alpha_meshgrid+beta_meshgrid+n))
            return np.exp(term1 + term2 + term3)
        grid_inst.make_level_matrix_on_grid_by_numpy_oper(posterior_joint_alpha_beta)
        grid_inst.set_variable_names(["alpha","beta"])
        grid_inst.show_contourplot(levels=20)
        alpha_beta_samples = grid_inst.sampler(num_sample)
        return alpha_beta_samples

    def sampler(self, **kwargs):
        alpha, beta = self.alpha_beta_posterior_samples[kwargs["iter_idx"]]
        new_sample = [alpha, beta]
        for y, ny in self.observations:
            new_sample.append(betavariate(alpha+y, beta+ny))
        self.MC_sample.append(new_sample)

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        self.alpha_beta_posterior_samples = self.joint_posterior_alpha_beta_sampler(num_samples)
        super().generate_samples(num_samples, pid, verbose, print_iter_cycle)


if __name__=="__main__":
    sampler_inst = CondSampler_hw2p3(20220417)
    # sampler_inst.joint_posterior_alpha_beta_sampler()
    sampler_inst.generate_samples(50000, print_iter_cycle=5000)
    diag_inst = MCMC_Diag()
    diag_inst.set_mc_sample_from_MCMC_instance(sampler_inst)
    diag_inst.set_variable_names(["alpha","beta"]+["theta"+str(i) for i in range(1,11)])
    
    #3.8.c/d
    print("naive est")
    print([round(y/n,5) for y,n in zip(sampler_inst.y_for_each_j, sampler_inst.n_for_each_j)])
    diag_inst.print_summaries(5)
    diag_inst.show_hist((3,4))

    #3.8.e
    posterior_alphas = diag_inst.get_specific_dim_samples(0)
    posterior_betas = diag_inst.get_specific_dim_samples(1)
    posterior_thetas_without_data = [betavariate(alpha, beta) for alpha, beta in zip(posterior_alphas, posterior_betas)]
    n_sim_total_obs = 100
    predictive_y_under_n100 = []
    for theta in posterior_thetas_without_data:
        predictive_y_under_n100.append([sum(choices([0,1], weights=[1-theta, theta], k=n_sim_total_obs))])
    predictive_inst = MCMC_Diag()
    predictive_inst.set_mc_samples_from_list(predictive_y_under_n100)
    predictive_inst.set_variable_names(["y_new"])
    
    print("theta(for new district, without data) means:", mean(posterior_thetas_without_data))
    predictive_inst.print_summaries(5)
    predictive_inst.show_hist((1,1))
