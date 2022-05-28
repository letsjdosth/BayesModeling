import csv
import time
from random import seed

import numpy as np

from MCMC_Core import MCMC_base, MCMC_Gibbs
from info_criteria import InfomationCriteria
from sampler_gamma import Sampler_univariate_InvChisq, Sampler_univariate_InvGamma

class LM_base():
    def __init__(self, response_vec, design_matrix, rnd_seed=None) -> None:
        self.x = design_matrix
        self.y = response_vec

        self.n = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1]

        self.MC_sample = []

        if rnd_seed:
            seed(rnd_seed)
        self.np_rng = np.random.default_rng()

        self.xtx = np.transpose(self.x) @ self.x
        self.xty = np.transpose(self.x) @ self.y

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


class LM_normal_prior_known_sigma2(LM_base):
    def __init__(self, response_vec, design_matrix, prior_beta_mean, prior_beta_cov, sigma2, rnd_seed=None) -> None:
        # y|beta,sigma2 ~ N(X*beta, sigma2*I)
        # beta~N(mu0, V0)
        
        super().__init__(response_vec, design_matrix, rnd_seed=rnd_seed)
        self.prior_mu0 = prior_beta_mean
        self.prior_V0 = prior_beta_cov
        
        self.sigma2 = sigma2
        
    def _calculator_for_sampler(self):
        self.prior_V0_inv = np.linalg.inv(self.prior_V0)

        self.posterior_V1_inv = self.prior_V0_inv + self.xtx/self.sigma2
        self.posterior_V1 = np.linalg.inv(self.posterior_V1_inv)
        self.posterior_mu1 = self.posterior_V1 @ (self.prior_V0_inv@self.prior_mu0 + self.xty / self.sigma2)

    def sampler(self, **kwarg):
        new_beta = self.np_rng.multivariate_normal(self.posterior_mu1, self.posterior_V1)
        self.MC_sample.append(new_beta)


class Linear_Mixed_Model(LM_base):
    def __init__(self, response_vec, design_matrix, group_indicator,
                    hyper_mu_phi_list, hyper_tau2_a_list, hyper_tau2_b_list, hyper_sigma_a, hyper_sigma_b, 
                    rnd_seed=None) -> None:
        # y|beta,sigma2 ~ N(X*beta, sigma2*I)
        # beta_{k, of group_j} ~ N(mu_j, tau_j^2)
        # mu_j ~ N(0,hyper_phi_j)
        # tau_j^2 ~ inv.gamma(hyper_tau2_a_{j}, hyper_tau2_b_{j})
        
        # group_indicator: designate categorical column in design mattrix as the same number
        # if there are more than one categorical variable, use 0(start), 1. 2, 3, ...
        # one fixed effect should have its own category


        super().__init__(response_vec, design_matrix, rnd_seed=rnd_seed)
        self.group_indicator = group_indicator
        self.num_randeff_group = max(self.group_indicator)
        
        length_checker = [len(hyper_mu_phi_list), len(hyper_mu_ph)]
        
        self.sigma2 = sigma2
        
    def _calculator_for_sampler(self):
        self.prior_V0_inv = np.linalg.inv(self.prior_V0)

        self.posterior_V1_inv = self.prior_V0_inv + self.xtx/self.sigma2
        self.posterior_V1 = np.linalg.inv(self.posterior_V1_inv)
        self.posterior_mu1 = self.posterior_V1 @ (self.prior_V0_inv@self.prior_mu0 + self.xty / self.sigma2)

    def sampler(self, **kwarg):
        new_beta = self.np_rng.multivariate_normal(self.posterior_mu1, self.posterior_V1)
        self.MC_sample.append(new_beta)


class LM_random_eff_fixed_slope_noninfo_prior(MCMC_Gibbs):
    #prior p(mu, sigma^2) ~ 1*inv_gamma(self.hyper_tau2_0_shape, self.hyper_tau2_0_rate)
    #when 0, it get to be p(mu,sigma^2)~ sigma^(-2)
    #however, improper prior may cause a stuck in posterior
    def __init__(self, response_vec, design_matrix, rand_eff_group_col_indicator_list, initial, rnd_seed) -> None:
        # rand_eff_group_indicator_list: 1 if the variable is in the group / 0 if not
        #now, this class support only 'one' random effect group.
        self.rand_eff_group = rand_eff_group_col_indicator_list
        self.x = design_matrix
        self.y = response_vec
        self.hyper_tau2_1 = 10000
        self.hyper_mu1 = 0
        self.hyper_tau2_0_shape = 0.1 # tune here if needed
        self.hyper_tau2_0_rate = 0.1 # tune here if needed

        self.n = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1]

        self.MC_sample = [initial]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]

        seed(rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma()
        self.np_rng = np.random.default_rng()

        self.xtx = np.transpose(self.x) @ self.x
        self.xtx_inv = np.linalg.inv(self.xtx)
        self.xty = np.transpose(self.x) @ self.y

    def _full_conditional_sampler_beta(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        D_inv_list = []
        m_list = []
        for ind in self.rand_eff_group:
            if ind==0:
                D_inv_list.append(1/self.hyper_tau2_1)
                m_list.append(self.hyper_mu1)
            elif ind==1:
                D_inv_list.append(1/new_sample[3])
                m_list.append(new_sample[2])
            else:
                raise ValueError("check your random effect group indicator list.")
        D_inv = np.diag(D_inv_list)
        m = np.array(m_list)

        beta_precision = self.xtx / new_sample[1] + D_inv
        beta_variance = np.linalg.inv(beta_precision)
        beta_mean = beta_variance @ (self.xty / new_sample[1] + D_inv @ m)
        new_beta = self.np_rng.multivariate_normal(beta_mean, beta_variance)
        new_sample[0] = new_beta
        return new_sample

    def _full_conditional_sampler_sigma2(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        sigma2_shape = self.n/2
        resid = self.y - (self.x@new_sample[0])
        sigma2_rate = np.dot(resid, resid)/2
        new_sigma2 = self.inv_gamma_sampler.sampler(sigma2_shape, sigma2_rate)
        new_sample[1] = new_sigma2
        return new_sample

    def _full_conditional_sampler_mu0(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        mu0_sum = 0
        num_group_member = 0
        for i, ind in enumerate(self.rand_eff_group):
            if ind==1:
                num_group_member += 1
                mu0_sum += new_sample[0][i]
        mu0_mean = mu0_sum/num_group_member
        mu0_var = new_sample[3]/num_group_member
        new_mu0 = normalvariate(mu0_mean, mu0_var**0.5)
        new_sample[2] = new_mu0
        return new_sample

    def _full_conditional_sampler_tau2_0(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        tau2_rate = 0
        num_group_member = 0
        for i, ind in enumerate(self.rand_eff_group):
            if ind==1:
                num_group_member += 1
                tau2_rate += ((new_sample[0][i]-new_sample[2])**2)/2

        tau2_shape = num_group_member/2
        new_tau2 = self.inv_gamma_sampler.sampler(tau2_shape+self.hyper_tau2_0_shape, tau2_rate+self.hyper_tau2_0_rate)
        new_sample[3] = new_tau2
        return new_sample

    def sampler(self, **kwarg):
        last_sample = self.MC_sample[-1]
        new_sample = self._full_conditional_sampler_beta(last_sample)
        new_sample = self._full_conditional_sampler_sigma2(new_sample)
        new_sample = self._full_conditional_sampler_mu0(new_sample)
        new_sample = self._full_conditional_sampler_tau2_0(new_sample)
        self.MC_sample.append(new_sample)
    
