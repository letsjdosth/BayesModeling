from random import seed, normalvariate

import numpy as np

from MCMC_Core import MCMC_base, MCMC_Gibbs
from sampler_gamma import Sampler_univariate_InvChisq, Sampler_univariate_InvGamma

class LM_noninfo_prior(MCMC_base):
    #prior p(mu, sigma^2) ~ sigma^(-2)
    def __init__(self, response_vec, design_matrix, rnd_seed) -> None:
        self.x = design_matrix
        self.y = response_vec
            
        self.n = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1] #k+1

        self.MC_sample = []

        seed(rnd_seed)
        self.inv_chisq_sampler = Sampler_univariate_InvChisq()
        self.np_rng = np.random.default_rng()

        self.df = self.n - self.dim_beta
        self.xtx_inv = np.linalg.inv(np.transpose(self.x) @ self.x)
        self.beta_hat = self.xtx_inv @ np.transpose(self.x) @ self.y
        self.residual = self.y - (self.x @ self.beta_hat)
        self.s2_without_div_df = np.dot(self.residual, self.residual)

    def print_freqentist_result(self):
        print("beta_hat:", self.beta_hat)
        print("s2_without_div_df:", self.s2_without_div_df)
        print("s2_with_div_df:", self.s2_without_div_df/self.df)
        beta_hat_cov_mat = self.xtx_inv * (self.s2_without_div_df/self.df)
        beta_hat_var_list = [beta_hat_cov_mat[i,i] for i in range(beta_hat_cov_mat.shape[0])]
        print("beta_hat_var:", beta_hat_var_list)

    def sampler(self, **kwarg):
        inv_chisq_sample = self.inv_chisq_sampler.sampler_iter(1, self.df)[0]
        new_sigma = self.s2_without_div_df * inv_chisq_sample
        new_beta = self.np_rng.multivariate_normal(self.beta_hat, self.xtx_inv * new_sigma)
        self.MC_sample.append([new_sigma]+[x for x in new_beta])
    

class LM_random_eff_fixed_slope_noninfo_prior(MCMC_Gibbs):
    #prior p(mu, sigma^2) ~ sigma^(-2)
    def __init__(self, response_vec, design_matrix, rand_eff_group_col_indicator_list, initial, rnd_seed) -> None:
        # rand_eff_group_indicator_list: 1 if the variable is in the group / 0 if not
        #now, this class support only 'one' random effect group.
        self.rand_eff_group = rand_eff_group_col_indicator_list
        self.x = design_matrix
        self.y = response_vec
        self.hyper_tau2_1 = 100
        self.hyper_mu1 = 0


        self.n = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1] #k+1

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
        resid = self.y - self.x@new_sample[0]
        sigma2_rate = np.dot(resid, resid)/2
        new_sigma2 = self.inv_gamma_sampler.sampler(sigma2_shape, sigma2_rate)
        new_sample[1] = new_sigma2
        return new_sample

    def _full_conditional_sampler_mu0(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        mu0_mean = 0
        num_group_member = 0
        for i, ind in enumerate(self.rand_eff_group):
            if ind==1:
                num_group_member += 1
                mu0_mean += new_sample[0][i]
        mu0_mean = mu0_mean/num_group_member
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
        new_tau2 = self.inv_gamma_sampler.sampler(tau2_shape, tau2_rate)
        new_sample[3] = new_tau2
        return new_sample

    def sampler(self, **kwarg):
        last_sample = self.MC_sample[-1]
        new_sample = self._full_conditional_sampler_beta(last_sample)
        new_sample = self._full_conditional_sampler_sigma2(new_sample)
        new_sample = self._full_conditional_sampler_mu0(new_sample)
        new_sample = self._full_conditional_sampler_tau2_0(new_sample)
        self.MC_sample.append(new_sample)
    




if __name__=="__main__":
    # test_x = np.array([[1,x*0.5] for x in range(30)])
    # from random import normalvariate
    # test_y = test_x[:,0]*2 + test_x[:,1]*1.3 + np.array([normalvariate(0,0.1) for _ in test_x])
    # print(test_y)

    # lm_inst = LM_noninfo_prior(test_y, test_x, 20220519)
    # lm_inst.generate_samples(10000)
    # lm_inst.print_freqentist_result()
    
    # from MCMC_Core import MCMC_Diag
    # diag_inst = MCMC_Diag()
    # diag_inst.set_mc_sample_from_MCMC_instance(lm_inst)
    # diag_inst.set_variable_names(["sigma2", "beta0", "beta1"])
    # diag_inst.print_summaries(round=8)
    # diag_inst.show_hist((1,3))
    # diag_inst.show_scatterplot(1,2)
    

    test2_x = np.array([
        [1,0,0,0,1],
        [1,0,0,0,2],
        [1,0,0,0,3],
        [1,0,0,0,4],
        [1,0,0,0,5],
        [1,0,0,0,6],
        [1,0,0,0,7],

        [0,1,0,0,1],
        [0,1,0,0,2],
        [0,1,0,0,3],
        [0,1,0,0,4],
        [0,1,0,0,5],
        [0,1,0,0,6],
        [0,1,0,0,7],

        [0,0,1,0,-1],
        [0,0,1,0,-2],
        [0,0,1,0,-3],
        [0,0,1,0,-4],
        [0,0,1,0,-5],
        [0,0,1,0,-6],
        [0,0,1,0,-7],

        [0,0,0,1,-1],
        [0,0,0,1,-2],
        [0,0,0,1,-3],
        [0,0,0,1,-4],
        [0,0,0,1,-5],
        [0,0,0,1,-6],
        [0,0,0,1,-7],
        ])
    test2_y = test2_x[:,0]*(-1) + test2_x[:,1]*3 + test2_x[:,2]*(-2) + test2_x[:,3]*1 + test2_x[:,4]*1 + np.array([normalvariate(0, 0.4) for _ in test2_x])
    print(test2_y)
    test2_indicator = [1,1,1,1,0]
    #  0       1       2    3
    # [[beta], sigma2, mu0, tau2_0]
    test2_initial = [np.array([0,0,0,0,0]),1, 0, 1]
    lm_inst2 = LM_random_eff_fixed_slope_noninfo_prior(test2_y, test2_x, test2_indicator, test2_initial, 20220519)
    lm_inst2.generate_samples(50000, print_iter_cycle=10000)
    
    
    from MCMC_Core import MCMC_Diag
    diag_inst21 = MCMC_Diag()
    betas2 = [x[0] for x in lm_inst2.MC_sample]
    diag_inst21.set_mc_samples_from_list(betas2)
    diag_inst21.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4"])
    diag_inst21.print_summaries(round=8)
    diag_inst21.show_hist((1,5))
    diag_inst21.show_traceplot((1,5))

    diag_inst22 = MCMC_Diag()
    others2 = [x[1:4] for x in lm_inst2.MC_sample]
    diag_inst22.set_mc_samples_from_list(others2)
    diag_inst22.set_variable_names(["sigma2", "mu0", "tau2_0"])
    diag_inst22.print_summaries(round=8)
    diag_inst22.show_hist((1,3))
    diag_inst22.show_traceplot((1,3))