from random import seed

import numpy as np

from bayesian_tools.MCMC_Core import MCMC_base
from bayesian_tools.sampler_gamma import Sampler_univariate_InvChisq

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
    

if __name__=="__main__":
    test_x = np.array([[1,x*0.5] for x in range(30)])
    from random import normalvariate
    test_y = test_x[:,0]*2 + test_x[:,1]*1.3 + np.array([normalvariate(0,0.1) for _ in test_x])
    print(test_y)

    lm_inst = LM_noninfo_prior(test_y, test_x, 20220519)
    lm_inst.generate_samples(10000)
    lm_inst.print_freqentist_result()
    
    from MCMC_Core import MCMC_Diag
    diag_inst = MCMC_Diag()
    diag_inst.set_mc_sample_from_MCMC_instance(lm_inst)
    diag_inst.set_variable_names(["sigma2", "beta0", "beta1"])
    diag_inst.print_summaries(round=8)
    diag_inst.show_hist((1,3))
    diag_inst.show_scatterplot(1,2)
    