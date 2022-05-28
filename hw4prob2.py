from bayesian_tools.LM_Core import LM_base
from bayesian_tools.MCMC_Core import MCMC_Diag
from special_dist_sampler.sampler_gamma import Sampler_univariate_InvGamma
import numpy as np

class SimDataFactory:
    #fixed parameters
    def __init__(self) -> None:
        self.n = 20 # number of groups, idx: i
        self.m = 10 # numper of samples in each group, idx: j
        self.N = self.n * self.m #total number of samples
        self.p = 5 # number of coefficients, dix: k
        
        self.sigma2 = 1
        self.mu = [1] + [(i*1 - 2) for i in range(self.p)] # [mu_0, mu_1, ..., mu_5]
        self.tau2 = [1] + [0.5 for _ in range(self.p)] # [tau2_0, tau2_1, ..., tau2_5]

        self._generator()


    def _generator(self):
        np_rng = np.random.default_rng(seed=20220527)
        
        self.X_without_1 = np_rng.normal(0,1, size=(self.N, 5))
        self.X = np.c_[np.ones(self.N), self.X_without_1]

        self.beta = [] #[[beta01,beta11,...,beta51],[beta02,beta12,...,beta52]],...]
        for i in range(self.n):
            beta_i = []
            for mu_k, tau2_k in zip(self.mu, self.tau2):
                beta_i.append(np_rng.normal(mu_k, np.sqrt(tau2_k)))
            self.beta.append(np.array(beta_i))
        
        self.y = []
        self.group_idx = []
        for i in range(self.n):
            for j in range(self.m):
                epsilon_ij = np_rng.normal(0, np.sqrt(self.sigma2))
                self.y_ij = self.X[i*self.m + j,:] @ self.beta[i] + epsilon_ij
                self.y.append(self.y_ij)
                self.group_idx.append(i)
        self.y = np.array(self.y)

        self.intercept_one_hot_coded_mat = np.zeros((self.N, self.n))
        for i, group_idx in enumerate(self.group_idx):
            self.intercept_one_hot_coded_mat[i, group_idx] = 1

        
    def get_beta_for_group_i(self, i):
        # i = 0, 1,..., 19
        return self.beta[i]

    def get_yXi_for_fixed_eff_model(self):
        return (self.X, self.y, self.group_idx)
    
    def get_yXi_for_rand_intercept_model(self):
        rand_int_X = np.c_[self.intercept_one_hot_coded_mat, self.X_without_1]
        return (rand_int_X, self.y, self.group_idx)

    def get_yXi_for_rand_slope_model(self):
        self.x_hot_coded_mat_list = [np.zeros((self.N, self.n)) for _ in range(self.p)]
        for i, group_idx in enumerate(self.group_idx):
            for k, mat in enumerate(self.x_hot_coded_mat_list):
                mat[i, group_idx] = self.X_without_1[i, k]

        rand_slope_X = np.c_[self.intercept_one_hot_coded_mat, np.concatenate(self.x_hot_coded_mat_list, axis=1)]
        return (rand_slope_X, self.y, self.group_idx)

factory = SimDataFactory()
fixed_X, y, group_idx = factory.get_yXi_for_fixed_eff_model()
rand_int_X, y, group_idx = factory.get_yXi_for_rand_intercept_model()
rand_slope_X, y, group_idx = factory.get_yXi_for_rand_slope_model()


class HW4Prob2_Fixed(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.MC_sample = [initial]
        #[sigma2, [beta0, beta1,...,beta5]]
        
        # able to tune these
        self.hyper_mu = [0 for _ in range(6)]
        self.hyper_tau2 = [100 for _ in range(6)]
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        
        self.mu0 = np.array(self.hyper_mu)
        self.V0_inv = np.diag(1/np.array(self.hyper_tau2))
        self.V0_inv_times_mu0 = self.V0_inv @ self.mu0

    def full_conditional_sampler_beta(self, last_param):
        #[sigma2, [beta0, beta1,...,beta5]]
        sigma2 = last_param[0]
        cov_mat = np.linalg.inv(self.V0_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (self.V0_inv_times_mu0 + self.xty/sigma2)
        new_beta = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        new_sample = [sigma2, new_beta]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        #[sigma2, [beta0, beta1,...,beta5]]
        beta = np.array(last_param[1])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ beta
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [new_sigma2, beta]
        return new_sample


    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2(new)
        self.MC_sample.append(new)

fixed_initial = [1,[0,0,0,0,0,0]]
fixed_inst = HW4Prob2_Fixed(y, fixed_X, fixed_initial)
fixed_inst.generate_samples(10000)
# print(fixed_inst.MC_sample[-1])
fixed_beta = [sample[1] for sample in fixed_inst.MC_sample]
fixed_others = [[sample[0]] for sample in fixed_inst.MC_sample]
fixed_diag_inst1 = MCMC_Diag()
fixed_diag_inst1.set_mc_samples_from_list(fixed_beta)
fixed_diag_inst1.set_variable_names(["beta"+str(i) for i in range(6)])
fixed_diag_inst1.show_traceplot((2,3))
fixed_diag_inst1.show_hist((2,3))
fixed_diag_inst1.show_acf(30,(2,3))

fixed_diag_inst2 = MCMC_Diag()
fixed_diag_inst2.set_mc_samples_from_list(fixed_others)
fixed_diag_inst2.set_variable_names(["sigma2"])
fixed_diag_inst2.show_traceplot((1,1))
fixed_diag_inst2.show_hist((1,1))
fixed_diag_inst2.show_acf(30,(1,1))



class HW4Prob2_Rand_Int(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.MC_sample = [initial]
        #[sigma2, [[beta0_1,...,beta0_20], beta1,...,beta5], mu0, tau2_0]
        
        # able to tune these
        self.hyper_mu_except_intercept = [0 for _ in range(5)] #[mu1, mu2, mu3, mu4, mu5]
        self.hyper_tau2_except_intercept = [100 for _ in range(5)]
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        self.hyper_a_tau0 = 0.01
        self.hyper_b_tau0 = 0.01
        self.hyper_phi2_0 = 100

    def full_conditional_sampler_beta(self, last_param):
        #[sigma2, [[beta0_1,...,beta0_20], beta1,...,beta5], mu0, tau2_0]
        sigma2 = last_param[0]
        mu0 = last_param[2]
        tau2_0 = last_param[3]
        m_array = np.array([mu0]*20 + self.hyper_mu_except_intercept)
        D_inv = np.diag(1/np.array([tau2_0]*20 + self.hyper_tau2_except_intercept))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_beta_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        new_beta0 = new_beta_vec[0:20].tolist()
        new_beta_others = new_beta_vec[20:].tolist()
        new_beta_sample_form = [new_beta0]+new_beta_others
        new_sample = [last_param[0], new_beta_sample_form, last_param[2], last_param[3]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        #[sigma2, [[beta0_1,...,beta0_20], beta1,...,beta5], mu0, tau2_0]
        beta_vec = np.array(last_param[1][0]+last_param[1][1:])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ beta_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [new_sigma2, last_param[1], last_param[2], last_param[3]]
        return new_sample

    def full_conditional_sampler_mu0(self, last_param):
        # 0       1                                          2    3
        #[sigma2, [[beta0_1,...,beta0_20], beta1,...,beta5], mu0, tau2_0]
        tau2_0 = last_param[3]
        beta0 = last_param[1][0]
        param_var = 1/(20/tau2_0 + 1/self.hyper_phi2_0)
        param_mean = param_var * (np.sum(beta0) / tau2_0)
        new_mu0 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], new_mu0, last_param[3]]
        return new_sample
    
    def full_conditional_sampler_tau2_0(self, last_param):
        # 0       1                                          2    3
        #[sigma2, [[beta0_1,...,beta0_20], beta1,...,beta5], mu0, tau2_0]
        param_a = self.hyper_a_tau0 + 20 / 2

        beta0 = np.array(last_param[1][0])
        mu0 = last_param[2]
        resid = beta0 - mu0
        param_b = self.hyper_b_tau0 + np.dot(resid, resid) / 2
        new_tau2_0 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], new_tau2_0]
        return new_sample


    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu0(new)
        new = self.full_conditional_sampler_tau2_0(new)
        self.MC_sample.append(new)

# 0       1                                          2    3
#[sigma2, [[beta0_1,...,beta0_20], beta1,...,beta5], mu0, tau2_0]
rand_int_initial = [1,[[0 for _ in range(20)],0,0,0,0,0],0,1]
rand_int_inst = HW4Prob2_Rand_Int(y, rand_int_X, rand_int_initial)
rand_int_inst.generate_samples(10000)

rand_int_beta0 = [sample[1][0] for sample in rand_int_inst.MC_sample]
rand_int_beta1_5 = [sample[1][1:] for sample in rand_int_inst.MC_sample]
rand_int_others = [[sample[0], sample[2], sample[3]] for sample in rand_int_inst.MC_sample]
rand_int_diag_inst1 = MCMC_Diag()
rand_int_diag_inst1.set_mc_samples_from_list(rand_int_beta0)
rand_int_diag_inst1.set_variable_names(["beta0_"+str(i) for i in range(1,21)])
rand_int_diag_inst1.show_traceplot((4,5))
rand_int_diag_inst1.show_hist((4,5))
rand_int_diag_inst1.show_acf(30,(4,5))

rand_int_diag_inst2 = MCMC_Diag()
rand_int_diag_inst2.set_mc_samples_from_list(rand_int_beta1_5)
rand_int_diag_inst2.set_variable_names(["beta"+str(i) for i in range(1,6)])
rand_int_diag_inst2.show_traceplot((2,3))
rand_int_diag_inst2.show_hist((2,3))
rand_int_diag_inst2.show_acf(30,(2,3))

rand_int_diag_inst3 = MCMC_Diag()
rand_int_diag_inst3.set_mc_samples_from_list(rand_int_others)
rand_int_diag_inst3.set_variable_names(["sigma2", "mu0", "tau2_0"])
rand_int_diag_inst3.show_traceplot((1,3))
rand_int_diag_inst3.show_hist((1,3))
rand_int_diag_inst3.show_acf(30,(1,3))




class HW4Prob2_Rand_Slope(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.MC_sample = [initial]
        # 0       1                                                                            2              3
        #[sigma2, [[beta0_1,...,beta0_20], [beta1_1,...,beta1_20],...,[beta5_1,...,beta5_20]], [mu0,...,mu5], [tau2_0,...,tau2_5]]
        
        # able to tune these
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        self.hyper_a_tau = [0.01 for _ in range(6)]
        self.hyper_b_tau = [0.01 for _ in range(6)]
        self.hyper_phi2 = [100 for _ in range(6)]

    def full_conditional_sampler_beta(self, last_param):
        # 0       1                                                                            2              3
        #[sigma2, [[beta0_1,...,beta0_20], [beta1_1,...,beta1_20],...,[beta5_1,...,beta5_20]], [mu0,...,mu5], [tau2_0,...,tau2_5]]
        sigma2 = last_param[0]
        mu = last_param[2]
        tau2 = last_param[3]
        m_array = np.array([mu_k for mu_k in mu for _ in range(20)])
        D_inv = np.diag(1/np.array([tau2_k for tau2_k in tau2 for _ in range(20)]))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_beta_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        new_beta_sample_form = []
        for i in range(6):
            new_beta_sample_form.append(new_beta_vec[(i*20) : ((i+1)*20)])
        new_sample = [last_param[0], new_beta_sample_form, last_param[2], last_param[3]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        # 0       1                                                                            2              3
        #[sigma2, [[beta0_1,...,beta0_20], [beta1_1,...,beta1_20],...,[beta5_1,...,beta5_20]], [mu0,...,mu5], [tau2_0,...,tau2_5]]
        beta_vec = np.array([item for sublist in last_param[1] for item in sublist])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ beta_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [new_sigma2, last_param[1], last_param[2], last_param[3]]
        return new_sample

    def full_conditional_sampler_mu(self, last_param):
        # 0       1                                                                            2              3
        #[sigma2, [[beta0_1,...,beta0_20], [beta1_1,...,beta1_20],...,[beta5_1,...,beta5_20]], [mu0,...,mu5], [tau2_0,...,tau2_5]]
        new_mu = []
        for k in range(6):
            tau2_k = last_param[3][k]
            beta_k = last_param[1][k]
            param_var = 1/(20/tau2_k + 1/self.hyper_phi2[k])
            param_mean = param_var * (np.sum(beta_k) / tau2_k)
            new_mu_k = self.np_rng.normal(param_mean, np.sqrt(param_var))
            new_mu.append(new_mu_k)
        new_sample = [last_param[0], last_param[1], new_mu, last_param[3]]
        return new_sample
    
    def full_conditional_sampler_tau2(self, last_param):
        # 0       1                                                                            2              3
        #[sigma2, [[beta0_1,...,beta0_20], [beta1_1,...,beta1_20],...,[beta5_1,...,beta5_20]], [mu0,...,mu5], [tau2_0,...,tau2_5]]
        new_tau2 = []
        for k in range(6):
            param_a = self.hyper_a_tau[k] + 20 / 2
            beta_k = np.array(last_param[1][k])
            mu_k = last_param[2][k]
            resid = beta_k - mu_k
            param_b = self.hyper_b_tau[k] + np.dot(resid, resid) / 2
            new_tau2_k = self.inv_gamma_sampler.sampler(param_a, param_b)
            new_tau2.append(new_tau2_k)
          
        new_sample = [last_param[0], last_param[1], last_param[2], new_tau2]
        return new_sample


    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu(new)
        new = self.full_conditional_sampler_tau2(new)
        self.MC_sample.append(new)
    
# 0       1                                                                            2              3
#[sigma2, [[beta0_1,...,beta0_20], [beta1_1,...,beta1_20],...,[beta5_1,...,beta5_20]], [mu0,...,mu5], [tau2_0,...,tau2_5]]
rand_slope_initial = [1, [[0 for _ in range(20)] for _ in range(6)], [0 for _ in range(6)], [1 for _ in range(6)]]
rand_slope_inst = HW4Prob2_Rand_Slope(y, rand_slope_X, rand_slope_initial)

rand_slope_inst.generate_samples(10000)

rand_slope_beta0 = [sample[1][0] for sample in rand_slope_inst.MC_sample]
rand_slope_beta1 = [sample[1][1] for sample in rand_slope_inst.MC_sample]
rand_slope_beta2 = [sample[1][2] for sample in rand_slope_inst.MC_sample]
rand_slope_beta3 = [sample[1][3] for sample in rand_slope_inst.MC_sample]
rand_slope_beta4 = [sample[1][4] for sample in rand_slope_inst.MC_sample]
rand_slope_beta5 = [sample[1][5] for sample in rand_slope_inst.MC_sample]
rand_slope_beta5 = [sample[1][5] for sample in rand_slope_inst.MC_sample]
rand_slope_mu = [sample[2] for sample in rand_slope_inst.MC_sample]
rand_slope_tau2 = [sample[3] for sample in rand_slope_inst.MC_sample]
rand_slope_sigma2 = [[sample[0]] for sample in rand_slope_inst.MC_sample]

rand_slope_diag_inst0 = MCMC_Diag()
rand_slope_diag_inst0.set_mc_samples_from_list(rand_slope_beta0)
rand_slope_diag_inst0.set_variable_names(["beta0_"+str(i) for i in range(1,21)])
rand_slope_diag_inst0.show_traceplot((4,5))
rand_slope_diag_inst0.show_hist((4,5))
rand_slope_diag_inst0.show_acf(30,(4,5))

rand_slope_diag_inst1 = MCMC_Diag()
rand_slope_diag_inst1.set_mc_samples_from_list(rand_slope_beta1)
rand_slope_diag_inst1.set_variable_names(["beta1_"+str(i) for i in range(1,21)])
rand_slope_diag_inst1.show_traceplot((4,5))
rand_slope_diag_inst1.show_hist((4,5))
rand_slope_diag_inst1.show_acf(30,(4,5))

rand_slope_diag_inst2 = MCMC_Diag()
rand_slope_diag_inst2.set_mc_samples_from_list(rand_slope_beta2)
rand_slope_diag_inst2.set_variable_names(["beta2_"+str(i) for i in range(1,21)])
rand_slope_diag_inst2.show_traceplot((4,5))
rand_slope_diag_inst2.show_hist((4,5))
rand_slope_diag_inst2.show_acf(30,(4,5))

rand_slope_diag_inst3 = MCMC_Diag()
rand_slope_diag_inst3.set_mc_samples_from_list(rand_slope_beta3)
rand_slope_diag_inst3.set_variable_names(["beta3_"+str(i) for i in range(1,21)])
rand_slope_diag_inst3.show_traceplot((4,5))
rand_slope_diag_inst3.show_hist((4,5))
rand_slope_diag_inst3.show_acf(30,(4,5))

rand_slope_diag_inst4 = MCMC_Diag()
rand_slope_diag_inst4.set_mc_samples_from_list(rand_slope_beta4)
rand_slope_diag_inst4.set_variable_names(["beta4_"+str(i) for i in range(1,21)])
rand_slope_diag_inst4.show_traceplot((4,5))
rand_slope_diag_inst4.show_hist((4,5))
rand_slope_diag_inst4.show_acf(30,(4,5))

rand_slope_diag_inst5 = MCMC_Diag()
rand_slope_diag_inst5.set_mc_samples_from_list(rand_slope_beta5)
rand_slope_diag_inst5.set_variable_names(["beta5_"+str(i) for i in range(1,21)])
rand_slope_diag_inst5.show_traceplot((4,5))
rand_slope_diag_inst5.show_hist((4,5))
rand_slope_diag_inst5.show_acf(30,(4,5))

rand_slope_diag_inst6 = MCMC_Diag()
rand_slope_diag_inst6.set_mc_samples_from_list(rand_slope_mu)
rand_slope_diag_inst6.set_variable_names(["mu"+str(i) for i in range(6)])
rand_slope_diag_inst6.show_traceplot((2,3))
rand_slope_diag_inst6.show_hist((2,3))
rand_slope_diag_inst6.show_acf(30,(2,3))

rand_slope_diag_inst7 = MCMC_Diag()
rand_slope_diag_inst7.set_mc_samples_from_list(rand_slope_tau2)
rand_slope_diag_inst7.set_variable_names(["tau2_"+str(i) for i in range(6)])
rand_slope_diag_inst7.show_traceplot((2,3))
rand_slope_diag_inst7.show_hist((2,3))
rand_slope_diag_inst7.show_acf(30,(2,3))

rand_slope_diag_inst8 = MCMC_Diag()
rand_slope_diag_inst8.set_mc_samples_from_list(rand_slope_sigma2)
rand_slope_diag_inst8.set_variable_names(["sigma2"])
rand_slope_diag_inst8.show_traceplot((1,1))
rand_slope_diag_inst8.show_hist((1,1))
rand_slope_diag_inst8.show_acf(30,(1,1))

