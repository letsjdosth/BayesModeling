import csv
from random import seed

import numpy as np

from bayesian_tools.LM_Core import LM_base
from bayesian_tools.MCMC_Core import MCMC_Diag
from special_dist_sampler.sampler_gamma import Sampler_univariate_InvGamma

if __name__=="__main__":
    seed(20220530)

class MTCARsData:
    def __init__(self):
        self._load()
        self._normalize()

    def _load(self, file_path = "dataset/mtcars.csv"):
        self.qsec = []
        self.covariates_with_1 = []

        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader) 
            #header 
            # 0     1   2       3   4       5   6       7   8   9       10
            # mpg	cyl	disp	hp	drat	wt	qsec	vs	am	gear	carb
            for row in csv_reader:
                self.qsec.append(float(row[6]))
                
                covariate_i = row[0:6] + row[7:]
                self.covariates_with_1.append([1]+[float(x) for x in covariate_i] )
                
        self.num_data = len(self.qsec)
        self.covariates_with_1 = np.array(self.covariates_with_1)
    
    def _normalize(self):
        for k in range(1,11):
            std_k = np.std(self.covariates_with_1[:, k])
            self.covariates_with_1[:,k] = self.covariates_with_1[:,k] / std_k
        
    def get_yX(self):
        return (np.array(self.qsec), self.covariates_with_1)

if __name__=="__main__":
    factory_inst = MTCARsData()
    mtcars_y, mtcars_X = factory_inst.get_yX()
    # print(mtcars_y)
    # print(mtcars_X.shape)


class HW4Prob5(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.num_data = len(self.y)
        self.p = design_matrix.shape[1] - 1 #without intercept

        self.MC_sample = [initial]
        # 0       1      2                         3             4
        #[sigma2, beta0, [beta1,beta2,...,beta10], [z1,...,z10], pi]
        
        # able to tune these
        self.hyper_tau2_beta0 = 10
        self.hyper_tau2_0 = 0.01
        self.hyper_tau2_1 = 100 #tau1>tau0
        self.hyper_alpha_pi = 1
        self.hyper_beta_pi = 1
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01

    def full_conditional_sampler_beta(self, last_param):
        # 0       1      2                         3             4
        #[sigma2, beta0, [beta1,beta2,...,beta10], [z1,...,z10], pi]
        sigma2 = last_param[0]
        z = last_param[3]
        
        D_diag_vec = [self.hyper_tau2_beta0]
        for z_k in z:
            if z_k == 1:
                D_diag_vec.append(sigma2 * self.hyper_tau2_1)
            else:
                D_diag_vec.append(sigma2 * self.hyper_tau2_0)

        D_inv = np.diag(1 / np.array(D_diag_vec))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (self.xty/sigma2)
        new_coeff_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        new_beta0 = new_coeff_vec[0]
        new_beta = new_coeff_vec[1:].tolist()

        new_sample = [last_param[0], new_beta0, new_beta, last_param[3], last_param[4]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        # 0       1      2                         3             4
        #[sigma2, beta0, [beta1,beta2,...,beta10], [z1,...,z10], pi]
        beta_square = np.array([b**2 for b in last_param[2]])
        z = np.array(last_param[3])
        coeff_vec = np.array([last_param[1]] + last_param[2])
        
        param_a = self.hyper_a_sigma + (self.num_data + self.p) / 2
        resid = self.y - self.x @ coeff_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2 + np.dot(z, beta_square) / (2*self.hyper_tau2_1) + np.dot((1-z), beta_square) / (2*self.hyper_tau2_0)
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [new_sigma2, last_param[1], last_param[2], last_param[3], last_param[4]]
        return new_sample

    def full_conditional_sampler_z(self, last_param):
        # 0       1      2                         3             4
        #[sigma2, beta0, [beta1,beta2,...,beta10], [z1,...,z10], pi]
        sigma2 = last_param[0]
        pi = last_param[4]
        new_z = []
        for beta_k in last_param[2]:
            A_k = pi*(sigma2 * self.hyper_tau2_1)**(-0.5) * np.exp(- beta_k**2 / (2*sigma2*self.hyper_tau2_1))
            B_k = (1-pi)*(sigma2 * self.hyper_tau2_0)**(-0.5) * np.exp(- beta_k**2 / (2*sigma2*self.hyper_tau2_0))
            bern_p = A_k/(A_k + B_k)
            unif_sample = self.np_rng.random()
            if unif_sample < bern_p:
                new_z.append(1)
            else:
                new_z.append(0)
        new_sample = [last_param[0], last_param[1], last_param[2], new_z, last_param[4]]
        return new_sample

    def full_conditional_sampler_pi(self, last_param):
        # 0       1      2                         3             4
        #[sigma2, beta0, [beta1,beta2,...,beta10], [z1,...,z10], pi]
        z_sum = np.sum(last_param[3])
        param_alpha = self.hyper_alpha_pi + z_sum
        param_beta = self.hyper_beta_pi + self.p - z_sum
        new_pi = self.np_rng.beta(param_alpha, param_beta)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], new_pi]
        return new_sample


    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_z(new)
        new = self.full_conditional_sampler_pi(new)
        self.MC_sample.append(new)


if __name__=="__main__":
    # 0       1      2                         3             4
    #[sigma2, beta0, [beta1,beta2,...,beta10], [z1,...,z10], pi]
    prob5_initial = [0.1, 0, [0 for _ in range(10)], [0 for _ in range(10)], 0.5]
    prob5_inst = HW4Prob5(mtcars_y, mtcars_X, prob5_initial)
    prob5_inst.generate_samples(100000)

    # print(prob5_inst.MC_sample[-2])
    # print(prob5_inst.MC_sample[-1])

    prob5_beta = [[sample[1]] + sample[2] for sample in prob5_inst.MC_sample]
    prob5_z = [sample[3] for sample in prob5_inst.MC_sample]
    prob5_others = [[sample[0], sample[4]] for sample in prob5_inst.MC_sample]

    prob5_diag_inst2 = MCMC_Diag()
    prob5_diag_inst2.set_mc_samples_from_list(prob5_z)
    prob5_diag_inst2.set_variable_names(["z_"+str(i) for i in range(1,11)])
    prob5_diag_inst2.thinning(30)
    prob5_diag_inst2.show_traceplot((2,5))
    prob5_diag_inst2.show_hist((2,5))
    prob5_diag_inst2.show_acf(30,(2,5))


    prob5_diag_inst1 = MCMC_Diag()
    prob5_diag_inst1.set_mc_samples_from_list(prob5_beta)
    prob5_diag_inst1.set_variable_names(["beta_"+str(i) for i in range(11)])
    prob5_diag_inst1.thinning(30)
    prob5_diag_inst1.show_traceplot((3,4))
    prob5_diag_inst1.show_hist((3,4))
    prob5_diag_inst1.show_acf(30,(3,4))
    prob5_diag_inst1.show_boxplot([i for i in range(1,11)])
    prob5_diag_inst1.show_mean_CI_plot([i for i in range(1,11)])

    prob5_diag_inst3 = MCMC_Diag()
    prob5_diag_inst3.set_mc_samples_from_list(prob5_others)
    prob5_diag_inst3.set_variable_names(["sigma2", "pi"])
    prob5_diag_inst3.thinning(30)
    prob5_diag_inst3.show_traceplot((1,2))
    prob5_diag_inst3.show_hist((1,2))
    prob5_diag_inst3.show_acf(30,(1,2))