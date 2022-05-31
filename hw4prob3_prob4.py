import csv
from random import seed

import numpy as np

from bayesian_tools.LM_Core import LM_base
from bayesian_tools.MCMC_Core import MCMC_Diag
from special_dist_sampler.sampler_gamma import Sampler_univariate_InvGamma

if __name__=="__main__":
    seed(20220530)

class BroadwayData:
    def __init__(self):
        self._load()
        self._cal_rand_eff_design_matrix()

    def _load(self, file_path = "dataset/broadway.csv"):
        self.gross=[]
        self.id_show = []
        self.id_year = []
        self.id_week = []

        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader) 
            #header 
            #y                          i       j       k
            #gross	show	week_ending	id.show	id.year	id.week
            #0      1       2           3       4       5
            for row in csv_reader:
                self.gross.append(float(row[0]))
                self.id_show.append(int(row[3]))
                self.id_year.append(int(row[4]))
                self.id_week.append(int(row[5]))

        self.num_data = len(self.gross)
        self.num_cate_show = 10
        self.num_cate_year = 21
        self.num_cate_week = 53

    def _cal_rand_eff_design_matrix(self):
        self.show_ind_mat = np.zeros((self.num_data, self.num_cate_show))
        self.year_ind_mat = np.zeros((self.num_data, self.num_cate_year))
        self.year_value_mat = np.zeros((self.num_data, self.num_cate_show)) #for show (index i)
        self.week_ind_mat = np.zeros((self.num_data, self.num_cate_week))

        for n, (show, year, week) in enumerate(zip(self.id_show, self.id_year, self.id_week)):
            self.show_ind_mat[n, show-1] = 1
            self.year_ind_mat[n, year-1] = 1
            self.year_value_mat[n, show-1] = year
            self.week_ind_mat[n, week-1] = 1

        
    def get_yX_show_year_for_prob3(self):
        prob3_X = np.c_[self.show_ind_mat, self.year_ind_mat]
        return (np.array(self.gross), prob3_X, self.id_show, self.id_year)
    
    def get_yX_show_week_for_prob4(self):
        prob4_X = np.c_[self.show_ind_mat, self.week_ind_mat, self.year_value_mat]
        return (np.array(self.gross), prob4_X, self.id_show, self.id_week)

if __name__=="__main__":
    factory_inst = BroadwayData()
    y, prob3_X, id_show, id_year = factory_inst.get_yX_show_year_for_prob3()
    y, prob4_X, id_show, id_week = factory_inst.get_yX_show_week_for_prob4()


class HW4Prob3(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)


        self.num_data = len(self.y)
        self.num_cate_show = 10 #I
        self.num_cate_year = 21 #J
        self.num_cate_week = 53 #K

        self.MC_sample = [initial]
        # 0       1                       2                   3    4
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta21], mu_b, tau2_b]
        
        # able to tune these
        self.hyper_mu_a = 0
        self.hyper_tau2_a = 100
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        self.hyper_a_tau_b = 0.01
        self.hyper_b_tau_b = 0.01
        self.hyper_phi2_b = 0.01 #<- need to justify


    def full_conditional_sampler_alpha_and_beta(self, last_param):
        # 0       1                       2                   3    4
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta21], mu_b, tau2_b]
        sigma2 = last_param[0]
        mu_b = last_param[3]
        tau2_b = last_param[4]

        m_array = np.array([self.hyper_mu_a]*self.num_cate_show + [mu_b]*self.num_cate_year)
        D_inv = np.diag(1 / np.array([self.hyper_tau2_a]*self.num_cate_show + [tau2_b]*self.num_cate_year))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_coeff_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        
        new_alpha = new_coeff_vec[0:self.num_cate_show].tolist()
        new_beta = new_coeff_vec[self.num_cate_show:].tolist()
        new_sample = [last_param[0], new_alpha, new_beta, last_param[3], last_param[4]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        # 0       1                       2                   3    4
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta21], mu_b, tau2_b]
        coeff_vec = np.array(last_param[1] + last_param[2])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ coeff_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [new_sigma2, last_param[1], last_param[2], last_param[3], last_param[4]]
        return new_sample

    def full_conditional_sampler_mu_b(self, last_param):
        # 0       1                       2                   3    4
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta21], mu_b, tau2_b]
        tau2_b = last_param[4]
        beta = last_param[2]
        param_var = 1/(self.num_cate_year/tau2_b + 1/self.hyper_phi2_b)
        param_mean = param_var * (np.sum(beta) / tau2_b)
        new_mu_b = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], new_mu_b, last_param[4]]
        return new_sample
    
    def full_conditional_sampler_tau2_b(self, last_param):
        # 0       1                       2                   3    4
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta21], mu_b, tau2_b]
        param_a = self.hyper_a_tau_b + self.num_cate_year / 2

        beta = np.array(last_param[2])
        mu = np.array([last_param[3]]*self.num_cate_year)
        resid = beta - mu
        param_b = self.hyper_b_tau_b + np.dot(resid, resid) / 2
        new_tau2_b = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], new_tau2_b]
        return new_sample

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_alpha_and_beta(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu_b(new)
        new = self.full_conditional_sampler_tau2_b(new)
        self.MC_sample.append(new)



if __name__=="__main__":
    # 0       1                       2                   3    4
    #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta21], mu_b, tau2_b]
    prob3_initial = [0.1, [0 for _ in range(10)], [0 for _ in range(21)], 0, 0.1]
    prob3_inst = HW4Prob3(y, prob3_X, prob3_initial)
    prob3_inst.generate_samples(10000)

    # print(prob3_inst.MC_sample[-2])
    # print(prob3_inst.MC_sample[-1])

    prob3_alpha = [sample[1] for sample in prob3_inst.MC_sample]
    prob3_beta = [sample[2] for sample in prob3_inst.MC_sample]
    prob3_others = [[sample[0], sample[3], sample[4]] for sample in prob3_inst.MC_sample]

    prob3_diag_inst1 = MCMC_Diag()
    prob3_diag_inst1.set_mc_samples_from_list(prob3_alpha)
    prob3_diag_inst1.set_variable_names(["alpha_"+str(i) for i in range(1,11)])
    prob3_diag_inst1.show_traceplot((3,4))
    prob3_diag_inst1.show_hist((3,4))
    prob3_diag_inst1.show_acf(30,(3,4))

    prob3_diag_inst2 = MCMC_Diag()
    prob3_diag_inst2.set_mc_samples_from_list(prob3_beta)
    prob3_diag_inst2.set_variable_names(["beta_"+str(i) for i in range(1,22)])
    prob3_diag_inst2.show_traceplot((3,7))
    prob3_diag_inst2.show_hist((3,7))
    prob3_diag_inst2.show_acf(30,(3,7))

    prob3_diag_inst3 = MCMC_Diag()
    prob3_diag_inst3.set_mc_samples_from_list(prob3_others)
    prob3_diag_inst3.set_variable_names(["sigma2", "mu_b", "tau2_b"])
    prob3_diag_inst3.show_traceplot((1,3))
    prob3_diag_inst3.show_hist((1,3))
    prob3_diag_inst3.show_acf(30,(1,3))





class HW4Prob4(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.num_data = len(self.y)
        self.num_cate_show = 10 #I
        self.num_cate_year = 21 #J
        self.num_cate_week = 53 #K
        
        self.MC_sample = [initial]
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        
        # able to tune these
        self.hyper_mu_a = 0
        self.hyper_tau2_a = 100
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        self.hyper_a_tau_b = 0.01
        self.hyper_b_tau_b = 0.01
        self.hyper_a_tau_r = 0.01
        self.hyper_b_tau_r = 0.01
        self.hyper_phi2_b = 0.001 #<- need to justify
        self.hyper_phi2_r = 100 #<- need to justify

    
    def full_conditional_sampler_alpha_beta_and_gamma(self, last_param):
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        sigma2 = last_param[0]
        mu_b = last_param[4][0]
        mu_r = last_param[4][1]
        tau2_b = last_param[5][0]
        tau2_r = last_param[5][1]

        m_array = np.array([self.hyper_mu_a]*self.num_cate_show + [mu_b]*self.num_cate_week + [mu_r]*self.num_cate_show)
        D_inv = np.diag(1 / np.array([self.hyper_tau2_a]*self.num_cate_show + [tau2_b]*self.num_cate_week + [tau2_r]*self.num_cate_show))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_coeff_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        
        new_alpha = new_coeff_vec[0:self.num_cate_show].tolist()
        new_beta = new_coeff_vec[self.num_cate_show:(self.num_cate_show+self.num_cate_week)].tolist()
        new_gamma = new_coeff_vec[(self.num_cate_show+self.num_cate_week):].tolist()
        new_sample = [last_param[0], new_alpha, new_beta, new_gamma, last_param[4], last_param[5]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        coeff_vec = np.array(last_param[1] + last_param[2] + last_param[3])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ coeff_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [new_sigma2, last_param[1], last_param[2], last_param[3], last_param[4], last_param[5]]
        return new_sample

    def full_conditional_sampler_mu_b(self, last_param):
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        tau2_b = last_param[5][0]
        beta = last_param[2]
        param_var = 1/(self.num_cate_week/tau2_b + 1/self.hyper_phi2_b)
        param_mean = param_var * (np.sum(beta) / tau2_b)
        new_mu_b = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], [new_mu_b, last_param[4][1]], last_param[5]]
        return new_sample
    
    def full_conditional_sampler_tau2_b(self, last_param):
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        param_a = self.hyper_a_tau_b + self.num_cate_week / 2

        beta = np.array(last_param[2])
        mu_b_vec = np.array([last_param[4][0]]*self.num_cate_week)
        resid = beta - mu_b_vec
        param_b = self.hyper_b_tau_b + np.dot(resid, resid) / 2
        new_tau2_b = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], [new_tau2_b, last_param[5][1]]]
        return new_sample
    
    
    def full_conditional_sampler_mu_r(self, last_param):
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        tau2_r = last_param[5][1]
        gamma = last_param[3]
        param_var = 1/(self.num_cate_show/tau2_r + 1/self.hyper_phi2_r)
        param_mean = param_var * (np.sum(gamma) / tau2_r)
        new_mu_r = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], [last_param[4][0], new_mu_r], last_param[5]]
        return new_sample
    
    def full_conditional_sampler_tau2_r(self, last_param):
        # 0        1                       2                   3                     4             5
        #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
        param_a = self.hyper_a_tau_r + self.num_cate_show / 2

        gamma = np.array(last_param[3])
        mu_r_vec = np.array([last_param[4][1]]*self.num_cate_show)
        resid = gamma - mu_r_vec
        param_b = self.hyper_b_tau_r + np.dot(resid, resid) / 2
        new_tau2_r = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], [last_param[5][0], new_tau2_r]]
        return new_sample

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_alpha_beta_and_gamma(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu_b(new)
        new = self.full_conditional_sampler_tau2_b(new)
        new = self.full_conditional_sampler_mu_r(new)
        new = self.full_conditional_sampler_tau2_r(new)

        self.MC_sample.append(new)


if __name__=="__main__":
    # 0        1                       2                   3                     4             5
    #[sigma2, [alpha_1,...,alpha_10], [beta1,...,beta53], [gamma1,...,gamma10], [mu_b, mu_r], [tau2_b, tau2_r]]
    prob4_initial = [0.1, [0 for _ in range(10)], [0 for _ in range(53)], [0 for _ in range(10)], [0, 0], [0.1, 0.1]]
    prob4_inst = HW4Prob4(y, prob4_X, prob4_initial)
    prob4_inst.generate_samples(10000)

    # print(prob4_inst.MC_sample[-2])
    # print(prob4_inst.MC_sample[-1])

    prob4_alpha = [sample[1] for sample in prob4_inst.MC_sample]
    prob4_beta = [sample[2] for sample in prob4_inst.MC_sample]
    prob4_gamma = [sample[3] for sample in prob4_inst.MC_sample]
    prob4_others = [[sample[0], sample[4][0], sample[4][1], sample[5][0], sample[5][1]] for sample in prob4_inst.MC_sample]

    prob4_diag_inst1 = MCMC_Diag()
    prob4_diag_inst1.set_mc_samples_from_list(prob4_alpha)
    prob4_diag_inst1.set_variable_names(["alpha_"+str(i) for i in range(1,11)])
    prob4_diag_inst1.show_traceplot((3,4))
    prob4_diag_inst1.show_hist((3,4))
    prob4_diag_inst1.show_acf(30,(3,4))

    prob4_diag_inst2 = MCMC_Diag()
    prob4_diag_inst2.set_mc_samples_from_list(prob4_beta)
    prob4_diag_inst2.set_variable_names(["beta_"+str(i) for i in range(1,54)])
    prob4_diag_inst2.show_traceplot((3,7), [i for i in range(21)])
    # prob4_diag_inst2.show_hist((3,7), [i for i in range(21)])
    prob4_diag_inst2.show_acf(30,(3,7), [i for i in range(21)])


    prob4_diag_inst3 = MCMC_Diag()
    prob4_diag_inst3.set_mc_samples_from_list(prob4_gamma)
    prob4_diag_inst3.set_variable_names(["gamma_"+str(i) for i in range(1,11)])
    prob4_diag_inst3.show_traceplot((3,4))
    prob4_diag_inst3.show_hist((3,4))
    prob4_diag_inst3.show_acf(30,(3,4))

    prob4_diag_inst4 = MCMC_Diag()
    prob4_diag_inst4.set_mc_samples_from_list(prob4_others)
    prob4_diag_inst4.set_variable_names(["sigma2", "mu_b", "mu_r", "tau2_b", "tau2_r"])
    prob4_diag_inst4.show_traceplot((2,3))
    prob4_diag_inst4.show_hist((2,3))
    prob4_diag_inst4.show_acf(30,(2,3))

