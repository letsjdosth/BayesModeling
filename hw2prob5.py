import csv
from random import seed

import numpy as np

from bayesian_tools.MCMC_Core import MCMC_Diag, MCMC_Gibbs
from special_dist_sampler.sampler_gamma import Sampler_InvWishart

class MovieRating_Data:
    # p = 30 movies
    # n = 100 raters
    # keep n*p matrix
    def __init__(self):
        self.data_by_list = []
        self.data_by_nparray = None
        self._load()
        self.n, self.p = self.data_by_nparray.shape

    def _load(self, file_path = "dataset/SimMovieRating.csv"):
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader) #skip the header

            for row in csv_reader:
                float_row = [float(x) for x in row]
                self.data_by_list.append(float_row)
        
        self.data_by_nparray = np.array(self.data_by_list)


class Gibbs_HW5P6(MCMC_Gibbs):
    def __init__(self, initial, hyper_k0, MovieRating_Data_inst:MovieRating_Data):
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        super().__init__(initial)
        self.dim = len(initial[0])
        self.hyper_k0 = hyper_k0
        self.random_generator = np.random.default_rng()
        self.inv_wishart_generator = Sampler_InvWishart()
        self.data_inst = MovieRating_Data_inst


    def full_conditional_sampler_mu_u(self, last_param):
        new_sample = last_param #check if it cause a bug
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        mvn_cov = new_sample[1]/(self.data_inst.n + self.hyper_k0)
        mvn_mean = np.matmul(mvn_cov, np.sum(new_sample[4], axis=0))
        new_mu_u = self.random_generator.multivariate_normal(mvn_mean, mvn_cov)
        new_sample[0] = new_mu_u
        return new_sample

    def full_conditional_sampler_Sigma_u(self, last_param):
        new_sample = last_param
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        inv_wishart_df = self.data_inst.n + self.data_inst.p + 2
        inv_wishart_scale = np.identity(self.dim)
        for u_i in new_sample[4]:
            inv_wishart_scale += np.outer(u_i - new_sample[0], u_i - new_sample[0])
        inv_wishart_scale += (np.outer(new_sample[0], new_sample[0])*self.hyper_k0)
        new_Sigma_u = self.inv_wishart_generator.sampler_iter(1, inv_wishart_df, inv_wishart_scale)[0]
        new_sample[1] = new_Sigma_u
        return new_sample

    def full_conditional_sampler_mu_v(self, last_param):
        new_sample = last_param
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        mvn_cov = new_sample[3]/(self.data_inst.p + self.hyper_k0)
        mvn_mean = np.matmul(mvn_cov, np.sum(new_sample[5], axis=0))
        new_mu_v = self.random_generator.multivariate_normal(mvn_mean, mvn_cov)
        new_sample[2] = new_mu_v
        return new_sample

    def full_conditional_sampler_Sigma_v(self, last_param):
        new_sample = last_param
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        inv_wishart_df = 2*self.data_inst.p + 2
        inv_wishart_scale = np.identity(self.dim)
        for v_i in new_sample[5]:
            inv_wishart_scale += np.outer(v_i - new_sample[2], v_i - new_sample[2])
        inv_wishart_scale += (np.outer(new_sample[2], new_sample[2])*self.hyper_k0)
        new_Sigma_v = self.inv_wishart_generator.sampler_iter(1, inv_wishart_df, inv_wishart_scale)[0]
        new_sample[3] = new_Sigma_v
        return new_sample

    def full_conditional_sampler_u_vec(self, last_param):
        new_sample = last_param
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        mvn_cov_inv = np.linalg.inv(new_sample[1])
        for v_j in new_sample[5]:
            mvn_cov_inv += np.outer(v_j, v_j)
        mvn_cov = np.linalg.inv(mvn_cov_inv)
        

        mvn_mean_numerator_before_xijvj = np.matmul(np.linalg.inv(new_sample[1]), new_sample[0])
        for i in range(self.data_inst.n):
            mvn_mean_numerator_u_i = mvn_mean_numerator_before_xijvj + 0
            for x_ij, v_j in zip(self.data_inst.data_by_nparray[i,:], new_sample[5]):
                mvn_mean_numerator_u_i += (x_ij*v_j)
            mvn_mean_u_i = np.matmul(mvn_cov, mvn_mean_numerator_u_i)
            
            new_u_i = self.random_generator.multivariate_normal(mvn_mean_u_i, mvn_cov)
            new_sample[4][i] = new_u_i
        return new_sample

    def full_conditional_sampler_v_vec(self, last_param):
        new_sample = last_param
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        mvn_cov_inv = np.linalg.inv(new_sample[3])
        for u_i in new_sample[4]:
            mvn_cov_inv += np.outer(u_i, u_i)
        mvn_cov = np.linalg.inv(mvn_cov_inv)
        
        mvn_mean_numerator_before_xijui = np.matmul(np.linalg.inv(new_sample[3]), new_sample[2])
        for j in range(self.data_inst.p):
            mvn_mean_numerator_v_j = mvn_mean_numerator_before_xijui + 0
            for x_ij, u_i in zip(self.data_inst.data_by_nparray[:,j], new_sample[4]):
                mvn_mean_numerator_v_j += (x_ij*u_i)
            mvn_mean_v_j = np.matmul(mvn_cov, mvn_mean_numerator_v_j)
            new_v_j = self.random_generator.multivariate_normal(mvn_mean_v_j, mvn_cov)
            new_sample[5][j] = new_v_j
        return new_sample


    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        #index   0     1        2     3        4            5
        #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]
        new = [np.copy(x) for x in last[:4]] + [[np.copy(x) for x in last[4]]] + [[np.copy(x) for x in last[5]]]
        new = self.full_conditional_sampler_mu_u(new)
        new = self.full_conditional_sampler_Sigma_u(new)
        new = self.full_conditional_sampler_mu_v(new)
        new = self.full_conditional_sampler_Sigma_v(new)
        new = self.full_conditional_sampler_u_vec(new)
        new = self.full_conditional_sampler_v_vec(new)
        self.MC_sample.append(new)
    
if __name__=="__main__":
    seed(20220425)
    data_inst = MovieRating_Data()
    # print(len(data_inst.data_by_nparray[0,:]), data_inst.data_by_nparray[0,:])
    # print(len(data_inst.data_by_nparray[:,0]), data_inst.data_by_nparray[:,0])
    # print(data_inst.n) #100
    # print(data_inst.p) #30

    #index   0     1        2     3        4            5
    #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]

    # k = 1
    # initial1 = [np.array([0 for _ in range(k)]), np.identity(k), np.array([0 for _ in range(k)]), np.identity(k),
    #             [np.array([0 for _ in range(k)]) for _ in range(data_inst.n)], [np.array([0 for _ in range(k)]) for _ in range(data_inst.p)]]
    # gibbs_inst1 = Gibbs_HW5P6(initial1, hyper_k0 = 1, MovieRating_Data_inst = data_inst)
    # gibbs_inst1.generate_samples(5000)
    # diag_inst11 = MCMC_Diag()
    # diag_inst11.set_mc_samples_from_list([[sample[0], np.linalg.det(sample[1]), sample[2], np.linalg.det(sample[3])] for sample in gibbs_inst1.MC_sample])
    # diag_inst11.set_variable_names(["mu_u", "det(Sigma_u)", "mu_v", "det(Sigma_v)"])
    # diag_inst11.show_traceplot((2,2))
    # diag_inst12 = MCMC_Diag()
    # diag_inst12.set_mc_samples_from_list([sample[4] for sample in gibbs_inst1.MC_sample])
    # diag_inst12.set_variable_names(["u"+str(i+1) for i in range(data_inst.n)])
    # diag_inst12.show_traceplot((2,2), [0,1,2,3])
    # diag_inst13 = MCMC_Diag()
    # diag_inst13.set_mc_samples_from_list([sample[5] for sample in gibbs_inst1.MC_sample])
    # diag_inst13.set_variable_names(["v"+str(j+1) for j in range(data_inst.p)])
    # diag_inst13.show_traceplot((2,2), [0,1,2,3])
    



    #index   0     1        2     3        4            5
    #sample [mu_u, Sigma_u, mu_v, Sigma_v, [u1,...,un], [v1,...,vp]]

    k = 8
    initial2 = [np.array([0 for _ in range(k)]), np.identity(k), np.array([0 for _ in range(k)]), np.identity(k),
                [np.array([0 for _ in range(k)]) for _ in range(data_inst.n)], [np.array([0 for _ in range(k)]) for _ in range(data_inst.p)]]
    gibbs_inst2 = Gibbs_HW5P6(initial2, hyper_k0 = 1, MovieRating_Data_inst = data_inst)
    gibbs_inst2.generate_samples(3000)
    gibbs_inst2.MC_sample = gibbs_inst2.MC_sample[500:] #burn-in

    diag_inst21 = MCMC_Diag()
    diag_inst21.set_mc_samples_from_list([[np.linalg.norm(sample[0]), np.linalg.det(sample[1]), np.linalg.norm(sample[2]), np.linalg.det(sample[3])] for sample in gibbs_inst2.MC_sample])
    diag_inst21.set_variable_names(["mu_u", "det(Sigma_u)", "mu_v", "det(Sigma_v)"])
    diag_inst21.show_traceplot((2,2))

    u_vec = [sample[4] for sample in gibbs_inst2.MC_sample]
    u_norm_vec = [[np.linalg.norm(u[0]), np.linalg.norm(u[1]), np.linalg.norm(u[2]), np.linalg.norm(u[3])] for u in u_vec]
    
    diag_inst22 = MCMC_Diag()
    diag_inst22.set_mc_samples_from_list(u_norm_vec)
    diag_inst22.set_variable_names(["u1_norm", "u2_norm", "u3_norm", "u4_norm"])
    diag_inst22.show_traceplot((2,2))
    
    v_vec = [sample[5] for sample in gibbs_inst2.MC_sample]
    v_norm_vec = [[np.linalg.norm(v[0]), np.linalg.norm(v[1]), np.linalg.norm(v[2]), np.linalg.norm(v[3])] for v in v_vec]
        
    diag_inst22 = MCMC_Diag()
    diag_inst22.set_mc_samples_from_list(v_norm_vec)
    diag_inst22.set_variable_names(["v1_norm", "v2_norm", "v3_norm", "v4_norm"])
    diag_inst22.show_traceplot((2,2))

    # u_vec_mean = np.mean(u_vec, axis=0)
    # print(u_vec_mean.shape)
    # print(u_vec_mean)
    
    v_vec_mean = np.mean(v_vec, axis=0)
    print(v_vec_mean.shape)
    print(v_vec_mean)
    