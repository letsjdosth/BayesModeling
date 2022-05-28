import numpy as np

class SimDataFactory:
    #fixed parameters
    def __init__(self) -> None:
        self.n = 20 # number of groups, idx: i
        self.m = 10 # numper of samples in each group, idx: j
        self.N = self.n * self.m #total number of samples
        self.p = 5 # number of coefficients, dix: k
        
        self.sigma2 = 1
        self.mu = [1] + [i for i in range(self.p)] # [mu_0, mu_1, ..., mu_5]
        self.tau2 = [1] + [2 for _ in range(self.p)] # [tau2_0, tau2_1, ..., tau2_5]

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
                self.y_ij = self.X[i*self.m + j,:] @ self.beta[i] # + epsilon_ij
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

