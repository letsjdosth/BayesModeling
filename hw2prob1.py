from math import log, exp, inf
from random import seed, expovariate, normalvariate
from functools import partial 

from bayesian_tools.MCMC_Core import MCMC_base, MCMC_Diag, MCMC_MH




class CondSampler_hw2p1a(MCMC_base):
    def __init__(self, random_seed):
        self.MC_sample = []
        self.y = [28,  8, -3,  7, -1,  1, 18, 12]
        self.sigma = [15, 10, 16, 11,  9, 11, 10, 18]
        self.seed = random_seed
        seed(random_seed)
    
    def tau_sampler_factory(self, initial, proposal_lambda):
        def proposal_sampler(_, lambd):
            #indep sampler
            return [expovariate(lambd)]

        def log_proposal_pdf(from_smpl, to_smpl, lambd):
            return log(lambd) - lambd * to_smpl[0]

        def log_target_pdf(smpl):
            tau = smpl[0]
            V_mu_inv = sum([1/(s**2+tau**2) for s in self.sigma])
            mu_hat = sum([y/(s**2+tau**2) for y, s in zip(self.y, self.sigma)]) / V_mu_inv
            term1 = -0.5 * log(V_mu_inv)
            term2 = -0.5 * sum([log(s**2+tau**2) for s in self.sigma])
            term3 = -0.5 * sum([1/(s**2+tau**2) * (y - mu_hat) for y, s in zip(self.y, self.sigma)])
            return term1 + term2 + term3
        
        log_proposal_pdf_with_fixed_lambda = partial(log_proposal_pdf, lambd=proposal_lambda)    
        proposal_sampler_with_fixed_lambda = partial(proposal_sampler, lambd=proposal_lambda)
        mcmc_initial = [initial]
        mcmc_mh_inst = MCMC_MH(
            log_target_pdf, log_proposal_pdf_with_fixed_lambda, proposal_sampler_with_fixed_lambda, 
            mcmc_initial, self.seed+1)
        return mcmc_mh_inst
    
    def generate_tau(self, num_samples):
        mcmc_mh_inst = self.tau_sampler_factory(initial=0.1, proposal_lambda=(1/15))
        print("--generating tau --")
        mcmc_mh_inst.generate_samples(num_samples, verbose=False, print_iter_cycle=2000)
        
        mcmc_diag_inst = MCMC_Diag()
        mcmc_diag_inst.set_mc_sample_from_MCMC_instance(mcmc_mh_inst)
        # mcmc_diag_inst.show_traceplot((1,1))
        # mcmc_diag_inst.show_hist((1,1))
        self.tau_vec = mcmc_diag_inst.MC_sample[1:]
    
    def sampler(self, **kwargs):
        tau = self.tau_vec[kwargs["iter_idx"]][0]

        V_mu_inv = sum([1/(s**2+tau**2) for s in self.sigma])
        mu_hat = sum([y/(s**2+tau**2) for y, s in zip(self.y, self.sigma)]) / V_mu_inv
        mu = normalvariate(mu_hat, (1/V_mu_inv)**0.5)

        new_sample = [tau, mu, []]

        # [tau, mu, theta1, theta2, ..., theta_8]
        for i in range(8):
            V_j = 1/(1/self.sigma[i]**2 + 1/tau**2)
            theta_j_hat = (self.y[i]/self.sigma[i]**2 + mu/tau**2) * V_j
            new_sample[2].append(normalvariate(theta_j_hat, V_j**0.5))
        
        self.MC_sample.append(new_sample)

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        self.generate_tau(num_samples+1)
        super().generate_samples(num_samples, pid, verbose, print_iter_cycle)


class CondSampler_hw2p1b(MCMC_base):
    def __init__(self, random_seed):
        self.MC_sample = []
        self.y = [28,  8, -3,  7, -1,  1, 18, 12]
        self.sigma = [15, 10, 16, 11,  9, 11, 10, 18]
        self.seed = random_seed
        seed(random_seed)
        
    def sampler(self, **kwargs):
        tau = inf
        mu = None
        new_sample = [tau, mu, []]
        # [tau, mu, theta1, theta2, ..., theta_8]
        for i in range(8):
            V_j = self.sigma[i]**2
            theta_j_hat = self.y[i]
            new_sample[2].append(normalvariate(theta_j_hat, V_j**0.5))
        
        self.MC_sample.append(new_sample)

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        super().generate_samples(num_samples, pid, verbose, print_iter_cycle)


class CondSampler_hw2p1d(CondSampler_hw2p1a):
    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        self.tau_vec = [[0.00000000001] for _ in range(num_samples+1)]
        super(CondSampler_hw2p1a, self).generate_samples(num_samples, pid, verbose, print_iter_cycle)


#========================

def best_counting(mc_samples_theta_is, prob=True):
    max_case_counting = [0 for _ in range(8)]
    for sample in mc_samples_theta_is:
        max_idx = sample.index(max(sample))
        max_case_counting[max_idx] += 1
    
    if prob:
        return [round(x/sum(max_case_counting),5) for x in max_case_counting]
    else:
        return max_case_counting


def pair_comparing_counting(pair_0, pair_1, mc_samples_theta_is, prob=True):
    #pair_0, pair_1: 1,...,8
    better_case_counting = [0 for _ in range(2)]
    for sample in mc_samples_theta_is:
        pair_0_theta = sample[pair_0-1]
        pair_1_theta = sample[pair_1-1]
        if pair_0_theta > pair_1_theta:
            better_case_counting[0] += 1
        elif pair_1_theta > pair_0_theta:
            better_case_counting[1] += 1
    if prob:
        return [round(x/sum(better_case_counting),5) for x in better_case_counting]
    else:
        return better_case_counting

if __name__=="__main__":
    #3-(a)
    cond_sampler_inst_a = CondSampler_hw2p1a(20220415)
    cond_sampler_inst_a.generate_samples(10000, print_iter_cycle=5000)
    
    diag_inst_a1 = MCMC_Diag()
    tau_mu_samples_a = [sample[:2] for sample in cond_sampler_inst_a.MC_sample]
    diag_inst_a1.set_mc_samples_from_list(tau_mu_samples_a)
    diag_inst_a1.set_variable_names(["tau","mu"])
    diag_inst_a1.show_traceplot((1,2))
    diag_inst_a1.show_hist((1,2))
    
    diag_inst_a2 = MCMC_Diag()
    theta_i_samples_a = [sample[2] for sample in cond_sampler_inst_a.MC_sample]
    diag_inst_a2.set_mc_samples_from_list(theta_i_samples_a)
    diag_inst_a2.set_variable_names(["theta1","theta2","theta3","theta4","theta5","theta6","theta7","theta8"])
    diag_inst_a2.show_hist((2,4))
    
    print("probability that ith coaching is the best:\n", best_counting(theta_i_samples_a))
    for i in range(1, 9):
        for j in range(i+1, 9):
            print("school", i, "vs school", j, ":", pair_comparing_counting(i, j, theta_i_samples_a))
    
    #3-(b)
    cond_sampler_inst_b = CondSampler_hw2p1b(20220415+2)
    cond_sampler_inst_b.generate_samples(10000, print_iter_cycle=5000)
    diag_inst_b2 = MCMC_Diag()
    theta_i_samples_b = [sample[2] for sample in cond_sampler_inst_b.MC_sample]
    diag_inst_b2.set_mc_samples_from_list(theta_i_samples_b)
    diag_inst_b2.set_variable_names(["theta1","theta2","theta3","theta4","theta5","theta6","theta7","theta8"])
    diag_inst_b2.show_hist((2,4))
    print("probability that ith coaching is the best:\n", best_counting(theta_i_samples_b))
    for i in range(1, 9):
        for j in range(i+1, 9):
            print("school", i, "vs school", j, ":", pair_comparing_counting(i, j, theta_i_samples_b))
    

    #3-(d)
    cond_sampler_inst_d = CondSampler_hw2p1d(20220415+4)
    cond_sampler_inst_d.generate_samples(10000, print_iter_cycle=5000)
    
    diag_inst_d1 = MCMC_Diag()
    tau_mu_samples_d = [sample[:2] for sample in cond_sampler_inst_d.MC_sample]
    diag_inst_d1.set_mc_samples_from_list(tau_mu_samples_d)
    diag_inst_d1.set_variable_names(["tau","mu"])
    diag_inst_d1.show_hist_specific_dim(0)
    
    diag_inst_d2 = MCMC_Diag()
    theta_i_samples_d = [sample[2] for sample in cond_sampler_inst_d.MC_sample]
    diag_inst_d2.set_mc_samples_from_list(theta_i_samples_d)
    diag_inst_d2.set_variable_names(["theta1","theta2","theta3","theta4","theta5","theta6","theta7","theta8"])
    diag_inst_d2.show_hist((2,4))
    
    print("probability that ith coaching is the best:\n", best_counting(theta_i_samples_d))
    for i in range(1, 9):
        for j in range(i+1, 9):
            print("school", i, "vs school", j, ":", pair_comparing_counting(i, j, theta_i_samples_d))