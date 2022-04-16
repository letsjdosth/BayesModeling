from math import log, exp
from random import seed, expovariate, normalvariate
from functools import partial 

from bayesian_tools.MCMC_Core import MCMC_base, MCMC_Diag, MCMC_MH




class CondSampler_hw2p1():
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
    
    def generate_tau(self, num_samples=10000):
        mcmc_mh_inst = self.tau_sampler_factory(initial=0.1, proposal_lambda=(1/15))
        mcmc_mh_inst.generate_samples(num_samples)
        
        mcmc_diag_inst = MCMC_Diag()
        mcmc_diag_inst.set_mc_sample_from_MCMC_instance(mcmc_mh_inst)
        mcmc_diag_inst.show_traceplot((1,1))
        mcmc_diag_inst.show_hist((1,1))
        self.tau_vec = mcmc_diag_inst.MC_sample
    
    def sampler(self, **kwargs):
        tau = self.tau_vec[kwargs["iter"]]
        

    def generate_samples(self):
        pass
        # tau = self.tau_vec[iter_idx]
        # mu = 


if __name__=="__main__":
    cond_sampler_inst = CondSampler_hw2p1(20220415)
    cond_sampler_inst.generate_tau()


    