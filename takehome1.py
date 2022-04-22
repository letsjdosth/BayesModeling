import csv
from random import normalvariate, seed
from math import exp, log, lgamma
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from bayesian_tools.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag

class MeaslesData:
    def __init__(self):
        self._load()

    def _load(self, file_path = "dataset/measles.csv"):
        self.data_dict = {} 
        self.proportion_dict = {}
        self.sum_dict ={}
        self.index_to_key = {}
        self.key_to_idx = {}
        
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            #header: "cluster","region","y","n"
            next(csv_reader)

            j_index = 0
            for row in csv_reader:
                # cluster = int(row[0])
                region = str(row[1])
                y = int(row[2])
                n = int(row[3])
                
                if region in self.data_dict.keys():
                    self.data_dict[region].append((y,n))
                    self.proportion_dict[region].append(y/n)
                    self.sum_dict[region][0] += y
                    self.sum_dict[region][1] += n
                else:
                    j_index += 1
                    self.data_dict[region] = [(y,n)]
                    self.proportion_dict[region] = [y/n]
                    self.sum_dict[region] = [y, n]
                    self.key_to_idx[region] = j_index
                    self.index_to_key[j_index] = region
                    
    
    def _j_idx_checker(self, j):
        # from 1 to 37
        if j < 1 or j > 37:
            raise ValueError("j should be 1~37")

    def print_all_data(self):
        for key, val in self.data_dict.items():
            print(key, val)
    
    def get_jth_region_name(self, j):
        self._j_idx_checker(j)
        return self.index_to_key[j]

    def get_jth_region_y_n_data(self, j):
        self._j_idx_checker(j)
        return self.data_dict[self.index_to_key[j]]

    def get_jth_region_sum_y_sum_n(self,j):
        self._j_idx_checker(j)
        return self.sum_dict[self.index_to_key[j]]

    def get_jth_region_proportions(self, j):
        self._j_idx_checker(j)
        return self.proportion_dict[self.index_to_key[j]]

    def show_boxplot(self):
        proportions_list = [self.get_jth_region_proportions(j) for j in range(1, 38)]
        plt.boxplot(proportions_list)
        plt.show()
    
    def print_summary(self, round_digit=5):
        print("region | \t mean | \t variance | \t five number summary")
        for j in range(1, 38):
            region = self.index_to_key[j]
            proportions = self.data_dict[region]
            mean_j, var_j = (round(np.mean(proportions),round_digit), round(np.var(proportions), round_digit))
            fiv_number_summary_j = [np.min(proportions)] + list(np.quantile(proportions, [0.25, 0.5, 0.75])) + [np.max(proportions)]
            print(region, " | \t", mean_j, " | \t",var_j, " | \t",fiv_number_summary_j)
        #transform to produce latex table syntax

pregdata_inst = MeaslesData()

# pregdata_inst.show_boxplot()
# print(pregdata_inst.get_jth_region_y_n_data(11))
# print(pregdata_inst.get_jth_region_sum_y_sum_n(11))
# print(pregdata_inst.get_jth_region_proportions(1))
# pregdata_inst.show_boxplot()
# pregdata_inst.print_summary()


class MCMC_Gibbs_TH1(MCMC_Gibbs):
    def __init__(self, initial, hyper_sigma2_mu, hyper_sigma2_d):
        self.MC_sample = [initial]
        self.hyper_sigma2_mu = hyper_sigma2_mu
        self.hyper_sigma2_d = hyper_sigma2_d

    def full_conditional_sampler_mu(self, last_param, j_region):
        #sample: [mu_1,...,mu_J=37, d]
        new_sample = [x for x in last_param]
        
        def log_proposal_pdf(from_smpl, to_smpl):
            #symmetric
            return 0 

        def proposal_sampler(from_smpl, sigma2):
            proposal = normalvariate(from_smpl[0], sigma2**0.5)
            return [proposal]

        def log_target_pdf(eval_pt, d, y_j, n_j):
            r =  eval_pt[0]
            d_odds = (1-d)/d
            log_target_pdf_val = -r**2/(2*self.hyper_sigma2_mu) 
            log_target_pdf_val += (lgamma(d_odds * exp(r) / (1+exp(r)) + y_j) + lgamma(d_odds / (1+exp(r)) + n_j - y_j))
            log_target_pdf_val -= (lgamma(d_odds * exp(r) / (1+exp(r))) + lgamma(d_odds / (1+exp(r))))
            return log_target_pdf_val

        y_j, n_j = pregdata_inst.get_jth_region_sum_y_sum_n(j_region)
        log_target_pdf_with_d = partial(log_target_pdf, d=new_sample[-1], y_j=y_j, n_j=n_j)
        proposal_sampler_with_sigma2 = partial(proposal_sampler, sigma2=0.01)
        initial_val = [new_sample[j_region-1]]
        mc_mh_inst = MCMC_MH(log_target_pdf_with_d, log_proposal_pdf, proposal_sampler_with_sigma2, initial_val)
        mc_mh_inst.generate_samples(3, verbose=False)
        new_logit_mu_j = mc_mh_inst.MC_sample[-1][0]
        new_mu_j = exp(new_logit_mu_j)/(1+exp(new_logit_mu_j))
        new_sample[j_region-1] = new_mu_j

        return new_sample

    
    def full_conditional_sampler_d(self, last_param):
        #sample: [mu_1,...,mu_J=37, d]
        new_sample = [x for x in last_param]

        
        def log_proposal_pdf(from_smpl, to_smpl):
            #symmetric
            return 0 

        def proposal_sampler(from_smpl, sigma2):
            proposal = normalvariate(from_smpl[0], sigma2**0.5)
            return [proposal]
        
        def log_target_pdf(eval_pt, mu_vec):
            q =  eval_pt[0]
            log_target_pdf_val = -q**2/(2*self.hyper_sigma2_d) 
            for j in range(1, 38):
                y_j, n_j = pregdata_inst.get_jth_region_sum_y_sum_n(j)
                log_target_pdf_val += (lgamma(exp(-q)) + lgamma(exp(-q)*mu_vec[j-1] + y_j) + lgamma(exp(-q)*(1-mu_vec[j-1]) + n_j - y_j))
                log_target_pdf_val -= (lgamma(exp(-q)*mu_vec[j-1]) + lgamma(exp(-q)*(1-mu_vec[j-1])) + lgamma(exp(-q) + n_j))
                
            return log_target_pdf_val
        
        log_target_pdf_with_mu_vec = partial(log_target_pdf, mu_vec=new_sample[0:-1])
        proposal_sampler_with_sigma2 = partial(proposal_sampler, sigma2=0.01)
        initial_val = [new_sample[-1]]
        mc_mh_inst = MCMC_MH(log_target_pdf_with_mu_vec, log_proposal_pdf, proposal_sampler_with_sigma2, initial_val)
        mc_mh_inst.generate_samples(3, verbose=False)
        new_logit_d_j = mc_mh_inst.MC_sample[-1][0]
        new_d_j = exp(new_logit_d_j)/(1+exp(new_logit_d_j))
        new_sample[-1] = new_d_j

        return new_sample

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = [x for x in last]
        #update new
        for j in range(1, 38):
            new = self.full_conditional_sampler_mu(new, j)
        new = self.full_conditional_sampler_d(new)
        self.MC_sample.append(new)

if __name__=="__main__":
    seed(20220425)
    initial1 = [0.1 for _ in range(38)]
    print(initial1)
    gibbs_inst1 = MCMC_Gibbs_TH1(initial1, 10, 10) #vague hyperparam
    gibbs_inst1.generate_samples(10000, print_iter_cycle=2000)
    

    diag_inst1 = MCMC_Diag()
    diag_inst1.set_mc_sample_from_MCMC_instance(gibbs_inst1)
    diag_inst1.set_variable_names(["mu"+str(i) for i in range(1,38)]+["d"])
    random_index_list = [1,30,32,37]
    diag_inst1.show_traceplot((2,2), random_index_list)
    diag_inst1.show_acf(30, (2,2), random_index_list)
    diag_inst1.print_summaries(6)

