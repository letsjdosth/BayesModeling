import numpy as np
import matplotlib.pyplot as plt

from special_dist_sampler.sampler_dirichlet import Sampler_Dirichlet
from special_dist_sampler.sampler_multinomial import Sampler_multinomial

from bayesian_tools.MCMC_Core import MCMC_Gibbs, MCMC_Diag

# Naive approach
## Using full posterior

prior_dir_parameter = np.array([1, 1, 1, 1]) #uniform
obs_data = np.array([1439, 78, 15, 16])
posterior_dir_parameter = prior_dir_parameter + obs_data

dir_sampler_inst = Sampler_Dirichlet()
dir_post_samples = np.array(dir_sampler_inst.sampler_iter(1000, posterior_dir_parameter))


print("mean:", round(np.mean(dir_post_samples[:,0]), 6))
print("median:", round(np.median(dir_post_samples[:,0]), 6))
print("var:", round(np.var(dir_post_samples[:,0]), 6))
print("95%CI:", [round(x, 6) for x in np.quantile(dir_post_samples[:,0], [0.025, 0.975])])

plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(range(len(dir_post_samples[:,i])), dir_post_samples[:,i])
    plt.xlabel("theta"+str(i+1))
    plt.axhline(np.mean(dir_post_samples[:,i]), color="red", linestyle="solid", linewidth=0.8)
    plt.axhline(obs_data[i]/sum(obs_data), color="blue", linestyle="solid", linewidth=0.8)
plt.show()


## Using marginal posterior directly
prior_beta_parameter = np.array([1, 3])
obs_marginal_data = np.array([1439, 78 + 15 + 16])
posterior_beta_parameter = prior_beta_parameter + obs_marginal_data

beta_sampler_inst = np.random.default_rng()
beta_post_samples = beta_sampler_inst.beta(posterior_beta_parameter[0], posterior_beta_parameter[1], 1000)

print("mean:", round(np.mean(beta_post_samples), 6))
print("median:", round(np.median(beta_post_samples), 6))
print("var:", round(np.var(beta_post_samples), 6))
print("95%CI:", [round(x, 6) for x in np.quantile(beta_post_samples, [0.025, 0.975])])


# considering MAR
class Gibbs_SloveniaData(MCMC_Gibbs):
    def __init__(self, initial_p, prior_param, set_seed):
        self.MC_sample = [initial_p]
        self.mn_sampler = Sampler_multinomial(set_seed=set_seed)
        self.dir_sampler = Sampler_Dirichlet(set_seed=set_seed)
        self.prior_param = prior_param

        self.obs = [[1439, 67, 159], [16, 16, 32], [144, 54, 136]]

    def full_conditional_sampler_missings(self, last_p):
        p11, p12, p21, p22 = tuple(last_p)
        
        x_1x_vec = self.mn_sampler.sampler(self.obs[0][2], [p11/(p11+p12), p12/(p11+p12)])
        x_2x_vec = self.mn_sampler.sampler(self.obs[1][2], [p21/(p21+p22), p22/(p21+p22)])
        x_x1_vec = self.mn_sampler.sampler(self.obs[2][0], [p11/(p11+p21), p21/(p11+p21)])
        x_x2_vec = self.mn_sampler.sampler(self.obs[2][1], [p12/(p12+p22), p22/(p12+p22)])
        x_xx_vec = self.mn_sampler.sampler(self.obs[2][2], [p11, p12, p21, p22])

        y11 = self.obs[0][0] + x_1x_vec[0] + x_x1_vec[0] + x_xx_vec[0]
        y12 = self.obs[0][1] + x_1x_vec[1] + x_x2_vec[0] + x_xx_vec[1]
        y21 = self.obs[1][0] + x_2x_vec[0] + x_x1_vec[1] + x_xx_vec[2]
        y22 = self.obs[1][1] + x_2x_vec[1] + x_x2_vec[1] + x_xx_vec[3]
        return (y11, y12, y21, y22)
    
    def full_conditional_sampler_p(self, last_p, now_y):
        posterior_param  = [a + y for a,y in zip(last_p, now_y)]
        return self.dir_sampler.sampler(posterior_param)

    def sampler(self):
        last_p = self.MC_sample[-1]
        new_y = self.full_conditional_sampler_missings(last_p)
        new_p = self.full_conditional_sampler_p(last_p, new_y)
        self.MC_sample.append(new_p)

gibbs_inst1 = Gibbs_SloveniaData([0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1], 20220406)
gibbs_inst1.generate_samples(10000, print_iter_cycle=2000)
gibbs_diag_inst1 = MCMC_Diag()
gibbs_diag_inst1.set_mc_sample_from_MCMC_instance(gibbs_inst1)
gibbs_diag_inst1.set_variable_names(["p11","p12","p21","p22"])
gibbs_diag_inst1.burnin(2000)
gibbs_diag_inst1.show_acf(30, (2,2))

gibbs_diag_inst1.print_summaries(round=3)
gibbs_diag_inst1.graphic_traceplot_mean=True
gibbs_diag_inst1.show_traceplot((2,2))

#p11
print("p11 ESS:", gibbs_diag_inst1.effective_sample_size(0))
gibbs_diag_inst1.show_hist_specific_dim(0, True)
