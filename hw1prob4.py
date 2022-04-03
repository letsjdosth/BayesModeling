from random import seed, uniform

from bayesian_tools.MCMC_Core import MCMC_Gibbs, MCMC_Diag

class MCMC_Gibbs_hw1p5(MCMC_Gibbs):
    def __init__(self, initial, c):
        super().__init__(initial)
        self.c = c
    
    def full_conditional_sampler_x(self, last_param):
        new_sample = [x for x in last_param]
        new_x = uniform(max(0, new_sample[1]-self.c), min(1, new_sample[1]+self.c))
        new_sample[0] = new_x
        return new_sample

    def full_conditional_sampler_y(self, last_param):
        new_sample = [x for x in last_param]
        new_y = uniform(max(0, new_sample[0]-self.c), min(1, new_sample[0]+self.c))
        new_sample[1] = new_y
        return new_sample

    def sampler(self):
        last = self.MC_sample[-1]
        new = [x for x in last] #[nu, theta]
        #update new
        new = self.full_conditional_sampler_x(new)
        new = self.full_conditional_sampler_y(new)
        self.MC_sample.append(new)

seed(20220402)

initial = [0.5, 0.5]
gibbs_inst1 = MCMC_Gibbs_hw1p5(initial, c=0.3)
gibbs_inst1.generate_samples(1000)
gibbs_inst1_diag = MCMC_Diag()
gibbs_inst1_diag.set_mc_sample_from_MCMC_instance(gibbs_inst1)
gibbs_inst1_diag.set_variable_names(["x","y"])
gibbs_inst1_diag.show_traceplot((1,2))
# gibbs_inst1_diag.show_acf(30, (1,2))
# gibbs_inst1_diag.show_hist((1,2))
gibbs_inst1_diag.show_scatterplot(0,1)


gibbs_inst2 = MCMC_Gibbs_hw1p5(initial, c=0.05)
gibbs_inst2.generate_samples(1000)
gibbs_inst2_diag = MCMC_Diag()
gibbs_inst2_diag.set_mc_sample_from_MCMC_instance(gibbs_inst2)
gibbs_inst2_diag.set_variable_names(["x","y"])
gibbs_inst2_diag.show_traceplot((1,2))
# gibbs_inst2_diag.show_acf(30, (1,2))
# gibbs_inst2_diag.show_hist((1,2))
gibbs_inst2_diag.show_scatterplot(0,1)


gibbs_inst3 = MCMC_Gibbs_hw1p5(initial, c=0.01)
gibbs_inst3.generate_samples(1000)
gibbs_inst3_diag = MCMC_Diag()
gibbs_inst3_diag.set_mc_sample_from_MCMC_instance(gibbs_inst3)
gibbs_inst3_diag.set_variable_names(["x","y"])
gibbs_inst3_diag.show_traceplot((1,2))
# gibbs_inst3_diag.show_acf(30, (1,2))
# gibbs_inst3_diag.show_hist((1,2))
gibbs_inst3_diag.show_scatterplot(0,1)

