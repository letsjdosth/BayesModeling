import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayesian_tools.linear_model import LM_noninfo_prior, LM_random_eff_fixed_slope_noninfo_prior, Regression_Model_Checker, InfomationCriteria_for_LM
from bayesian_tools.MCMC_Core import MCMC_Diag
from bayesian_tools.info_criteria import InfomationCriteria

class DesignMatrixFactory:
    def __init__(self):
        self._load()

    def _load(self, file_path = "dataset/covid.csv"):
        self.unnormalized_covid_data = pd.read_csv(file_path)
        self.mean_dict = self.unnormalized_covid_data.mean(numeric_only=True)
        self.scale_dict = self.unnormalized_covid_data.std(numeric_only=True)
        self.unnormalized_response = self.unnormalized_covid_data["total_cases_per_million"]

        #normalize covariates
        self.covid_data=(self.unnormalized_covid_data-self.unnormalized_covid_data.mean(numeric_only=True))/self.unnormalized_covid_data.std(numeric_only=True)
        for object_var in ["continent", "location"]:
            self.covid_data[object_var] = self.unnormalized_covid_data[object_var]
        self.normalized_response = self.covid_data["total_cases_per_million"]

        self.covid_data["intercept"] = np.ones(self.covid_data.shape[0])
        self.covid_data_coded_continent = pd.get_dummies(self.covid_data, columns=["continent"])
        # print(self.covid_data_coded_continent.keys())

    def make_response_vector(self, normalize=True):
        if normalize:
            return np.array(self.normalized_response)
        else:
            return np.array(self.unnormalized_response)

    def make_design_matrix_with_intercept(self, variable_list):
        variable_list = ["intercept"] + variable_list
        return np.array(self.covid_data[variable_list])
    
    def make_design_matrix_without_intercept(self, variable_list):
        return np.array(self.covid_data[variable_list])

    def make_design_matrix_with_continent_indicator(self, variable_list):
        #no intercept! first five are indicator variables for 'continent'
        variable_list = ['continent_Africa', 'continent_Asia', 'continent_Europe', 'continent_North America', 'continent_South America'] + variable_list
        return np.array(self.covid_data_coded_continent[variable_list])

factory_inst = DesignMatrixFactory()
selected_variables = [
    "total_vaccinations_per_hundred",
    "population_density",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    # "cardiovasc_death_rate",
    # "diabetes_prevalence",
    "male_smokers"
    ]

### model1: without continent ###

model1_y = np.log(factory_inst.make_response_vector(normalize=False))
model1_x = factory_inst.make_design_matrix_with_intercept(selected_variables)

lm_inst1 = LM_noninfo_prior(model1_y, model1_x, 20220519)
lm_inst1.generate_samples(10000, print_iter_cycle=2500)

diag_inst1 = MCMC_Diag()
diag_inst1.set_mc_sample_from_MCMC_instance(lm_inst1)
diag_inst1.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(model1_x.shape[1])])
diag_inst1.print_summaries(round=8)
lm_inst1.print_freqentist_result()

# diag_inst1.show_hist((1,1), [0])
# diag_inst1.show_hist((2,3), [1,2,3,4,5,6])
# diag_inst1.show_scatterplot(1,2)

beta_samples1 = [np.array(x[1:]) for x in diag_inst1.MC_sample]
sigma2_samples1 = diag_inst1.get_specific_dim_samples(0)
checker_inst1 = Regression_Model_Checker(model1_y, model1_x, beta_samples1, sigma2_samples1)
checker_inst1.show_residual_plot()
checker_inst1.show_residual_normalProbplot()
for i in range(5):
    checker_inst1.show_posterior_predictive_at_given_data_point(i, show=False)
plt.show()

IC_inst1 = InfomationCriteria_for_LM(model1_y, model1_x, beta_samples1, sigma2_samples1)
print("DIC:", IC_inst1.DIC())
print("DIC_alt:", IC_inst1.DIC_alt())
print("WAIC:", IC_inst1.WAIC())
print("WAIC_alt:", IC_inst1.WAIC_alt())

## cv
## 35: ['United States']
## 29: ['Sri Lanka']
model1_training_without_us_x = np.delete(model1_x, obj=35, axis=0)
model1_training_without_us_y = np.delete(model1_y, obj=35, axis=0)
model1_testing_us_x = model1_x[35,]
model1_testing_us_y = model1_y[35]

lm_inst1_cv1 = LM_noninfo_prior(model1_training_without_us_y, model1_training_without_us_x, 20220519)
lm_inst1_cv1.generate_samples(10000, print_iter_cycle=2500)

diag_inst1_cv1 = MCMC_Diag()
diag_inst1_cv1.set_mc_sample_from_MCMC_instance(lm_inst1_cv1)
diag_inst1_cv1.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(model1_x.shape[1])])
diag_inst1_cv1.print_summaries(round=8)

beta_samples1_cv1 = [np.array(x[1:]) for x in diag_inst1_cv1.MC_sample]
sigma2_samples1_cv1 = diag_inst1_cv1.get_specific_dim_samples(0)
checker_inst1_cv1 = Regression_Model_Checker(model1_training_without_us_y, model1_training_without_us_x, beta_samples1_cv1, sigma2_samples1_cv1)
checker_inst1_cv1.show_posterior_predictive_at_new_point(model1_testing_us_x, model1_testing_us_y)


model1_training_without_sl_x = np.delete(model1_x, obj=29, axis=0)
model1_training_without_sl_y = np.delete(model1_y, obj=29, axis=0)
model1_testing_sl_x = model1_x[29,]
model1_testing_sl_y = model1_y[29]

lm_inst1_cv2 = LM_noninfo_prior(model1_training_without_sl_y, model1_training_without_sl_x, 20220519)
lm_inst1_cv2.generate_samples(10000, print_iter_cycle=2500)

diag_inst1_cv2 = MCMC_Diag()
diag_inst1_cv2.set_mc_sample_from_MCMC_instance(lm_inst1_cv2)
diag_inst1_cv2.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(model1_x.shape[1])])
diag_inst1_cv2.print_summaries(round=8)

beta_samples1_cv2 = [np.array(x[1:]) for x in diag_inst1_cv2.MC_sample]
sigma2_samples1_cv2 = diag_inst1_cv2.get_specific_dim_samples(0)
checker_inst1_cv2 = Regression_Model_Checker(model1_training_without_sl_y, model1_training_without_sl_x, beta_samples1_cv2, sigma2_samples1_cv2)
checker_inst1_cv2.show_posterior_predictive_at_new_point(model1_testing_sl_x, model1_testing_sl_y)



### model2: with continent, random-intercept model ###

model2_y = np.log(factory_inst.make_response_vector(normalize=False))
model2_x = factory_inst.make_design_matrix_with_continent_indicator(selected_variables)
model2_param_dim = model2_x.shape[1]

rnd_eff_indicator = [1 for _ in range(5)]+[0 for _ in range(model2_param_dim-5)]
model2_initial = [[11 for _ in range(5)]+[0 for _ in range(model2_param_dim-5)], 1, 11, 1]
#  0       1       2    3
# [[beta], sigma2, mu0, tau2_0]
lm_randeff_inst2 = LM_random_eff_fixed_slope_noninfo_prior(model2_y, model2_x, rnd_eff_indicator, model2_initial, 20220519)
lm_randeff_inst2.generate_samples(10000)

diag_inst21 = MCMC_Diag()
beta_samples2 = [x[0] for x in lm_randeff_inst2.MC_sample]
diag_inst21.set_mc_samples_from_list(beta_samples2)
diag_inst21.set_variable_names(["beta"+str(i) for i in range(model2_param_dim)])
diag_inst21.burnin(3000)
diag_inst21.print_summaries(round=8)
# diag_inst21.show_hist((3,5))
# diag_inst21.show_traceplot((3,5))
# diag_inst21.show_scatterplot(0,1)
# diag_inst21.show_scatterplot(0,6)
# diag_inst21.show_scatterplot(1,7)
# diag_inst21.show_scatterplot(2,8)

diag_inst22 = MCMC_Diag()
others2 = [x[1:4] for x in lm_randeff_inst2.MC_sample]
diag_inst22.set_mc_samples_from_list(others2)
diag_inst22.set_variable_names(["sigma2", "mu0", "tau2_0"])
diag_inst22.burnin(3000)
diag_inst22.print_summaries(round=8)
diag_inst22.show_hist((1,3))
diag_inst22.show_traceplot((1,3))
diag_inst22.show_scatterplot(1,2)


beta_samples2 = diag_inst21.MC_sample
sigma2_samples2 = diag_inst22.get_specific_dim_samples(0)
checker_inst2 = Regression_Model_Checker(model2_y, model2_x, beta_samples2, sigma2_samples2)
checker_inst2.show_residual_plot()
checker_inst2.show_residual_normalProbplot()
for i in range(5):
    checker_inst2.show_posterior_predictive_at_given_data_point(i, show=False)
plt.show()

IC_inst2 = InfomationCriteria_for_LM(model2_y, model2_x, beta_samples2, sigma2_samples2)
print("DIC:", IC_inst2.DIC())
print("DIC_alt:", IC_inst2.DIC_alt())
print("WAIC:", IC_inst2.WAIC())
print("WAIC_alt:", IC_inst2.WAIC_alt())


## cv
## 35: ['United States']
## 29: ['Sri Lanka']
model2_training_without_us_x = np.delete(model2_x, obj=35, axis=0)
model2_training_without_us_y = np.delete(model2_y, obj=35, axis=0)
model2_testing_us_x = model2_x[35,]
model2_testing_us_y = model2_y[35]

lm_randeff_inst2_cv1 = LM_random_eff_fixed_slope_noninfo_prior(model2_training_without_us_y, model2_training_without_us_x, rnd_eff_indicator, model2_initial, 20220519)
lm_randeff_inst2_cv1.generate_samples(10000, print_iter_cycle=2500)

diag_inst21_cv1 = MCMC_Diag()
beta_samples2_cv1 = [x[0] for x in lm_randeff_inst2_cv1.MC_sample]
diag_inst21_cv1.set_mc_samples_from_list(beta_samples2_cv1)
diag_inst21_cv1.set_variable_names(["beta"+str(i) for i in range(model2_param_dim)])
diag_inst21_cv1.burnin(3000)
diag_inst21_cv1.print_summaries(round=8)

diag_inst22_cv1 = MCMC_Diag()
others2_cv1 = [x[1:4] for x in lm_randeff_inst2_cv1.MC_sample]
diag_inst22_cv1.set_mc_samples_from_list(others2_cv1)
diag_inst22_cv1.set_variable_names(["sigma2", "mu0", "tau2_0"])
diag_inst22_cv1.burnin(3000)
diag_inst22_cv1.print_summaries(round=8)

sigma2_samples2_cv1 = diag_inst22_cv1.get_specific_dim_samples(0)
checker_inst2_cv1 = Regression_Model_Checker(model2_training_without_us_y, model2_training_without_us_x, beta_samples2_cv1, sigma2_samples2_cv1)
checker_inst2_cv1.show_posterior_predictive_at_new_point(model2_testing_us_x, model2_testing_us_y)


model2_training_without_sl_x = np.delete(model2_x, obj=29, axis=0)
model2_training_without_sl_y = np.delete(model2_y, obj=29, axis=0)
model2_testing_sl_x = model2_x[29,]
model2_testing_sl_y = model2_y[29]

lm_randeff_inst2_cv2 = LM_random_eff_fixed_slope_noninfo_prior(model2_training_without_sl_y, model2_training_without_sl_x, rnd_eff_indicator, model2_initial, 20220519)
lm_randeff_inst2_cv2.generate_samples(10000, print_iter_cycle=2500)

diag_inst21_cv2 = MCMC_Diag()
beta_samples2_cv2 = [x[0] for x in lm_randeff_inst2_cv2.MC_sample]
diag_inst21_cv2.set_mc_samples_from_list(beta_samples2_cv2)
diag_inst21_cv2.set_variable_names(["beta"+str(i) for i in range(model2_param_dim)])
diag_inst21_cv2.burnin(3000)
diag_inst21_cv2.print_summaries(round=8)

diag_inst22_cv2 = MCMC_Diag()
others2_cv2 = [x[1:4] for x in lm_randeff_inst2_cv2.MC_sample]
diag_inst22_cv2.set_mc_samples_from_list(others2_cv2)
diag_inst22_cv2.set_variable_names(["sigma2", "mu0", "tau2_0"])
diag_inst22_cv2.burnin(3000)
diag_inst22_cv2.print_summaries(round=8)

sigma2_samples2_cv2 = diag_inst22_cv2.get_specific_dim_samples(0)
checker_inst2_cv2 = Regression_Model_Checker(model2_training_without_sl_y, model2_training_without_sl_x, beta_samples2_cv2, sigma2_samples2_cv2)
checker_inst2_cv2.show_posterior_predictive_at_new_point(model2_testing_sl_x, model2_testing_sl_y)


# ### model3: with continent, fixed effect model ###

lm_fixeff_inst3 = LM_noninfo_prior(model2_y, model2_x, 20220519)
lm_fixeff_inst3.generate_samples(10000, print_iter_cycle=2500)
diag_inst3 = MCMC_Diag()
diag_inst3.set_mc_sample_from_MCMC_instance(lm_fixeff_inst3)
diag_inst3.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(model2_param_dim)])
diag_inst3.print_summaries(round=8)
diag_inst3.show_hist((3,5))

sigma2_samples3 = diag_inst3.get_specific_dim_samples(0)
beta_samples3 = [np.array(x[1:]) for x in diag_inst3.MC_sample]
checker_inst2 = Regression_Model_Checker(model2_y, model2_x, beta_samples3, sigma2_samples3)
checker_inst2.show_residual_plot()
checker_inst2.show_residual_normalProbplot()
for i in range(5):
    checker_inst2.show_posterior_predictive_at_given_data_point(i, show=False)
plt.show()

IC_inst3 = InfomationCriteria_for_LM(model2_y, model2_x, beta_samples3, sigma2_samples3)
print("DIC:", IC_inst3.DIC())
print("DIC_alt:", IC_inst3.DIC_alt())
print("WAIC:", IC_inst3.WAIC())
print("WAIC_alt:", IC_inst3.WAIC_alt())


## cv
## 35: ['United States']
## 29: ['Sri Lanka']
model3_training_without_us_x = np.delete(model2_x, obj=35, axis=0)
model3_training_without_us_y = np.delete(model2_y, obj=35, axis=0)
model3_testing_us_x = model2_x[35,]
model3_testing_us_y = model2_y[35]

lm_inst3_cv1 = LM_noninfo_prior(model3_training_without_us_y, model3_training_without_us_x, 20220519)
lm_inst3_cv1.generate_samples(10000, print_iter_cycle=2500)

diag_inst3_cv1 = MCMC_Diag()
diag_inst3_cv1.set_mc_sample_from_MCMC_instance(lm_inst3_cv1)
diag_inst3_cv1.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(model1_x.shape[1])])
diag_inst3_cv1.print_summaries(round=8)

beta_samples3_cv1 = [np.array(x[1:]) for x in diag_inst3_cv1.MC_sample]
sigma2_samples3_cv1 = diag_inst3_cv1.get_specific_dim_samples(0)
checker_inst3_cv1 = Regression_Model_Checker(model3_training_without_us_y, model3_training_without_us_x, beta_samples3_cv1, sigma2_samples3_cv1)
checker_inst3_cv1.show_posterior_predictive_at_new_point(model3_testing_us_x, model3_testing_us_y)


model3_training_without_sl_x = np.delete(model2_x, obj=29, axis=0)
model3_training_without_sl_y = np.delete(model2_y, obj=29, axis=0)
model3_testing_sl_x = model2_x[29,]
model3_testing_sl_y = model2_y[29]


lm_inst3_cv2 = LM_noninfo_prior(model3_training_without_sl_y, model3_training_without_sl_x, 20220519)
lm_inst3_cv2.generate_samples(10000, print_iter_cycle=2500)

diag_inst3_cv2 = MCMC_Diag()
diag_inst3_cv2.set_mc_sample_from_MCMC_instance(lm_inst3_cv2)
diag_inst3_cv2.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(model1_x.shape[1])])
diag_inst3_cv2.print_summaries(round=8)

beta_samples3_cv2 = [np.array(x[1:]) for x in diag_inst3_cv2.MC_sample]
sigma2_samples3_cv2 = diag_inst3_cv2.get_specific_dim_samples(0)
checker_inst3_cv2 = Regression_Model_Checker(model3_training_without_sl_y, model3_training_without_sl_x, beta_samples3_cv2, sigma2_samples3_cv2)
checker_inst3_cv2.show_posterior_predictive_at_new_point(model3_testing_sl_x, model3_testing_sl_y)