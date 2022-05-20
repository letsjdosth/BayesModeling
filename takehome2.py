import numpy as np
import pandas as pd

from bayesian_tools.linear_model import LM_noninfo_prior
from bayesian_tools.MCMC_Core import MCMC_Diag

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
        #no intercept!
        variable_list = ['continent_Africa', 'continent_Asia', 'continent_Europe', 'continent_North America', 'continent_South America'] + variable_list
        return np.array(self.covid_data_coded_continent[variable_list])

factory_inst = DesignMatrixFactory()
# print(factory_inst.make_design_matrix_with_intercept(["aged_65_older"]))
# print(factory_inst.make_design_matrix_with_continent_indicator(["aged_65_older"]))

model1_y = np.log(factory_inst.make_response_vector(normalize=False))
model1_x = factory_inst.make_design_matrix_with_intercept([
    "total_vaccinations_per_hundred",
    "population_density",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "cardiovasc_death_rate",
    "diabetes_prevalence",
    "male_smokers"
    ])
print(model1_y)
print(model1_x)

lm_inst1 = LM_noninfo_prior(model1_y, model1_x, 20220519)
lm_inst1.generate_samples(10000, print_iter_cycle=2500)

diag_inst = MCMC_Diag()
diag_inst.set_mc_sample_from_MCMC_instance(lm_inst1)
diag_inst.set_variable_names(["sigma2"]+["beta"+str(i) for i in range(9)])
diag_inst.print_summaries(round=8)
lm_inst1.print_freqentist_result()

# diag_inst.show_hist((1,1), [0])
# diag_inst.show_hist((2,3), [1,2,3,4,5,6])
# diag_inst.show_scatterplot(1,2)
