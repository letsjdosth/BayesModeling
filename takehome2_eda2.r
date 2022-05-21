covid_data = read.csv("dataset/covid.csv")
head(covid_data)

covid_data["log_total_cases_per_million"] = log(covid_data["total_cases_per_million"])

library(corrplot)
covid_data_numeric = covid_data[-c(1,2)]
corrplot(cor(covid_data_numeric), method="circle")


chosen_data1 = covid_data[c(
    "log_total_cases_per_million",
    "total_vaccinations_per_hundred",
    "population_density",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "cardiovasc_death_rate", #no
    "diabetes_prevalence", #no
    "male_smokers"
)]
pairs(chosen_data1)

chosen_data2 = covid_data[c(
    "log_total_cases_per_million",
    "total_vaccinations_per_hundred",
    "population_density",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "male_smokers"
)]
pairs(chosen_data2)

#[2] no log
plot(log_total_cases_per_million ~ total_vaccinations_per_hundred, data=chosen_data2)

#[3] log
plot(log_total_cases_per_million ~ population_density, data=chosen_data2)
abline(lm(log_total_cases_per_million ~ population_density, data=chosen_data2))
plot(log_total_cases_per_million ~ log(population_density), data=chosen_data2)
abline(lm(log_total_cases_per_million ~ log(population_density), data=chosen_data2))

#[4]
plot(log_total_cases_per_million ~ aged_65_older, data=chosen_data2)
abline(lm(log_total_cases_per_million ~ aged_65_older, data=chosen_data2))
plot(log_total_cases_per_million ~ log(aged_65_older), data=chosen_data2)
abline(lm(log_total_cases_per_million ~ log(aged_65_older), data=chosen_data2))

#[5] log
plot(log_total_cases_per_million ~ gdp_per_capita, data=chosen_data2)
abline(lm(log_total_cases_per_million ~ gdp_per_capita, data=chosen_data2))
plot(log_total_cases_per_million ~ log(gdp_per_capita), data=chosen_data2)
abline(lm(log_total_cases_per_million ~ log(gdp_per_capita), data=chosen_data2))

#[6]
plot(log_total_cases_per_million ~ hospital_beds_per_thousand, data=chosen_data2)
abline(lm(log_total_cases_per_million ~ hospital_beds_per_thousand, data=chosen_data2))
plot(log_total_cases_per_million ~ log(hospital_beds_per_thousand), data=chosen_data2)
abline(lm(log_total_cases_per_million ~ log(hospital_beds_per_thousand), data=chosen_data2))

#[7]
plot(log_total_cases_per_million ~ male_smokers, data=chosen_data2)
abline(lm(log_total_cases_per_million ~ male_smokers, data=chosen_data2))
plot(log_total_cases_per_million ~ log(male_smokers), data=chosen_data2)
abline(lm(log_total_cases_per_million ~ log(male_smokers), data=chosen_data2))


chosen_data2_log_transformed = chosen_data2
chosen_data2_log_transformed[3] = log(chosen_data2[3])
chosen_data2_log_transformed[5] = log(chosen_data2[5])
pairs(chosen_data2_log_transformed)
corrplot(cor(chosen_data2_log_transformed), method="circle")
