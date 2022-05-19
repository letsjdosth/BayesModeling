covid_data = read.csv("dataset/covid.csv")
head(covid_data)

library(corrplot)
covid_data_numeric = covid_data[-c(1,2)]
corrplot(cor(covid_data_numeric), method="circle")


colnames(covid_data)
#indicator
"continent"
"location"

#total
"total_cases_per_million" # y

##====================================================
## exclude
"total_cases"
plot(total_cases_per_million ~ total_cases, data = covid_data)
#exclude it because of its meaning
"total_deaths"
plot(total_cases_per_million ~ total_deaths, data = covid_data)
#exclude it because they have reversed causal relationship

##====================================================
# either one
"total_boosters_per_hundred"
plot(total_cases_per_million ~ total_boosters_per_hundred, data = covid_data)
"total_vaccinations_per_hundred"
plot(total_cases_per_million ~ total_vaccinations_per_hundred, data = covid_data)
"people_fully_vaccinated_per_hundred"
plot(total_cases_per_million ~ people_fully_vaccinated_per_hundred, data = covid_data)

subset_data = covid_data[c("total_boosters_per_hundred", "total_vaccinations_per_hundred", "people_fully_vaccinated_per_hundred")]
pairs(subset_data)

##====================================================
#new (7-day smoothed): do not use
"new_cases"
"new_deaths"
"new_cases_per_million"

##====================================================
#related to covid
# either one (they have almost linear rel, except for extreme values)
"tests_per_case"
plot(total_cases_per_million ~ tests_per_case, data = covid_data)
"positive_rate"#exclude it. too similar meaning with
plot(total_cases_per_million ~ positive_rate, data = covid_data)

##====================================================
#other characteristics
##====================================================
## perhaps pop density
"population"
#Population (latest available values)
plot(total_cases_per_million ~ population, data = covid_data)

"population_density"
#Number of people divided by land area, measured in square kilometers, most recent year available
plot(total_cases_per_million ~ population_density, data = covid_data)

plot(population ~ population_density, data = covid_data) #hmm?

##====================================================
## one of median_age/aged_65_older/aged_70_older
"median_age"
#Median age of the population, UN projection for 2020
plot(total_cases_per_million ~ median_age, data = covid_data)

"aged_65_older"
#Share of the population that is 65 years and older, most recent year available
plot(total_cases_per_million ~ aged_65_older, data = covid_data)

"aged_70_older"
#Share of the population that is 70 years and older in 2015
plot(total_cases_per_million ~ aged_70_older, data = covid_data)

plot(median_age ~ aged_65_older, data = covid_data) #super-colinearity
plot(median_age ~ aged_70_older, data = covid_data)
plot(aged_70_older ~ aged_65_older, data = covid_data)
##====================================================
## one of gdp_per_capita/human_development_index
"gdp_per_capita"
#Gross domestic product at purchasing power parity (constant 2011 international dollars), most recent year available
plot(total_cases_per_million ~ gdp_per_capita, data = covid_data)

"human_development_index"
#A composite index measuring average achievement in three basic dimensions of human development
#a long and healthy life, knowledge and a decent standard of living. Values for 2019
plot(total_cases_per_million ~ human_development_index, data = covid_data)

"life_expectancy"
#Life expectancy at birth in 2019
plot(total_cases_per_million ~ life_expectancy, data = covid_data)


plot(gdp_per_capita ~ human_development_index, data = covid_data) #two are linear

##====================================================

"hospital_beds_per_thousand"
#Hospital beds per 1,000 people, most recent year available since 2010
plot(total_cases_per_million ~ hospital_beds_per_thousand, data = covid_data)

"cardiovasc_death_rate"
#Death rate from cardiovascular disease in 2017 (annual number of deaths per 100,000 people)
plot(total_cases_per_million ~ cardiovasc_death_rate, data = covid_data)

"diabetes_prevalence"
#Diabetes prevalence (% of population aged 20 to 79) in 2017
plot(total_cases_per_million ~ diabetes_prevalence, data = covid_data)

"female_smokers"
#Share of women who smoke, most recent year available
plot(total_cases_per_million ~ female_smokers, data = covid_data)

"male_smokers"
#Share of men who smoke, most recent year available
plot(total_cases_per_million ~ male_smokers, data = covid_data)

subset_data = covid_data[c("hospital_beds_per_thousand","life_expectancy","cardiovasc_death_rate","diabetes_prevalence",
    "female_smokers","male_smokers")]
pairs(subset_data)
