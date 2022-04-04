from math import log, exp, pi, tan
from random import choices, random, seed
from statistics import mean, median, variance

import matplotlib.pyplot as plt

seed(20220402)
observations = [43, 44, 45, 46.5, 47.5]

# draw posterior samples using grid approximation
def unnormalized_cauchy_log_pdf(eval_pt, location, scale=1):
    denom = 1+((eval_pt-location)/scale)**2
    return -log(denom)

m = 1000
grid = [i/m for i in range(0, 100*m)]
grid.append(100)
posterior_on_grid = []

for grid_pt in grid:
    obs_log_likelihood = [unnormalized_cauchy_log_pdf(obs, grid_pt) for obs in observations]
    log_posterior_val = sum(obs_log_likelihood) - 2
    posterior_on_grid.append(exp(log_posterior_val))

posterior_normalizer_on_grid = sum(posterior_on_grid)
posterior_on_grid = [val / posterior_normalizer_on_grid for val in posterior_on_grid]

# show (part of a)
plt.plot(grid, posterior_on_grid)
plt.xlabel("theta")
plt.show()

# posterior draws (part of b)
posterior_samples_from_grid = choices(grid, posterior_on_grid, k=1000)
print("==posterior==")
print("mean:", round(mean(posterior_samples_from_grid), 3))
print("median:", round(median(posterior_samples_from_grid), 3))
print("variance:", round(variance(posterior_samples_from_grid), 3))
plt.hist(posterior_samples_from_grid, bins=10)
plt.xlabel("theta")
plt.xlim([0,100])
plt.show()

# posterior predictive draws (part of c)
def cauchy_inverse_cdf(eval_prob, location, scale=1):
    quantile_val = location +  scale * tan(pi*(eval_prob-0.5))
    return quantile_val

predictive_samples = [cauchy_inverse_cdf(random(), loc) for loc in posterior_samples_from_grid]
print("==posterior predictive==")
print("mean:", round(mean(predictive_samples),3))
print("median:", round(median(predictive_samples),3))
print("variance:", round(variance(predictive_samples),3))
plt.hist(predictive_samples, bins=500)
plt.xlabel("y6")
plt.xlim([45-100, 45+100])
plt.show()
