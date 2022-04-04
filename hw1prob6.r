# (part of) BDA3 chap3 problem2 

set.seed(20220403)
dir_prior_parameters = c(0.01, 0.01, 0.01)

alpha1_posterior_samples = rbeta(10000,
    dir_prior_parameters[1] + 294, dir_prior_parameters[2] + 307)
mean(alpha1_posterior_samples)
alpha2_posterior_samples = rbeta(10000,
    dir_prior_parameters[1] + 288, dir_prior_parameters[2] + 332)
mean(alpha2_posterior_samples)

diff_posterior_samples = alpha2_posterior_samples - alpha1_posterior_samples
hist(diff_posterior_samples, breaks=100, 
    main="", xlab="alpha2-alpha1", freq=FALSE)

prop_diff_over0 = sum(diff_posterior_samples>0) / 10000
prop_diff_over0
