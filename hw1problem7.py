import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


observations = np.array([10, 10, 12, 11, 9])
obs_mean = np.mean(observations)
obs_S2 = sum([(y-obs_mean)**2 for y in observations])
print(obs_mean, obs_S2/4)

# grid_x = np.linspace(9, 12, 1000)
grid_x = np.linspace(obs_mean-3, obs_mean+3, 1000)
grid_y = np.linspace(0.01, 3, 1000)
meshgrid_x, meshgrid_y = np.meshgrid(grid_x, grid_y)

# def generate_contour_level_matrix_as_true(mu_meshgrid, sigma2_meshgrid):
#     value_mat = np.zeros(mu_meshgrid.shape)
#     for i in range(mu_meshgrid.shape[0]):
#         for j in range(mu_meshgrid.shape[1]):
#             mu = mu_meshgrid[i,j]
#             sigma2 = sigma2_meshgrid[i,j]
#             term1 = scipy.stats.norm.pdf(mu, loc=obs_mean, scale=np.sqrt(sigma2/5))
#             term2 = scipy.stats.invgamma.pdf(sigma2, a=2, scale=obs_S2/2)
#             value_mat[i,j] = term1*term2
#     return value_mat

def generate_contour_level_matrix_as_true(mu_meshgrid, sigma2_meshgrid):
    term1 = scipy.stats.norm.logpdf(mu_meshgrid, loc=obs_mean, scale=np.sqrt(sigma2_meshgrid/5))
    term2 = scipy.stats.invgamma.logpdf(sigma2_meshgrid, a=2, scale=obs_S2/2)
    #def scipy.stats.invgamma: scaled after inversion from gamma distribution
    return np.exp(term1+term2)/np.sum(np.exp(term1+term2))

value_mesh_as_true = generate_contour_level_matrix_as_true(meshgrid_x, meshgrid_y)


def generate_contour_level_matrix_as_rounded(mu_meshgrid, sigma2_meshgrid):
    z_mesh = np.ones(mu_meshgrid.shape)
    for obs in observations:
        term1 = scipy.stats.norm.cdf(obs, loc=mu_meshgrid-0.5, scale=np.sqrt(sigma2_meshgrid))
        term2 = scipy.stats.norm.cdf(obs, loc=mu_meshgrid+0.5, scale=np.sqrt(sigma2_meshgrid))
        z_mesh *= (term1 - term2)
    z_mesh = z_mesh/sigma2_meshgrid
    return z_mesh/np.sum(z_mesh)

value_mesh_as_rounded = generate_contour_level_matrix_as_rounded(meshgrid_x, meshgrid_y)



def mean_from_meshgrid(meshgrid_x, meshgrid_y, meshgrid_val, print_round=5):
    margianl_x_prob = np.sum(meshgrid_val, axis=0)
    margianl_x_prob = margianl_x_prob/np.sum(margianl_x_prob)
    x_mean = np.sum(meshgrid_x[0,:] * margianl_x_prob)
    x2_mean = np.sum((meshgrid_x[0,:]**2) * margianl_x_prob)
    x_var = x2_mean - x_mean**2

    marginal_y_prob = np.sum(meshgrid_val, axis=1)
    marginal_y_prob = marginal_y_prob/np.sum(marginal_y_prob)
    y_mean = np.sum(meshgrid_y[:,0] * marginal_y_prob)
    y2_mean = np.sum((meshgrid_y[:,0]**2) * marginal_y_prob)
    y_var = y2_mean - y_mean**2
    return (round(x_mean, print_round), round(x_var, print_round)), (round(y_mean, print_round), round(y_var, print_round))

print(mean_from_meshgrid(meshgrid_x, meshgrid_y, value_mesh_as_true))
print(mean_from_meshgrid(meshgrid_x, meshgrid_y, value_mesh_as_rounded))


ax1 = plt.subplot(121)
plt.contour(meshgrid_x, meshgrid_y, value_mesh_as_true, levels=10)
ax2 = plt.subplot(122, sharex=ax1)
plt.contour(meshgrid_x, meshgrid_y, value_mesh_as_rounded, levels=10)
plt.show()


# # choose
# vectorized_mesh_x = np.reshape(meshgrid_x, (meshgrid_x.size,1))
# vectorized_mesh_y = np.reshape(meshgrid_y, (meshgrid_y.size,1))
# vectorized_mesh_as_rounded = np.reshape(value_mesh_as_rounded, value_mesh_as_rounded.size)
# print(vectorized_mesh_as_rounded.size)
# rng = np.random.default_rng()
# posterior_samples_idx = rng.choice(np.arange(vectorized_mesh_as_rounded.size), size=5, p=vectorized_mesh_as_rounded)
# print(posterior_samples_idx)


# z1_vec = []
# z2_vec = []
# for idx in posterior_samples_idx:
#     rng.normal(loc=vectorized_mesh_x[idx], scale=np.sqrt(vectorized_mesh_y[idx]))
