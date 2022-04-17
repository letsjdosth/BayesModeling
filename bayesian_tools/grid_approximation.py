import numpy as np
import matplotlib.pyplot as plt

class GridApprox1D():
    pass

class GridApprox2D():
    def __init__(self, x_start, x_end, x_num, y_start, y_end, y_num):
        self.grid_x = np.linspace(x_start, x_end, x_num)
        self.grid_y = np.linspace(y_start, y_end, y_num)
        self.meshgrid_x, self.meshgrid_y = np.meshgrid(self.grid_x, self.grid_y)

        #names
        self.variable_names = None
        self.graphic_use_variable_name=False

    def set_variable_names(self, name_list):
        if len(name_list)!=2:
            raise ValueError("check your name_list : it should be length 2")
        self.variable_names = name_list
        self.graphic_use_variable_name=True

    def _make_marginal_prob(self):
        self.margianl_x_prob = np.sum(self.level_mat, axis=0)
        self.margianl_x_prob = self.margianl_x_prob/np.sum(self.margianl_x_prob)
        self.marginal_y_prob = np.sum(self.level_mat, axis=1)
        self.marginal_y_prob = self.marginal_y_prob/np.sum(self.marginal_y_prob)

    def make_level_matrix_on_grid_by_loop(self, func):
        #slow!
        self.level_mat = np.zeros(self.meshgrid_x.shape)
        for i in range(self.level_mat.shape[0]):
            for j in range(self.level_mat.shape[1]):
                x = self.meshgrid_x[i,j]
                y = self.meshgrid_y[i,j]
                self.level_mat[i,j] = func(x, y)
        self._make_marginal_prob()
    
    def make_level_matrix_on_grid_by_numpy_oper(self, func_supporting_np_array_oper):
        # func should support a numpy-array type argument (i.e. meshgrid)
        self.level_mat = func_supporting_np_array_oper(self.meshgrid_x, self.meshgrid_y)
        self._make_marginal_prob()

    def summary_from_meshgrid(self, round_ndigits=5):
        x_mean = np.sum(self.meshgrid_x[0,:] * self.margianl_x_prob)
        x2_mean = np.sum((self.meshgrid_x[0,:]**2) * self.margianl_x_prob)
        x_var = x2_mean - x_mean**2

        y_mean = np.sum(self.meshgrid_y[:,0] * self.marginal_y_prob)
        y2_mean = np.sum((self.meshgrid_y[:,0]**2) * self.marginal_y_prob)
        y_var = y2_mean - y_mean**2
        return (round(x_mean, round_ndigits), round(x_var, round_ndigits)), \
                (round(y_mean, round_ndigits), round(y_var, round_ndigits))

    def show_contourplot(self, levels=10, show=True):
        plt.contour(self.meshgrid_x, self.meshgrid_y, self.level_mat, levels=levels)
        if self.graphic_use_variable_name:
            plt.xlabel(self.variable_names[0])
            plt.ylabel(self.variable_names[1])
        if show:
            plt.show()
    
    def show_marginal_curve_specific_dim(self, dim_idx, show=False):
        if dim_idx == 0 or dim_idx == "x":
            grid = self.grid_x
            hist_data = self.margianl_x_prob
        elif dim_idx == 1 or dim_idx == "y":
            grid = self.grid_y
            hist_data = self.marginal_y_prob
        else:
            raise ValueError("check your dim_idx: 0(or 'x') or 1(or 'y')")

        plt.plot(grid, hist_data)
        if self.graphic_use_variable_name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")
        if show:
            plt.show()

    def show_marginal_curve(self, figure_grid_dim, choose_dims=None, show=True):
        grid_row = figure_grid_dim[0]
        grid_column= figure_grid_dim[1]
        if choose_dims is None:
            choose_dims = [0, 1]
       
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, dim_idx in enumerate(choose_dims):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_marginal_curve_specific_dim(dim_idx)
        if show:
            plt.show()

    def sampler(self, num_samples):
        vectorized_mesh_x = np.reshape(self.meshgrid_x, (self.meshgrid_x.size,1))
        vectorized_mesh_y = np.reshape(self.meshgrid_y, (self.meshgrid_y.size,1))
        vectorized_level_mat = np.reshape(self.level_mat, self.level_mat.size)
        rng = np.random.default_rng()
        posterior_samples_idx = rng.choice(np.arange(vectorized_level_mat.size), size=num_samples, p=vectorized_level_mat)
        samples = [(vectorized_mesh_x[i][0], vectorized_mesh_y[i][0]) for i in posterior_samples_idx]
        return samples



if __name__=="__main__":
    observations = np.array([10, 10, 12, 11, 9])
    obs_mean = np.mean(observations)
    obs_S2 = sum([(y-obs_mean)**2 for y in observations])
    print(obs_mean, obs_S2/4) #10.4, 1.3

    import scipy.stats
    def generate_contour_level_matrix_as_true(mu_meshgrid, sigma2_meshgrid):
        term1 = scipy.stats.norm.logpdf(mu_meshgrid, loc=obs_mean, scale=np.sqrt(sigma2_meshgrid/5))
        term2 = scipy.stats.invgamma.logpdf(sigma2_meshgrid, a=2, scale=obs_S2/2)
        # def scipy.stats.invgamma: scaled after inversion from gamma distribution
        # scipy invgamma ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html#scipy.stats.invgamma
        return np.exp(term1+term2)/np.sum(np.exp(term1+term2))

    grid2d_inst = GridApprox2D(obs_mean-3, obs_mean+3, 1000, 0.01, 3, 1000)
    grid2d_inst.set_variable_names(["mu","sigma2"])
    grid2d_inst.make_level_matrix_on_grid_by_numpy_oper(generate_contour_level_matrix_as_true)
    grid2d_inst.show_contourplot()
    grid2d_inst.show_marginal_curve((1,2))
    print(grid2d_inst.summary_from_meshgrid())

    samples = grid2d_inst.sampler(10000)
    from MCMC_Core import MCMC_Diag
    grid2d_diag_inst = MCMC_Diag()
    grid2d_diag_inst.set_mc_samples_from_list(samples)
    grid2d_diag_inst.show_hist((1,2))
    grid2d_diag_inst.print_summaries(5)