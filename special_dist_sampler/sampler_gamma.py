from random import seed, gammavariate
import numpy as np

class GammaBase:
    def __init__(self, set_seed=None):
        if set_seed is not None:
            seed(set_seed)

    def _parameter_support_checker(self, alpha_shape, beta_rate):
        if alpha_shape<=0:
            raise ValueError("alpha should be >0")
        if beta_rate<=0:
            raise ValueError("beta should be >0")

    def sampler(self):
        pass

    def sampler_iter(self, sample_size: int, alpha_shape, beta_rate):
        samples = []
        for _ in range(sample_size):
            samples.append(self.sampler(alpha_shape, beta_rate))
        return samples

class Sampler_univariate_Gamma(GammaBase):
    def __init__(self, set_seed=None):
        super().__init__(set_seed)

    def sampler(self, alpha_shape, beta_rate):
        self._parameter_support_checker(alpha_shape, beta_rate)
        return gammavariate(alpha_shape, 1/beta_rate)
    
class Sampler_univariate_InvGamma(GammaBase):
    def __init__(self, set_seed=None):
        super().__init__(set_seed)
    
    def sampler(self, alpha_shape, beta_rate):
        self._parameter_support_checker(alpha_shape, beta_rate)
        return 1/gammavariate(alpha_shape, 1/beta_rate)

class Sampler_univariate_Chisq(GammaBase):
    # chisq(v) ~ gamma(shape=v/2, rate=1/2)
    def __init__(self, set_seed=None):
        self.gamma_sampler_inst = Sampler_univariate_Gamma(set_seed)

    def sampler_iter(self, sample_size: int, df):
        return self.gamma_sampler_inst.sampler_iter(sample_size, df/2, 0.5)

# class Sampler_univariate_InvChisq():
#     pass

# ============================================================================

class Sampler_Wishart:
    def __init__(self, set_seed=None):
        if set_seed is not None:
            self.random_generator = np.random.default_rng(seed=set_seed)
        else:
            self.random_generator = np.random.default_rng()

    def _parameter_support_checker(self, df, V_scale, p_dim):
        # need cost
        if df <= (p_dim-1):
            raise ValueError("degrees of freedom should be > p-1")
        if not np.allclose(V_scale, V_scale.T, rtol=1e-05, atol=1e-08):
            raise ValueError("V_scale should be symmetric")
        eigvals = np.linalg.eigvals(V_scale)
        if any([val<0 for val in eigvals]):
            raise ValueError("V_scale should be positive definite")

    def _sampler(self, df: int, V_scale: np.array, p_dim):
        # do not use it directly (parameter support check is too costly, so I move it to the head of 'sampler_iter()')
        mvn_samples = self.random_generator.multivariate_normal(np.zeros((p_dim,)), V_scale, size=df)
        wishart_sample = np.zeros(V_scale.shape)
        for mvn_sample in mvn_samples:
            wishart_sample += (np.outer(mvn_sample, mvn_sample))
        return wishart_sample

    def sampler_iter(self, sample_size: int, df: int, V_scale: np.array):
        p_dim = V_scale.shape[0]
        self._parameter_support_checker(df, V_scale, p_dim)

        samples = []
        for _ in range(sample_size):
            samples.append(self._sampler(df, V_scale, p_dim))
        return samples

class Sampler_InvWishart:
    def __init__(self, set_seed=None):
        self.wishart_sampler = Sampler_Wishart(set_seed)
    
    def sampler_iter(self, sample_size: int, df:int, G_scale: np.array):
        # X ~ Wishart(df,V) <=> 1/X ~inv.wishart(df,V^(-1)=G)
        # caution: inefficient (hmm.. how can I improve it?)
        V_scale = np.linalg.inv(G_scale)
        wishart_samples = self.wishart_sampler.sampler_iter(sample_size, df, V_scale)
        return [np.linalg.inv(wishart_sample) for wishart_sample in wishart_samples]

# class Sampler_multivariate_Invgamma:
#     pass


if __name__=="__main__":
    chisq_inst = Sampler_univariate_Chisq(GammaBase)
    v = 2
    test_chisq_samples = chisq_inst.sampler_iter(10000, v)
    from statistics import mean, variance
    print(mean(test_chisq_samples), (v/2)/(1/2), "\n", variance(test_chisq_samples), (v/2)/(1/4))


    wishart_inst = Sampler_Wishart(set_seed=20220420)
    print(wishart_inst.sampler_iter(3, 5, np.array([[2,-1],[-1,3]])))

    inv_wishart_inst = Sampler_InvWishart(set_seed=20220420)
    inv_wishart_samples = inv_wishart_inst.sampler_iter(2, 5, np.array([[2,-1],[-1,3]]))
    print(inv_wishart_samples)