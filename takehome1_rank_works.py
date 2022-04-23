import matplotlib.pyplot as plt

class ReportRank():
    def __init__(self, p_samples):
        self.MC_sample = p_samples
        self._cal_ranking()

    def _cal_ranking(self):
        self.rank_list_for_each_region_j = [[] for _ in range(37)]
        self.rank_list_for_each_rank_k = [[] for _ in range(37)]
        for sample in self.MC_sample:
            idx_pairs = [(j_minus_1+1, x) for j_minus_1, x in enumerate(sample)]
            idx_pairs.sort(key=lambda x: -x[1])
            sorted_idx = [j for (j, _) in idx_pairs]
            for rank_minus_1, j in enumerate(sorted_idx):
                self.rank_list_for_each_region_j[j-1].append(rank_minus_1+1)
                self.rank_list_for_each_rank_k[rank_minus_1].append(j)

    def rank_frequency_for_fixed_jth_region(self,j):
        return self.rank_list_for_each_region_j[j-1]

    def rank_frequency_for_fixed_kth_ranking(self,k):
        return self.rank_list_for_each_rank_k[k-1]


    def marginal_for_jth_region(self,j):
        #be careful to the index (return list index 0 : rank==1(highst))
        j_th_rank_freq = self.rank_frequency_for_fixed_jth_region(j)
        counting_on_rank_grid = [0 for _ in range(1,38)]
        for rank in j_th_rank_freq:
            counting_on_rank_grid[rank-1] += 1
        freq_on_rank_grid = [x/sum(counting_on_rank_grid) for x in counting_on_rank_grid]
        return freq_on_rank_grid

    def marginal_for_kth_rank(self,k):
        #be careful to the index (return list index 0 : j==1)
        k_th_rank_freq = self.rank_frequency_for_fixed_kth_ranking(k)
        counting_on_region_grid = [0 for _ in range(1,38)]
        for region in k_th_rank_freq:
            counting_on_region_grid[region-1] += 1
        freq_on_region_grid = [x/sum(counting_on_region_grid) for x in counting_on_region_grid]
        return freq_on_region_grid

    def rank_barchart_for_jth_region(self,j):
        rank_grid = [i for i in range(1,38)]
        counting_on_rank_grid = self.marginal_for_jth_region(j)
        plt.bar(rank_grid, counting_on_rank_grid)
        plt.title(str(j)+"th region")
        plt.xlabel('Rank(1==highst)')
        plt.show()

    def rank_barchart_for_kth_ranking(self,k):
        region_grid = [i for i in range(1,38)]
        counting_on_region_grid = self.marginal_for_kth_rank(k)
        plt.bar(region_grid, counting_on_region_grid)
        plt.title(str(k)+"th rank")
        plt.xlabel('Region')
        plt.show()

