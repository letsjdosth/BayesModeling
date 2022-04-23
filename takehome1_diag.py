from random import seed

from bayesian_tools.MCMC_Core import MCMC_Diag

import takehome1_sampler as tk1
from takehome1_rank_works import ReportRank

if __name__=="__main__":
    seed(20220423)

    # eda
    print(tk1.measles_data_inst.get_jth_region_y_n_data(11))
    print(tk1.measles_data_inst.get_jth_region_sum_y_sum_n(11))
    print(tk1.measles_data_inst.get_jth_region_proportions(1))
    tk1.measles_data_inst.show_boxplot()
    tk1.measles_data_inst.print_summary()

    high_rank_list = [j-1 for j in [10, 9, 33, 6, 13]]
    low_rank_list = [j-1 for j in [31, 30, 21, 29, 28]]
    
    # indep model
    indepmodel_sampler_inst = tk1.CondSampler_IndepModel_TH1(1,1) #vague prior
    indepmodel_sampler_inst.generate_samples(30000, print_iter_cycle=10000)
    
    indepmodel_diag_inst = MCMC_Diag()
    indepmodel_diag_inst.set_mc_sample_from_MCMC_instance(indepmodel_sampler_inst)
    indepmodel_diag_inst.set_variable_names(["p"+str(i) for i in range(1,38)])
    indepmodel_diag_inst.print_summaries(round=4, latex_table_format=True)
    indepmodel_diag_inst.show_hist((2,3), high_rank_list)
    indepmodel_diag_inst.show_hist((2,3), low_rank_list)


    indepmodel_report_rank_inst = ReportRank(indepmodel_sampler_inst.MC_sample)
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(1) #10
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(2) #9
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(3) #33
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(4) #6
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(5) #13
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(37) #31
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(36) #30
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(35) #21
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(34) #29
    indepmodel_report_rank_inst.rank_barchart_for_kth_ranking(33) #28

    indepmodel_report_rank_inst.rank_barchart_for_jth_region(10)
    indepmodel_report_rank_inst.rank_barchart_for_jth_region(9)
    indepmodel_report_rank_inst.rank_barchart_for_jth_region(33)
    indepmodel_report_rank_inst.rank_barchart_for_jth_region(31)
    indepmodel_report_rank_inst.rank_barchart_for_jth_region(30)
    indepmodel_report_rank_inst.rank_barchart_for_jth_region(21)


    # hierarchical model    
    hmodel_diag_inst1 = MCMC_Diag()
    hmodel_diag_inst1.set_mc_sample_from_csv("takehome1_mu_d_samples")
    hmodel_diag_inst1.set_variable_names(["mu"+str(i) for i in range(1,38)]+["d"])
    hmodel_diag_inst1.burnin(5000)
    hmodel_diag_inst1.thinning(20)

    
    ## for d and mu
    hmodel_diag_inst1.show_traceplot((2,3), high_rank_list+[37])
    hmodel_diag_inst1.show_acf(30, (2,3), high_rank_list+[37])
    hmodel_diag_inst1.show_hist((2,3), high_rank_list)
    hmodel_diag_inst1.show_traceplot((2,3), low_rank_list+[37])
    hmodel_diag_inst1.show_acf(30, (2,3), low_rank_list+[37])
    hmodel_diag_inst1.show_hist((2,3), low_rank_list)
    hmodel_diag_inst1.show_hist((1,1), [37])
    
    hmodel_diag_inst1.print_summaries(4, latex_table_format=True)
    
    ## for p
    hmodel_p_sampler_inst1 = tk1.CondSampler_HModel_TH1(hmodel_diag_inst1.MC_sample)
    hmodel_p_sampler_inst1.generate_samples()
    hmodel_diag_inst_for_p_1 = MCMC_Diag()
    hmodel_diag_inst_for_p_1.set_mc_sample_from_MCMC_instance(hmodel_p_sampler_inst1)
    hmodel_diag_inst_for_p_1.set_variable_names(["p"+str(i) for i in range(1,38)])
    hmodel_diag_inst_for_p_1.show_hist((2,3), high_rank_list)
    hmodel_diag_inst_for_p_1.show_hist((2,3), low_rank_list)
    hmodel_diag_inst_for_p_1.print_summaries(4, latex_table_format=True)
    
    hmodel_report_rank_inst = ReportRank(hmodel_p_sampler_inst1.MC_sample)
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(1) #10
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(2) #9
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(3) #33
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(4) #6
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(5) #13
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(37) #31
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(36) #30
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(35) #21
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(34) #29
    hmodel_report_rank_inst.rank_barchart_for_kth_ranking(33) #28

    hmodel_report_rank_inst.rank_barchart_for_jth_region(10)
    hmodel_report_rank_inst.rank_barchart_for_jth_region(9)