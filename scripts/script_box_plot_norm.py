import numpy as np
import matplotlib.pyplot as plt

#### class acceptable attacked
##adv attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_4000_pts_20_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_4000_pts_50_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_4000_pts_100_iter.csv"
##random attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_random_4000_pts_20_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_random_4000_pts_50_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_random_4000_pts_100_iter.csv"

#### class non-acceptable attacked
##adv attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_4000_pts_20_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_4000_pts_50_iter_nonacc.csv"
f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_4000_pts_100_iter_nonacc.csv"
##random attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_random_4000_pts_20_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_random_4000_pts_50_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_exec_norm/results_exec_random_4000_pts_100_iter_nonacc.csv"


data=np.genfromtxt(f,delimiter=",")
data_t = np.transpose(data)

fig, ax = plt.subplots()

ax.boxplot(data_t, labels = np.logspace(-6,6,num=7,endpoint=True))

ax.set_ylim(-200,4200)

#ax.set_xlabel("displacement step size")
#ax.set_ylabel("Number of successful attacks (over 4000)")

plt.show()



