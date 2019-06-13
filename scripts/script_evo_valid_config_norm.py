import numpy as np
import matplotlib.pyplot as plt

#### class acceptable attacked
##adv attacks
f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_adv_attack_4000_20_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_adv_attack_4000_50_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_adv_attack_4000_100_iter.csv"
##random attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_random_attack_4000_20_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_random_attack_4000_50_iter.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_random_attack_4000_100_iter.csv"
#### class non-acceptable attacked
##adv attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_adv_attack_4000_20_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_adv_attack_4000_50_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_adv_attack_4000_100_iter_nonacc.csv"
##random attacks
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_random_attack_4000_20_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_random_attack_4000_50_iter_nonacc.csv"
#f="../results/4000_gen_adv_config/result_per_nb_step/result_valid_norm/result_valid_config_random_attack_4000_100_iter_nonacc.csv"

data=np.genfromtxt(f,delimiter=",")
data_useful=data[1:]

#fig, ax = plt.subplots()

tixs= np.logspace(-6,6,num=7,endpoint=True)
xi=[i for i in range(0,len(tixs))]

plt.scatter(xi,data_useful[:,0])

plt.xticks(xi,tixs)
plt.ylim(-200,4200)

#plt.xlabel("displacement step size")
#plt.ylabel("Number of valid adversarial configurations (over 4000)")

plt.show()
