#locally plotting FINAL RESULTS
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# print(os.getcwd())
# def plot_CIFAR_VCL_GAUS_LAPLACE(filename, vcl, vcl_laplacian, acc_or_lik):
#     #plt.rc('text', usetex=True)
#     #plt.rc('font', family='serif')

#     fig = plt.figure(figsize=(7,3))
#     ax = plt.gca()
#     plt.plot(np.arange(len(vcl))+1, vcl_laplacian, label='VCL_Laplacian', linewidth=2)#, marker='o')
#     plt.plot(np.arange(len(vcl))+1, vcl, label='VCL_Gaussian', linewidth=2)#, marker='o')
#     ax.set_xticks(range(5, 51, 5))
#     if acc_or_lik == 'acc':
#         ax.set_ylabel('Average accuracy', fontsize = 14)
#     if acc_or_lik == 'lik':
#         ax.set_ylabel('Average Test Log-Likelihood', fontsize = 14)
#     ax.set_xlabel('\# tasks', fontsize = 14)
#     ax.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     fig.savefig(filename, bbox_inches='tight')
#     plt.show()

# #####CIFAR100 RESULTS!
# # Specify the file path
# file_path = 'CIFAR100/CIFAR100_vcl_result_gaussian_99.pkl'

# # Open the pickle file for reading
# with open(file_path, 'rb') as file:
#     # Load the data from the pickle file
#     data = pickle.load(file)

# vcl_avg_acc = np.nanmean(data[0], 1)
# vcl_avg_lik = np.nanmean(data[1], 1)

# file_path2 = 'CIFAR100/CIFAR100_vcl_result_laplacian_99.pkl'

# # Open the pickle file for reading
# with open(file_path2, 'rb') as file:
#     # Load the data from the pickle file
#     data = pickle.load(file)

# vcl_avg_acc_lap = np.nanmean(data[0], 1)
# vcl_avg_lik_lap = np.nanmean(data[1], 1)

# plot_CIFAR_VCL_GAUS_LAPLACE('cifar_acc_final_plot.jpg', vcl_avg_acc, vcl_avg_acc_lap, 'acc')
# plt.close()
# plot_CIFAR_VCL_GAUS_LAPLACE('cifar_lik_final_plot.jpg', vcl_avg_lik, vcl_avg_lik_lap, 'lik')
# plt.close()

# ####NOW AVERAGING OVER THE 5 SEEDS FOR GAUSSIAN SPLIT MNIST ##########


# ############GAUS SPLIT ##############################################


VCL_GAUS_SPLIT_ACC = []
VCL_GAUS_SPLIT_LIK = []
RAND_VCL_GAUS_SPLIT_ACC = []
RAND_VCL_GAUS_SPLIT_LIK = []
KCEN_VCL_GAUS_SPLIT_ACC = []
KCEN_VCL_GAUS_SPLIT_LIK = []

for a in [95,96,97,98,99]:
    
    # Specify the file path
    base_folder = f'GAUS_SPLIT_95_TO_99/{a}/'
  
    vcl_file_path = base_folder + f'vcl_result_gaussian_{a}.pkl'
    # Open the pickle files for reading
    with open(vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    vcl_avg_acc_gaus = np.nanmean(data[0], 1)
    vcl_avg_lik_gaus = np.nanmean(data[1], 1)
 
    
    VCL_GAUS_SPLIT_ACC = VCL_GAUS_SPLIT_ACC + [vcl_avg_acc_gaus]
    VCL_GAUS_SPLIT_LIK = VCL_GAUS_SPLIT_LIK + [vcl_avg_lik_gaus]
    
    rand_vcl_file_path = base_folder + f'rand_vcl_result_gaussian_{a}.pkl'
    # Open the pickle files for reading
    with open(rand_vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    rand_vcl_avg_acc_gaus = np.nanmean(data[0], 1)
    rand_vcl_avg_lik_gaus = np.nanmean(data[1], 1)

    RAND_VCL_GAUS_SPLIT_ACC = RAND_VCL_GAUS_SPLIT_ACC + [rand_vcl_avg_acc_gaus]
    RAND_VCL_GAUS_SPLIT_LIK = RAND_VCL_GAUS_SPLIT_LIK + [rand_vcl_avg_lik_gaus]

    
    kcen_vcl_file_path = base_folder + f'kcen_vcl_result_gaussian_{a}.pkl'
    # Open the pickle files for reading
    with open(kcen_vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    kcen_vcl_avg_acc_gaus = np.nanmean(data[0], 1)
    kcen_vcl_avg_lik_gaus = np.nanmean(data[1], 1)
    
    KCEN_VCL_GAUS_SPLIT_ACC = KCEN_VCL_GAUS_SPLIT_ACC + [kcen_vcl_avg_acc_gaus]
    KCEN_VCL_GAUS_SPLIT_LIK = KCEN_VCL_GAUS_SPLIT_LIK + [kcen_vcl_avg_lik_gaus]


meaned_acc_gaus_split_vcl = np.mean(np.array(VCL_GAUS_SPLIT_ACC), axis = 0) 
sd_acc_gaus_split_vcl = np.std(np.array(VCL_GAUS_SPLIT_ACC), axis = 0)

meaned_acc_gaus_split_rand_vcl = np.mean(np.array(RAND_VCL_GAUS_SPLIT_ACC), axis = 0) 
sd_acc_gaus_split_rand_vcl = np.std(np.array(RAND_VCL_GAUS_SPLIT_ACC), axis = 0)

meaned_acc_gaus_split_kcen_vcl = np.mean(np.array(KCEN_VCL_GAUS_SPLIT_ACC), axis = 0)
sd_acc_gaus_split_kcen_vcl = np.std(np.array(KCEN_VCL_GAUS_SPLIT_ACC), axis = 0)

#now for the likelihoods
meaned_lik_gaus_split_vcl = np.mean(np.array(VCL_GAUS_SPLIT_LIK), axis = 0) 
sd_lik_gaus_split_vcl = np.std(np.array(VCL_GAUS_SPLIT_LIK), axis = 0)

meaned_lik_gaus_split_rand_vcl = np.mean(np.array(RAND_VCL_GAUS_SPLIT_LIK), axis = 0)
sd_lik_gaus_split_rand_vcl = np.std(np.array(RAND_VCL_GAUS_SPLIT_LIK), axis = 0)

meaned_lik_gaus_split_kcen_vcl = np.mean(np.array(KCEN_VCL_GAUS_SPLIT_LIK), axis = 0)
sd_lik_gaus_split_kcen_vcl = np.std(np.array(KCEN_VCL_GAUS_SPLIT_LIK), axis = 0)

      ##################LAPLACE SPLIT ######################################
VCL_LAPLACE_SPLIT_ACC = []
VCL_LAPLACE_SPLIT_LIK = []
RAND_VCL_LAPLACE_SPLIT_ACC = []
RAND_VCL_LAPLACE_SPLIT_LIK = []
KCEN_VCL_LAPLACE_SPLIT_ACC = []
KCEN_VCL_LAPLACE_SPLIT_LIK = []

for a in [95,96,97,98,99]:
    
    # Specify the file path
    base_folder = f'LAPLACE_SPLIT_95_TO_99/{a}/'

    vcl_file_path = base_folder + f'vcl_result_laplacian_{a}.pkl'
    # Open the pickle files for reading
    with open(vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    vcl_avg_acc_laplace = np.nanmean(data[0], 1)
    vcl_avg_lik_laplace = np.nanmean(data[1], 1)
    
    VCL_LAPLACE_SPLIT_ACC = VCL_LAPLACE_SPLIT_ACC + [vcl_avg_acc_laplace]
    VCL_LAPLACE_SPLIT_LIK = VCL_LAPLACE_SPLIT_LIK + [vcl_avg_lik_laplace]
    
    rand_vcl_file_path = base_folder + f'rand_vcl_result_laplacian_{a}.pkl'
    # Open the pickle files for reading
    with open(rand_vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    rand_vcl_avg_acc_laplace = np.nanmean(data[0], 1)
    rand_vcl_avg_lik_laplace = np.nanmean(data[1], 1)
 
    ###THESE 
    
    RAND_VCL_LAPLACE_SPLIT_ACC = RAND_VCL_LAPLACE_SPLIT_ACC + [rand_vcl_avg_acc_laplace]
    RAND_VCL_LAPLACE_SPLIT_LIK = RAND_VCL_LAPLACE_SPLIT_LIK + [rand_vcl_avg_lik_laplace]
    #print(RAND_VCL_LAPLACE_SPLIT_ACC)
    
    kcen_vcl_file_path = base_folder + f'kcen_vcl_result_laplacian_{a}.pkl'
    # Open the pickle files for reading
    with open(kcen_vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    kcen_vcl_avg_acc_laplace = np.nanmean(data[0], 1)
    kcen_vcl_avg_lik_laplace = np.nanmean(data[1], 1)
    
    KCEN_VCL_LAPLACE_SPLIT_ACC = KCEN_VCL_LAPLACE_SPLIT_ACC + [kcen_vcl_avg_acc_laplace]
    KCEN_VCL_LAPLACE_SPLIT_LIK = KCEN_VCL_LAPLACE_SPLIT_LIK + [kcen_vcl_avg_lik_laplace]


meaned_acc_laplace_split_vcl = np.mean(np.array(VCL_LAPLACE_SPLIT_ACC), axis = 0) 
sd_acc_laplace_split_vcl = np.std(np.array(VCL_LAPLACE_SPLIT_ACC), axis = 0)

meaned_acc_laplace_split_rand_vcl = np.mean(np.array(RAND_VCL_LAPLACE_SPLIT_ACC), axis = 0) 
sd_acc_laplace_split_rand_vcl = np.std(np.array(RAND_VCL_LAPLACE_SPLIT_ACC), axis = 0)

meaned_acc_laplace_split_kcen_vcl = np.mean(np.array(KCEN_VCL_LAPLACE_SPLIT_ACC), axis = 0)
sd_acc_laplace_split_kcen_vcl = np.std(np.array(KCEN_VCL_LAPLACE_SPLIT_ACC), axis = 0)

#now for the likelihoods
meaned_lik_laplace_split_vcl = np.mean(np.array(VCL_LAPLACE_SPLIT_LIK), axis = 0) 
sd_lik_laplace_split_vcl = np.std(np.array(VCL_LAPLACE_SPLIT_LIK), axis = 0)

meaned_lik_laplace_split_rand_vcl = np.mean(np.array(RAND_VCL_LAPLACE_SPLIT_LIK), axis = 0)
sd_lik_laplace_split_rand_vcl = np.std(np.array(RAND_VCL_LAPLACE_SPLIT_LIK), axis = 0)

meaned_lik_laplace_split_kcen_vcl = np.mean(np.array(KCEN_VCL_LAPLACE_SPLIT_LIK), axis = 0)
sd_lik_laplace_split_kcen_vcl = np.std(np.array(KCEN_VCL_LAPLACE_SPLIT_LIK), axis = 0)


##############################NOWW FOR PERMUTED####################

############GAUS PERMUTED ##############################################


VCL_GAUS_PERMUTED_ACC = []
VCL_GAUS_PERMUTED_LIK = []
RAND_VCL_GAUS_PERMUTED_ACC = []
RAND_VCL_GAUS_PERMUTED_LIK = []
KCEN_VCL_GAUS_PERMUTED_ACC = []
KCEN_VCL_GAUS_PERMUTED_LIK = []

for a in [95,96,97, 98, 99]:
    
    # Specify the file path
    base_folder = f'GAUS_PERMUTED_95_TO_99/{a}/'
    #print('somethign')
    vcl_file_path = base_folder + f'vcl_result_gaussian_{a}.pkl'
    # Open the pickle files for reading
    with open(vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        
    #if a == 95:
    #    print('FFFF', data)
    vcl_avg_acc_gaus = np.nanmean(data[0], 1)
    vcl_avg_lik_gaus = np.nanmean(data[1], 1)
    #print('worked')
    
    VCL_GAUS_PERMUTED_ACC = VCL_GAUS_PERMUTED_ACC + [vcl_avg_acc_gaus]
    VCL_GAUS_PERMUTED_LIK = VCL_GAUS_PERMUTED_LIK + [vcl_avg_lik_gaus]
    
    rand_vcl_file_path = base_folder + f'rand_vcl_result_gaussian_{a}.pkl'
    # Open the pickle files for reading
    with open(rand_vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    rand_vcl_avg_acc_gaus = np.nanmean(data[0], 1)
    rand_vcl_avg_lik_gaus = np.nanmean(data[1], 1)
    #print(rand_vcl_avg_lik_gaus)
    ###THESE 
    
    RAND_VCL_GAUS_PERMUTED_ACC = RAND_VCL_GAUS_PERMUTED_ACC + [rand_vcl_avg_acc_gaus]
    RAND_VCL_GAUS_PERMUTED_LIK = RAND_VCL_GAUS_PERMUTED_LIK + [rand_vcl_avg_lik_gaus]
    #print(RAND_VCL_GAUS_PERMUTED_ACC)
    
    kcen_vcl_file_path = base_folder + f'kcen_vcl_result_gaussian_{a}.pkl'
    # Open the pickle files for reading
    with open(kcen_vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    kcen_vcl_avg_acc_gaus = np.nanmean(data[0], 1)
    kcen_vcl_avg_lik_gaus = np.nanmean(data[1], 1)
    
    KCEN_VCL_GAUS_PERMUTED_ACC = KCEN_VCL_GAUS_PERMUTED_ACC + [kcen_vcl_avg_acc_gaus]
    KCEN_VCL_GAUS_PERMUTED_LIK = KCEN_VCL_GAUS_PERMUTED_LIK + [kcen_vcl_avg_lik_gaus]


meaned_acc_gaus_permuted_vcl = np.mean(np.array(VCL_GAUS_PERMUTED_ACC), axis = 0) 
sd_acc_gaus_permuted_vcl = np.std(np.array(VCL_GAUS_PERMUTED_ACC), axis = 0)

meaned_acc_gaus_permuted_rand_vcl = np.mean(np.array(RAND_VCL_GAUS_PERMUTED_ACC), axis = 0) 
sd_acc_gaus_permuted_rand_vcl = np.std(np.array(RAND_VCL_GAUS_PERMUTED_ACC), axis = 0)

meaned_acc_gaus_permuted_kcen_vcl = np.mean(np.array(KCEN_VCL_GAUS_PERMUTED_ACC), axis = 0)
sd_acc_gaus_permuted_kcen_vcl = np.std(np.array(KCEN_VCL_GAUS_PERMUTED_ACC), axis = 0)

#now for the likelihoods
meaned_lik_gaus_permuted_vcl = np.mean(np.array(VCL_GAUS_PERMUTED_LIK), axis = 0) 
sd_lik_gaus_permuted_vcl = np.std(np.array(VCL_GAUS_PERMUTED_LIK), axis = 0)

meaned_lik_gaus_permuted_rand_vcl = np.mean(np.array(RAND_VCL_GAUS_PERMUTED_LIK), axis = 0)
sd_lik_gaus_permuted_rand_vcl = np.std(np.array(RAND_VCL_GAUS_PERMUTED_LIK), axis = 0)

meaned_lik_gaus_permuted_kcen_vcl = np.mean(np.array(KCEN_VCL_GAUS_PERMUTED_LIK), axis = 0)
sd_lik_gaus_permuted_kcen_vcl = np.std(np.array(KCEN_VCL_GAUS_PERMUTED_LIK), axis = 0)

      ##################LAPLACE PERMUTED ######################################
VCL_LAPLACE_PERMUTED_ACC = []
VCL_LAPLACE_PERMUTED_LIK = []
RAND_VCL_LAPLACE_PERMUTED_ACC = []
RAND_VCL_LAPLACE_PERMUTED_LIK = []
KCEN_VCL_LAPLACE_PERMUTED_ACC = []
KCEN_VCL_LAPLACE_PERMUTED_LIK = []

for a in [95,96,97, 98, 99]:
    
    # Specify the file path
    base_folder = f'LAPLACE_PERMUTED_95_TO_99/{a}/'
    #print('somethign')
    vcl_file_path = base_folder + f'vcl_result_laplace_{a}.pkl'
    # Open the pickle files for reading
    with open(vcl_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        gone_wrong_1 =  np.nanmean(data[0], 1)
    vcl_avg_acc_laplace = np.nanmean(data[0], 1)
    vcl_avg_lik_laplace = np.nanmean(data[1], 1)
    #print('worked')
    
    VCL_LAPLACE_PERMUTED_ACC = VCL_LAPLACE_PERMUTED_ACC + [vcl_avg_acc_laplace]
    VCL_LAPLACE_PERMUTED_LIK = VCL_LAPLACE_PERMUTED_LIK + [vcl_avg_lik_laplace]
    
    # rand_vcl_file_path = base_folder + f'rand_vcl_result_laplace_{a}.pkl'
    # # Open the pickle files for reading
    # with open(rand_vcl_file_path, 'rb') as file:
    #     # Load the data from the pickle file
    #     data = pickle.load(file)
    #     gone_wrong_2 = np.nanmean(data[0], 1)
    # rand_vcl_avg_acc_laplace = np.nanmean(data[0], 1)
    # rand_vcl_avg_lik_laplace = np.nanmean(data[1], 1)
    # #print(rand_vcl_avg_lik_laplace)
    # ###THESE 
    
    # RAND_VCL_LAPLACE_PERMUTED_ACC = RAND_VCL_LAPLACE_PERMUTED_ACC + [rand_vcl_avg_acc_laplace]
    # RAND_VCL_LAPLACE_PERMUTED_LIK = RAND_VCL_LAPLACE_PERMUTED_LIK + [rand_vcl_avg_lik_laplace]
    # #print(RAND_VCL_LAPLACE_PERMUTED_ACC)
    
    # kcen_vcl_file_path = base_folder + f'kcen_vcl_result_laplace_{a}.pkl'
    # # Open the pickle files for reading
    # with open(kcen_vcl_file_path, 'rb') as file:
    #     # Load the data from the pickle file
    #     data = pickle.load(file)
    #     gone_wrong_3 =  np.nanmean(data[0], 1)
    # kcen_vcl_avg_acc_laplace = np.nanmean(data[0], 1)
    # kcen_vcl_avg_lik_laplace = np.nanmean(data[1], 1)
    # print('YUP', gone_wrong_2 == gone_wrong_3)
    
    # KCEN_VCL_LAPLACE_PERMUTED_ACC = KCEN_VCL_LAPLACE_PERMUTED_ACC + [kcen_vcl_avg_acc_laplace]
    # KCEN_VCL_LAPLACE_PERMUTED_LIK = KCEN_VCL_LAPLACE_PERMUTED_LIK + [kcen_vcl_avg_lik_laplace]


#print('1', VCL_LAPLACE_PERMUTED_ACC)
#print('2', VCL_GAUS_PERMUTED_ACC)

#print('3', KCEN_VCL_LAPLACE_SPLIT_ACC)
#print('4', KCEN_VCL_GAUS_SPLIT_ACC)

meaned_acc_laplace_permuted_vcl = np.mean(np.array(VCL_LAPLACE_PERMUTED_ACC), axis = 0) 
sd_acc_laplace_permuted_vcl = np.std(np.array(VCL_LAPLACE_PERMUTED_ACC), axis = 0)

# meaned_acc_laplace_permuted_rand_vcl = np.mean(np.array(RAND_VCL_LAPLACE_PERMUTED_ACC), axis = 0) 
# sd_acc_laplace_permuted_rand_vcl = np.std(np.array(RAND_VCL_LAPLACE_PERMUTED_LIK), axis = 0)

# meaned_acc_laplace_permuted_kcen_vcl = np.mean(np.array(KCEN_VCL_LAPLACE_PERMUTED_ACC), axis = 0)
# sd_acc_laplace_permuted_kcen_vcl = np.std(np.array(KCEN_VCL_LAPLACE_PERMUTED_LIK), axis = 0)

#now for the likelihoods
meaned_lik_laplace_permuted_vcl = np.mean(np.array(VCL_LAPLACE_PERMUTED_LIK), axis = 0) 
sd_lik_laplace_permuted_vcl = np.std(np.array(VCL_LAPLACE_PERMUTED_LIK), axis = 0)

# meaned_lik_laplace_permuted_rand_vcl = np.mean(np.array(RAND_VCL_LAPLACE_PERMUTED_LIK), axis = 0)
# sd_lik_laplace_permuted_rand_vcl = np.std(np.array(RAND_VCL_LAPLACE_PERMUTED_LIK), axis = 0)

# meaned_lik_laplace_permuted_kcen_vcl = np.mean(np.array(KCEN_VCL_LAPLACE_PERMUTED_LIK), axis = 0)
# sd_lik_laplace_permuted_kcen_vcl = np.std(np.array(KCEN_VCL_LAPLACE_PERMUTED_LIK), axis = 0)


##############################





#############################################################################################

def plot_error_bars(filename, vcl, vcl_sd, rand_vcl, rand_vcl_sd, kcen_vcl, kcen_vcl_sd, acc_or_lik):
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    #print(vcl_sd)
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    
    #plt.errorbar(np.arange(len(vcl)) + 1, vcl, yerr=vcl_sd, fmt='o', capsize=5, color = 'cornflowerblue')
    #plt.errorbar(np.arange(len(rand_vcl)) + 1, rand_vcl, yerr=rand_vcl_sd, fmt='o', capsize=5)
    #plt.errorbar(np.arange(len(kcen_vcl)) + 1, kcen_vcl, yerr=kcen_vcl_sd, fmt='o', capsize=5)
    
    ax.set_xticks(range(1, len(vcl)+1))
    if acc_or_lik == 'acc':
        ax.set_ylabel('Average accuracy')
    if acc_or_lik == 'lik':
        ax.set_ylabel('Average Test Log-Likelihood')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_shaded_ONE_dist_3_vcl(filename, vcl, vcl_sd, rand_vcl, rand_vcl_sd, kcen_vcl, kcen_vcl_sd, acc_or_lik):
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    #plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    
    # Create shaded area around the line plot
    plt.fill_between(np.arange(len(vcl))+1, (vcl - vcl_sd), (vcl + vcl_sd), color='lightblue', alpha=0.4)
    # Create shaded area around the line plot
    plt.fill_between(np.arange(len(vcl))+1, rand_vcl - rand_vcl_sd, rand_vcl + rand_vcl_sd, color='orange', alpha=0.4)
    # Create shaded area around the line plot
    #plt.fill_between(np.arange(len(vcl))+1, kcen_vcl - kcen_vcl_sd, kcen_vcl + kcen_vcl_sd, color='lightgreen', alpha=0.4)

    
    ax.set_xticks(range(1, len(vcl)+1))
    if acc_or_lik == 'acc':
        ax.set_ylabel('Average accuracy')
    if acc_or_lik == 'lik':
        ax.set_ylabel('Average Test Log-Likelihood')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_shaded_two_dist_1_vcl(filename, vcl_gaus, vcl_sd_gaus, vcl_laplace, vcl_sd_laplace, acc_or_lik):
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl_gaus))+1, vcl_gaus, label='VCL Gaussian', marker='o')
    plt.plot(np.arange(len(vcl_laplace))+1, vcl_laplace, label='VCL Laplacian', marker='o')
    #plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    
    # Create shaded area around the line plot
    plt.fill_between(np.arange(len(vcl_gaus))+1, (vcl_gaus - vcl_sd_gaus), (vcl_gaus + vcl_sd_gaus), color='lightblue', alpha=0.4)
    # Create shaded area around the line plot
    plt.fill_between(np.arange(len(vcl_laplace))+1, vcl_laplace - vcl_sd_laplace, vcl_laplace + vcl_sd_laplace, color='orange', alpha=0.4)
    # Create shaded area around the line plot
    #plt.fill_between(np.arange(len(vcl))+1, kcen_vcl - kcen_vcl_sd, kcen_vcl + kcen_vcl_sd, color='lightgreen', alpha=0.4)

    
    ax.set_xticks(range(1, len(vcl_gaus)+1))
    if acc_or_lik == 'acc':
        ax.set_ylabel('Average accuracy', fontsize = 14)
    if acc_or_lik == 'lik':
        ax.set_ylabel('Average Test Log-Likelihood', fontsize = 14)
    ax.set_xlabel('\# tasks', fontsize = 14)
    #plt.title(filename)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

#plot_error_bars('save_dis_boi.jpg', meaned_acc_gaus_split_vcl,sd_acc_gaus_split_vcl, meaned_acc_gaus_split_rand_vcl,sd_acc_gaus_split_rand_vcl,  meaned_acc_gaus_split_kcen_vcl, sd_acc_gaus_split_kcen_vcl, 'acc')

#plot_shaded_one_dist_3_vcl('save_dis_boi.jpg', meaned_acc_gaus_split_vcl,sd_acc_gaus_split_vcl, meaned_acc_gaus_split_rand_vcl,sd_acc_gaus_split_rand_vcl,  meaned_acc_gaus_split_kcen_vcl, sd_acc_gaus_split_kcen_vcl, 'acc')


#PLOT VCL PERMUTED
#plot_shaded_two_dist_1_vcl()

###PLOTTING THE VCL FOR PERMUTED

plot_shaded_two_dist_1_vcl('permuted_VCL_ACC.jpg',  meaned_acc_gaus_permuted_vcl, sd_acc_gaus_permuted_vcl,meaned_acc_laplace_permuted_vcl, sd_acc_laplace_permuted_vcl , 'acc')
plot_shaded_two_dist_1_vcl('permuted_VCL_LIK.jpg',  meaned_lik_gaus_permuted_vcl, sd_lik_gaus_permuted_vcl,meaned_lik_laplace_permuted_vcl, sd_lik_laplace_permuted_vcl , 'lik')

###PLOTTING THE VCL FOR SPLIT

plot_shaded_two_dist_1_vcl('split_VCL_ACC.jpg',  meaned_acc_gaus_split_vcl, sd_acc_gaus_split_vcl,meaned_acc_laplace_split_vcl, sd_acc_laplace_split_vcl , 'acc')
plot_shaded_two_dist_1_vcl('split_VCL_LIK.jpg',  meaned_lik_gaus_split_vcl, sd_lik_gaus_split_vcl,meaned_lik_laplace_split_vcl, sd_lik_laplace_split_vcl , 'lik')




####EXTRA FIGURES FOR THE APPENDIX
###PLOTTING THE RAND_VCL FOR PERMUTED
#plot_shaded_two_dist_1_vcl('permuted_RAND_VCL_ACC.jpg',  meaned_acc_gaus_permuted_rand_vcl, sd_acc_gaus_permuted_rand_vcl,meaned_acc_laplace_permuted_rand_vcl, sd_acc_laplace_permuted_rand_vcl , 'acc')
#plot_shaded_two_dist_1_vcl('permuted_RAND_VCL_LIK.jpg',  meaned_lik_gaus_permuted_rand_vcl, sd_lik_gaus_permuted_rand_vcl,meaned_lik_laplace_permuted_rand_vcl, sd_lik_laplace_permuted_rand_vcl , 'lik')

###PLOTTING THE RAND_VCL FOR SPLIT
plot_shaded_two_dist_1_vcl('split_RAND_VCL_ACC.jpg',  meaned_acc_gaus_split_rand_vcl, sd_acc_gaus_split_rand_vcl,meaned_acc_laplace_split_rand_vcl, sd_acc_laplace_split_rand_vcl , 'acc')
plot_shaded_two_dist_1_vcl('split_RAND_VCL_LIK.jpg',  meaned_lik_gaus_split_rand_vcl, sd_lik_gaus_split_rand_vcl,meaned_lik_laplace_split_rand_vcl, sd_lik_laplace_split_rand_vcl , 'lik')

###PLOTTING THE KCEN_VCL FOR PERMUTED
#plot_shaded_two_dist_1_vcl('permuted_KCEN_VCL_ACC.jpg',  meaned_acc_gaus_permuted_kcen_vcl, sd_acc_gaus_permuted_kcen_vcl,meaned_acc_laplace_permuted_kcen_vcl, sd_acc_laplace_permuted_kcen_vcl , 'acc')
#plot_shaded_two_dist_1_vcl('permuted_KCEN_VCL_LIK.jpg',  meaned_lik_gaus_permuted_kcen_vcl, sd_lik_gaus_permuted_kcen_vcl,meaned_lik_laplace_permuted_kcen_vcl, sd_lik_laplace_permuted_kcen_vcl , 'lik')

###PLOTTING THE KCEN_VCL FOR SPLIT
plot_shaded_two_dist_1_vcl('split_KCEN_VCL_ACC.jpg',  meaned_acc_gaus_split_kcen_vcl, sd_acc_gaus_split_kcen_vcl,meaned_acc_laplace_split_kcen_vcl, sd_acc_laplace_split_kcen_vcl , 'acc')
plot_shaded_two_dist_1_vcl('split_KCEN_VCL_LIK.jpg',  meaned_lik_gaus_split_kcen_vcl, sd_lik_gaus_split_kcen_vcl,meaned_lik_laplace_split_kcen_vcl, sd_lik_laplace_split_kcen_vcl , 'lik')

