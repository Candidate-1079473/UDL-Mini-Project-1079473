import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy
import random
import os

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()
        
        ######Now subset the data in order to make computation quicker #########

#         train_num_images = len(train_set[0])
#         print('train_num_images', train_num_images)
#         perm_indices = random.sample(range(0, train_num_images), 500)
#         np.random.shuffle(perm_indices)
#         print(perm_indices)
#         train_set = list(train_set)
#         train_set[0] = np.array([train_set[0][i] for i in perm_indices])
#         train_set[1] = np.array([train_set[1][i] for i in perm_indices])
#         train_set = (train_set[0], train_set[1])
#         
#         
#         valid_num_images = len(valid_set[0])
#         perm_indices = random.sample(range(0, valid_num_images), 100)
#         np.random.shuffle(perm_indices)
#         print(perm_indices)
#         valid_set = list(valid_set)
#         valid_set[0] = np.array([valid_set[0][i] for i in perm_indices])
#         valid_set[1] = np.array([valid_set[1][i] for i in perm_indices])
#         valid_set = (valid_set[0], valid_set[1])
#         
#         # #might be best to use entire test set?
#         test_num_images = len(test_set[0])
#         perm_indices = random.sample(range(0, test_num_images), 700)
#         np.random.shuffle(perm_indices)
#         print(perm_indices)
#         test_set = list(test_set)
#         test_set[0] = np.array([test_set[0][i] for i in perm_indices])
#         test_set[1] = np.array([test_set[1][i] for i in perm_indices])
#         test_set = (test_set[0], test_set[1])
        
        
        ##################################
        
        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            
            perm_inds = np.arange(self.X_train.shape[1], dtype=int)
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 5

def fix_seeds(a):
    random.seed(a)
    tf.set_random_seed(a)
    np.random.seed(a)

for a in [99]:
    # Run vanilla VCL
    #tf.set_random_seed(12)
    #np.random.seed(1)
#     fix_seeds(a)
#     coreset_size = 0
#     data_gen = PermutedMnistGenerator(num_tasks)
#     vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
#         coreset.rand_from_batch, coreset_size, batch_size, single_head)
#     print(vcl_result)
# 
#     results_dir = f'results/PERMUTED/LAPLACE/{a}/'
#     file_path = os.path.join(results_dir, f'vcl_result_laplace_{a}.pkl')
# 
#     # Save the object to a file using pickle
#     with open(file_path, 'wb') as f:
#         pickle.dump(vcl_result, f)

    # Run random coreset VCL
    tf.reset_default_graph()
    #tf.set_random_seed(12)
    #np.random.seed(1)
    fix_seeds(a)

    coreset_size = 200
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    print(rand_vcl_result)
    
    file_path = os.path.join(results_dir, f'rand_vcl_result_laplace_{a}.pkl')

    # Save the object to a file using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(rand_vcl_result, f)

    # Run k-center coreset VCL
    tf.reset_default_graph()
    #tf.set_random_seed(12)
    #np.random.seed(1)
    fix_seeds(a)

    data_gen = PermutedMnistGenerator(num_tasks)
    kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
        coreset.k_center, coreset_size, batch_size, single_head)
    print(kcen_vcl_result)
    
    file_path = os.path.join(results_dir, f'kcen_vcl_result_laplace_{a}.pkl')

    # Save the object to a file using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(rand_vcl_result, f)

    # Plot average accuracy - this calculates the arithetic mean 
    #vcl_avg_acc = np.nanmean(vcl_result[0], 1)
    #rand_vcl_avg_acc = np.nanmean(rand_vcl_result[0], 1)
    #kcen_vcl_avg_acc = np.nanmean(kcen_vcl_result[0], 1)
    
    
    
    #utils.plot(f'results/PERMUTED/LAPLACE/{a}/permuted_acc_laplace_{a}.jpg', vcl_avg_acc, rand_vcl_avg_acc, kcen_vcl_avg_acc, 'acc')

    # Plot average log_lik score
    #vcl_avg_log_lik = np.nanmean(vcl_result[1], 1)
    #rand_vcl_avg_log_lik = np.nanmean(rand_vcl_result[1], 1)
    #kcen_vcl_avg_log_lik = np.nanmean(kcen_vcl_result[1], 1)
    #utils.plot(f'results/PERMUTED/LAPLACE/{a}/permuted_log_lik_laplace_{a}.jpg', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')


