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

class SplitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()
    
        #######this is for creating a smaller data set for quicker computation ##############
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
#         valid_num_images = len(valid_set[0])
#         perm_indices = random.sample(range(0, valid_num_images), 100)
#         np.random.shuffle(perm_indices)
#         print(perm_indices)
#         valid_set = list(valid_set)
#         valid_set[0] = np.array([valid_set[0][i] for i in perm_indices])
#         valid_set[1] = np.array([valid_set[1][i] for i in perm_indices])
#         valid_set = (valid_set[0], valid_set[1])
#         
#         #might be best to use entire test set?
#         test_num_images = len(test_set[0])
#         perm_indices = random.sample(range(0, test_num_images), 100)
#         np.random.shuffle(perm_indices)
#         print(perm_indices)
#         test_set = list(test_set)
#         test_set[0] = np.array([test_set[0][i] for i in perm_indices])
#         test_set[1] = np.array([test_set[1][i] for i in perm_indices])
#         test_set = (test_set[0], test_set[1])
        ##############################################################
        
        
        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [256, 256]
batch_size = None
no_epochs = 120
single_head = False

# Run vanilla VCL
#tf.set_random_seed(12)
#np.random.seed(1)
def fix_seeds(a):
    random.seed(a)
    tf.set_random_seed(a)
    np.random.seed(a)

a = 96


fix_seeds(a)

coreset_size = 0
data_gen = SplitMnistGenerator()

#the output of the run_vcl function will now be a list with two elements (two matrices i think): 
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(vcl_result)

# Define the file path where you want to save the object
results_dir = f'results/{a}/'
file_path = os.path.join(results_dir, f'vcl_result_gaussian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(vcl_result, f)



# Run random coreset VCL
tf.reset_default_graph()
#tf.set_random_seed(12)
#np.random.seed(1)
fix_seeds(a)


coreset_size = 40
data_gen = SplitMnistGenerator()
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(rand_vcl_result)

# Define the file path where you want to save the object
file_path = os.path.join(results_dir, f'rand_vcl_result_gaussian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(rand_vcl_result, f)


# Run k-center coreset VCL
tf.reset_default_graph()
#tf.set_random_seed(12)
#np.random.seed(1)
fix_seeds(a)

data_gen = SplitMnistGenerator()
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)

#save result
# Define the file path where you want to save the object
file_path = os.path.join(results_dir, f'kcen_vcl_result_gaussian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(kcen_vcl_result, f)
    
    


# Plot average accuracy - this calculates the arithetic mean 
vcl_avg_acc = np.nanmean(vcl_result[0], 1)
rand_vcl_avg_acc = np.nanmean(rand_vcl_result[0], 1)
kcen_vcl_avg_acc = np.nanmean(kcen_vcl_result[0], 1)
utils.plot(f'results/gaussian_prior_split_acc_{a}.jpg', vcl_avg_acc, rand_vcl_avg_acc, kcen_vcl_avg_acc, 'acc')

# Plot average log_lik score
vcl_avg_log_lik = np.nanmean(vcl_result[1], 1)
rand_vcl_avg_log_lik = np.nanmean(rand_vcl_result[1], 1)
kcen_vcl_avg_log_lik = np.nanmean(kcen_vcl_result[1], 1)


print('we call utils.plot with the following arguments:', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')
utils.plot(f'results/gaussian_prior_split_log_lik_{a}.jpg', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')



a = 95


fix_seeds(a)

coreset_size = 0
data_gen = SplitMnistGenerator()

#the output of the run_vcl function will now be a list with two elements: 
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(vcl_result)

# Define the file path where you want to save the object
results_dir = f'results/{a}/'
file_path = os.path.join(results_dir, f'vcl_result_gaussian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(vcl_result, f)



# Run random coreset VCL
tf.reset_default_graph()
#tf.set_random_seed(12)
#np.random.seed(1)
fix_seeds(a)


coreset_size = 40
data_gen = SplitMnistGenerator()
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(rand_vcl_result)

# Define the file path where you want to save the object
file_path = os.path.join(results_dir, f'rand_vcl_result_gaussian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(rand_vcl_result, f)


# Run k-center coreset VCL
tf.reset_default_graph()
#tf.set_random_seed(12)
#np.random.seed(1)
fix_seeds(a)

data_gen = SplitMnistGenerator()
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)

#save result
# Define the file path where you want to save the object
file_path = os.path.join(results_dir, f'kcen_vcl_result_gaussian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(kcen_vcl_result, f)
    
    


# Plot average accuracy - this calculates the arithetic mean 
vcl_avg_acc = np.nanmean(vcl_result[0], 1)
rand_vcl_avg_acc = np.nanmean(rand_vcl_result[0], 1)
kcen_vcl_avg_acc = np.nanmean(kcen_vcl_result[0], 1)
utils.plot(f'results/gaussian_prior_split_acc_{a}.jpg', vcl_avg_acc, rand_vcl_avg_acc, kcen_vcl_avg_acc, 'acc')

# Plot average log_lik score
vcl_avg_log_lik = np.nanmean(vcl_result[1], 1)
rand_vcl_avg_log_lik = np.nanmean(rand_vcl_result[1], 1)
kcen_vcl_avg_log_lik = np.nanmean(kcen_vcl_result[1], 1)


print('we call utils.plot with the following arguments:', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')
utils.plot(f'results/gaussian_prior_split_log_lik_{a}.jpg', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')


