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
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def rgb2grey(data):
    # Reshape to (50000, 32, 32, 3) to make channel dimension the last dimension
    data_reshaped = np.transpose(data, (0, 2, 3, 1))
    # Perform dot product along the last dimension
    return np.dot(data_reshaped, [0.2989, 0.5870, 0.1140])


class SplitCifarGenerator():
    def __init__(self):
        file = '/mnt/sdd/MSc_projects/knowles/projects/1st-April-variational-continual-learning-master/ddm/data/cifar-100-python/train'
        cifar100_train = unpickle(file)
        train_set = cifar100_train[b'data']
        #train_set.shape is (50000 , 3072) # 3*32*32 = 3072
        train_set = train_set.reshape(50000,3,32,32)
        train_set = rgb2grey(train_set) #shape is now (50000,32,32)
        train_set = train_set.reshape(50000, 32*32)
        #######now get the labels for the train_set ########
        train_set_labels = cifar100_train[b'fine_labels']

        #fig.savefig(filename, bbox_inches='tight')
        #plt.close()
        
        #now load the test set
        file = '/mnt/sdd/MSc_projects/knowles/projects/1st-April-variational-continual-learning-master/ddm/data/cifar-100-python/test'
        cifar100_test = unpickle(file)
        test_set = cifar100_test[b'data']
        test_set = test_set.reshape(10000,3,32,32)
        test_set = rgb2grey(test_set) #shape is now (50000,32,32)
        test_set = test_set.reshape(10000, 32*32)
        #######now get the labels for the test_set ########
        test_set_labels = cifar100_test[b'fine_labels']

        
        #######this is for creating a smaller data set for quicker computation #####
#         train_num_images = len(train_set)
#         perm_indices = random.sample(range(0, train_num_images), 5000)
#         np.random.shuffle(perm_indices)
#         train_set = np.array([train_set[i] for i in perm_indices])
#         train_set_labels = np.array([train_set_labels[i] for i in perm_indices])
#         
#         test_num_images = len(test_set)
#         print('entire test_set before sampling has length', len(test_set))
#         perm_indices = random.sample(range(0, test_num_images), 1000)
#         np.random.shuffle(perm_indices)
#         test_set = np.array([test_set[i] for i in perm_indices])
#         test_set_labels = np.array([test_set_labels[i] for i in perm_indices])

        
        self.X_train = np.vstack(train_set) #stacking probably not needed
        self.X_test = np.array(test_set)
        self.train_label = np.hstack(train_set_labels) #probably not needed
        self.test_label = np.array(test_set_labels)

        self.sets_0 = [num for num in range(0, 99, 2)]
        self.sets_1 = [num for num in range(1, 100, 2)]
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

a = 99


fix_seeds(a)

coreset_size = 0
data_gen = SplitCifarGenerator()


#the output of the run_vcl function will now be a list with two elements
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)

print(vcl_result)

# Define the file path where you want to save the object
results_dir = 'results/CIFAR100/'
file_path = os.path.join(results_dir, f'CIFAR100_vcl_result_laplacian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(vcl_result, f)


# Run random coreset VCL
tf.reset_default_graph()
#tf.set_random_seed(12)
#np.random.seed(1)
fix_seeds(a)


coreset_size = 40
data_gen = SplitCifarGenerator()
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)

print(rand_vcl_result)

# Define the file path where you want to save the object
file_path = os.path.join(results_dir, f'CIFAR100_rand_vcl_result_laplacian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(rand_vcl_result, f)


# Run k-center coreset VCL
tf.reset_default_graph()
#tf.set_random_seed(12)
#np.random.seed(1)
fix_seeds(a)

data_gen = SplitCifarGenerator()
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)

#save result
# Define the file path where you want to save the object
file_path = os.path.join(results_dir, f'CIFAR100_kcen_vcl_result_laplacian_{a}.pkl')

# Save the object to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(kcen_vcl_result, f)
    



# Plot average accuracy - this calculates the arithetic mean 
vcl_avg_acc = np.nanmean(vcl_result[0], 1)
rand_vcl_avg_acc = np.nanmean(rand_vcl_result[0], 1)
kcen_vcl_avg_acc = np.nanmean(kcen_vcl_result[0], 1)
utils.plot(f'results/CIFAR100/CIFAR100_laplacian_prior_split_acc_{a}.jpg', vcl_avg_acc, rand_vcl_avg_acc, kcen_vcl_avg_acc, 'acc')

# Plot average log_lik score
vcl_avg_log_lik = np.nanmean(vcl_result[1], 1)
rand_vcl_avg_log_lik = np.nanmean(rand_vcl_result[1], 1)
kcen_vcl_avg_log_lik = np.nanmean(kcen_vcl_result[1], 1)


print('we call utils.plot with the following arguments:', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')
utils.plot(f'results/CIFAR100/CIFAR100_laplacian_prior_split_log_lik_{a}.jpg', vcl_avg_log_lik, rand_vcl_avg_log_lik, kcen_vcl_avg_log_lik, 'lik')





