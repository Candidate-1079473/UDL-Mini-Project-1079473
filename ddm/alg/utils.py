import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cla_models_multihead import MFVI_NN
import tensorflow as tf
    


def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y


def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size=None):
    mf_weights, mf_variances = model.get_weights()
    acc = []
    log_lik_scores = []

    if single_head:
        if len(x_coresets) > 0:
            x_train, y_train = merge_coresets(x_coresets, y_coresets)
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            final_model = MFVI_NN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
            final_model.train(x_train, y_train, 0, no_epochs, bsize)
        else:
            final_model = model

    for i in range(len(x_testsets)):
        print('we are on test set number:', i)
        if not single_head:
            if len(x_coresets) > 0:
                x_train, y_train = x_coresets[i], y_coresets[i]
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = MFVI_NN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
                final_model.train(x_train, y_train, i, no_epochs, bsize)
            else:
                final_model = model

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]


        pred = final_model.prediction_prob(x_test, head)
        
        #NOW COMPUTE THE ACC
        pred_mean = np.mean(pred, axis=0)
        pred_y = np.argmax(pred_mean, axis=1)
        y = np.argmax(y_test, axis=1)
        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)
        
        #NOW COMPUTE THE LOG_LIK
        curr_log_liks = []
        for y_true, y_pred in zip(y_test, pred_mean):
            # Compute log likelihood using predicted probabilities and true labels
            index = np.argmax(y_true)
            test_example_log_lik = np.log(y_pred[index])
            curr_log_liks.append(test_example_log_lik)
        
        curr_log_lik_score = np.mean(curr_log_liks)
        log_lik_scores.append(curr_log_lik_score)
        
        if len(x_coresets) > 0 and not single_head:
            final_model.close_session()

    if len(x_coresets) > 0 and single_head:
        final_model.close_session()

    return [acc, log_lik_scores]

def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score

def plot(filename, vcl, rand_vcl, kcen_vcl, acc_or_lik):
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    if acc_or_lik == 'acc':
        ax.set_ylabel('Average accuracy')
    if acc_or_lik == 'lik':
        ax.set_ylabel('Average Test Log-Likelihood')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()
