import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import Vanilla_NN, MFVI_NN

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])
    all_log_lik = np.array([])

    for task_id in range(data_gen.max_iter):
        print('the current task ID is', task_id)
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            ml_model.close_session()

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Train on non-coreset data
        mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)

        
        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        
        print('now model_train has ended: I just need to change the KL divergence for the bit inbetween!')
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        #note get_scores returns the prediction accu
        scores  = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
        acc = scores[0]
        log_lik_scores = scores[1]
        all_acc = utils.concatenate_results(acc, all_acc)
        all_log_lik = utils.concatenate_results(log_lik_scores, all_log_lik)


        mf_model.close_session()

    return [all_acc, all_log_lik]
