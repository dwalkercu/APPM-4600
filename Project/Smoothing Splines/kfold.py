import numpy as np

def k_folds(x_data, y_data, k):
    data_size = len(y_data)
    fold_size = data_size // k
    folds = np.zeros((k,2,fold_size), np.float64)

    new_ind = 0
    for i in range(k):
        folds[i,0] = x_data[new_ind:new_ind+fold_size]
        folds[i,1] = y_data[new_ind:new_ind+fold_size]
        new_ind = new_ind + fold_size

    return folds

def select_new_folds(folds, last_val_ind):
    last_val_ind += 1
    k = len(folds)

    if last_val_ind >= k:
        last_val_ind = 0

    val_fold = folds[last_val_ind]
    train_folds = np.concatenate((folds[:last_val_ind], folds[last_val_ind+1:]))

    return (val_fold,train_folds,last_val_ind)