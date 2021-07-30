from datetime import time

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
from models import BaselineModel, MT_MODEL, MT_doubleTeacher_MODEL
import tensorflow as tf


def train_baseline(dataset_name, X, Y, outer_cv=10, inner_cv=3, random_search_trials=50, inner_epochs=1,
                   outer_epochs=5):
    skf = StratifiedKFold(n_splits=outer_cv, random_state=7, shuffle=True)
    skf2 = StratifiedKFold(n_splits=inner_cv, random_state=7, shuffle=True)

    fold_var = 1

    list_of_res = []
    for train_index, val_index in skf.split(np.zeros(X.shape[0]), Y):
        results_dict = {}
        x_train = X[train_index]
        y_train = Y[train_index]
        x_val = X[val_index]
        y_val = Y[val_index]
        max_trail = 0
        max_acc = 0
        hyper_param_batchsize = np.array([32, 64, 128])
        hyper_param_lr = np.array([0.01, 0.001, 0.0001, 0.005])
        hyper_param_pooling = np.array(['max', 'avg', None])
        hyper_params_dict = {}
        for trail in range(0, random_search_trials):
            b = np.random.choice(hyper_param_batchsize)
            lr = np.random.choice(hyper_param_lr)
            po = np.random.choice(hyper_param_pooling)
            print(f"hyper params {(b, lr, po)}")
            acc_list = []
            for train_ind2, val_inds2 in skf2.split(np.zeros(x_train.shape[0]), y_train):
                x_train_hyperopt = x_train[train_ind2]
                y_train_hyperopt = y_train[train_ind2]
                x_val_hyperopt = x_train[val_inds2]
                y_val_hyperopt = y_train[val_inds2]
                classes = len(np.unique(y_train_hyperopt))
                model = BaselineModel(((x_train_hyperopt, y_train_hyperopt), (x_val_hyperopt, y_val_hyperopt)), classes,
                                      lr, b, po)
                model.train(inner_epochs)
                results = model.evaluate()
                acc_list.append(results['acc'])
                tf.keras.backend.clear_session()

            mean_acc = np.array(acc_list).mean()
            if mean_acc > max_acc:
                max_trail = trail
                max_acc = mean_acc
            hyper_params_dict[trail] = (lr, b, po, mean_acc)
            # for later need to save the results in the dict

        best_params = hyper_params_dict[max_trail]
        model = BaselineModel(((x_train, y_train), (x_val, y_val)), classes, best_params[0], best_params[1],
                              best_params[2])
        start_timer = time.time()
        model.train(outer_epochs)
        end_timer = time.time()
        eval_res = model.evaluate()
        results_dict['dataset_name'] = dataset_name
        results_dict['k-fold'] = fold_var
        results_dict['train_time'] = end_timer - start_timer
        results_dict.update(eval_res)
        list_of_res.append(results_dict)
        tf.keras.backend.clear_session()
        fold_var += 1
        tmp = pd.DataFrame(list_of_res)
        tmp.to_csv(f'Results/Baseline_{dataset_name}.csv')
    return pd.DataFrame(list_of_res)


def train_paper(dataset_name, X, Y, outer_cv=10, inner_cv=3, random_search_trials=50, inner_epochs=1, outer_epochs=5):
    skf = StratifiedKFold(n_splits=outer_cv, random_state=7, shuffle=True)
    skf2 = StratifiedKFold(n_splits=inner_cv, random_state=7, shuffle=True)

    fold_var = 1

    list_of_res = []
    for train_index, val_index in skf.split(np.zeros(X.shape[0]), Y):
        results_dict = {}
        x_train = X[train_index]
        y_train = Y[train_index]
        x_val = X[val_index]
        y_val = Y[val_index]
        max_trail = 0
        max_acc = 0
        hyper_param_dropoutstudent = np.array([0.1, 0.2, 0.3, 0.4])
        hyper_param_batchsize = np.array([32, 64, 128])
        hyper_param_droputteacher = np.array([0.1, 0.2, 0.3, 0.4])
        hyper_param_emarate = np.array([0.999, 0.95, 0.92, 0.98])
        hyper_params_dict = {}
        for trail in range(0, random_search_trials):
            ds = np.random.choice(hyper_param_dropoutstudent)
            b = np.random.choice(hyper_param_batchsize)
            dt = np.random.choice(hyper_param_droputteacher)
            e = np.random.choice(hyper_param_emarate)
            print(f"hyper params {(ds, b, dt, e)}")
            acc_list = []
            for train_ind2, val_inds2 in skf2.split(np.zeros(x_train.shape[0]), y_train):
                x_train_hyperopt = x_train[train_ind2]
                y_train_hyperopt = y_train[train_ind2]
                x_val_hyperopt = x_train[val_inds2]
                y_val_hyperopt = y_train[val_inds2]
                model = MT_MODEL(((x_train_hyperopt, y_train_hyperopt), (x_val_hyperopt, y_val_hyperopt)), b, dt, ds, e)
                # print(np.unique(y_train_hyperopt))
                model.train(inner_epochs)
                results = model.evaluate()
                acc_list.append(results['acc'])
                tf.keras.backend.clear_session()

            mean_acc = np.array(acc_list).mean()
            if mean_acc > max_acc:
                max_trail = trail
                max_acc = mean_acc
            hyper_params_dict[trail] = (b, dt, ds, e, mean_acc)
            # for later need to save the results in the dict

        best_params = hyper_params_dict[max_trail]
        model = MT_MODEL(((x_train, y_train), (x_val, y_val)), best_params[0], best_params[1], best_params[2],
                         best_params[3])
        start_timer = time.time()
        model.train(outer_epochs)
        end_timer = time.time()
        eval_res = model.evaluate()
        results_dict['dataset_name'] = dataset_name
        results_dict['k-fold'] = fold_var
        results_dict['train_time'] = end_timer - start_timer
        results_dict.update(eval_res)
        list_of_res.append(results_dict)
        tf.keras.backend.clear_session()
        fold_var += 1
        tmp = pd.DataFrame(list_of_res)
        tmp.to_csv(f'Results/Paper_{dataset_name}.csv')
    return pd.DataFrame(list_of_res)


def train_improve(dataset_name, X, Y, outer_cv=10, inner_cv=3, random_search_trials=50, inner_epochs=1, outer_epochs=5):
    skf = StratifiedKFold(n_splits=outer_cv, random_state=7, shuffle=True)
    skf2 = StratifiedKFold(n_splits=inner_cv, random_state=7, shuffle=True)

    fold_var = 1

    list_of_res = []
    for train_index, val_index in skf.split(np.zeros(X.shape[0]), Y):
        results_dict = {}
        x_train = X[train_index]
        y_train = Y[train_index]
        x_val = X[val_index]
        y_val = Y[val_index]
        max_trail = 0
        max_acc = 0
        hyper_param_dropoutstudent = np.array([0.1, 0.2, 0.3, 0.4])
        hyper_param_batchsize = np.array([32, 64, 128])
        hyper_param_droputteacher = np.array([0.1, 0.2, 0.3, 0.4])
        hyper_param_emarate = np.array([0.999, 0.95, 0.92, 0.98])
        hyper_params_dict = {}
        for trail in range(0, random_search_trials):
            ds = np.random.choice(hyper_param_dropoutstudent)
            b = np.random.choice(hyper_param_batchsize)
            dt = np.random.choice(hyper_param_droputteacher)
            e = np.random.choice(hyper_param_emarate)
            print(f"hyper params {(ds, b, dt, e)}")
            acc_list = []
            for train_ind2, val_inds2 in skf2.split(np.zeros(x_train.shape[0]), y_train):
                x_train_hyperopt = x_train[train_ind2]
                y_train_hyperopt = y_train[train_ind2]
                x_val_hyperopt = x_train[val_inds2]
                y_val_hyperopt = y_train[val_inds2]
                model = MT_doubleTeacher_MODEL(((x_train_hyperopt, y_train_hyperopt), (x_val_hyperopt, y_val_hyperopt)),
                                               b, dt, ds, dt, e)
                model.train(inner_epochs)
                results = model.evaluate()
                acc_list.append(results['acc'])
                tf.keras.backend.clear_session()

            mean_acc = np.array(acc_list).mean()
            if mean_acc > max_acc:
                max_trail = trail
                max_acc = mean_acc
            hyper_params_dict[trail] = (b, dt, ds, e, mean_acc)
            # for later need to save the results in the dict

        best_params = hyper_params_dict[max_trail]
        model = MT_doubleTeacher_MODEL(((x_train, y_train), (x_val, y_val)), best_params[0], best_params[1],
                                       best_params[2], best_params[1], best_params[3])
        start_timer = time.time()
        model.train(outer_epochs)
        end_timer = time.time()
        eval_res = model.evaluate()
        results_dict['dataset_name'] = dataset_name
        results_dict['k-fold'] = fold_var
        results_dict['train_time'] = end_timer - start_timer
        results_dict.update(eval_res)
        list_of_res.append(results_dict)
        tf.keras.backend.clear_session()
        fold_var += 1
        tmp = pd.DataFrame(list_of_res)
        tmp.to_csv(f'Results/Improve_{dataset_name}.csv')
    return pd.DataFrame(list_of_res)


def train(model_name, dataset_name, X, Y, outer_cv=10, inner_cv=3, random_search_trials=50, inner_epochs=1,
          outer_epochs=5):
    if model_name == 'baseline':
        return train_baseline(dataset_name, X, Y, outer_cv, inner_cv, random_search_trials, inner_epochs, outer_epochs)
    elif model_name == 'paper':
        return train_paper(dataset_name, X, Y, outer_cv, inner_cv, random_search_trials, inner_epochs, outer_epochs)
    elif model_name == 'improve':
        return train_improve(dataset_name, X, Y, outer_cv, inner_cv, random_search_trials, inner_epochs, outer_epochs)
    else:
        print(f"Error! Illegal model_name found: {model_name}. model_name values can be 'baseline' , 'paper', 'improve'")
