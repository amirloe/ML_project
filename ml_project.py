
import sys
import numpy as np
from dataLoader import data_load
from evaluation import friedman_test, stat_test
from experiments import train




def experiments(model_name, dataset_name):
    Models = ['baseline', 'paper', 'improve']
    Datasets = ["Beans", "Cifar100_1", "Cifar100_2", "Cifar100_3", 'Cifar100_4', 'Cifar100_5', 'Cmater', 'Ctb_1',
                'Ctb_2', 'Ctb_3',
                'Oxford_1', 'Oxford_2', 'Rps', 'Coloret', 'Casava',
                'stl_1','stl_2','stl_3', 'svhn_1','svhn_2','svhn_3']
    if dataset_name not in Datasets:
        print(f"Error dataset name {dataset_name} is not included.")
        return
    # Training parameters

    outer_cv = 10
    inner_cv = 3
    random_search_trials = 50
    inner_epochs = 5
    outer_epochs = 15
    x = np.load(f'Datasets/{dataset_name}_X.npy')
    y = np.load(f'Datasets/{dataset_name}_Y.npy')
    result = train(model_name, dataset_name, x, y, outer_cv, inner_cv, random_search_trials, inner_epochs, outer_epochs)
    result.to_csv(f'Results/{model_name}_{dataset_name}.csv')


if __name__ == "__main__":
    mod = sys.argv[1]
    if mod == "data load":
        data_load()

    elif mod == 'experiments':
        model_name = sys.argv[2]
        dataset_name = sys.argv[3]
        experiments(model_name=model_name, dataset_name=dataset_name)

    elif mod == 'evaluation':
        type = sys.argv[2]
        stat_test(type)

    else:
        print("Illegal mod! insert 'data load' , 'experiments', 'evaluation'")
