# ML project
Amir Loewenthal 205629124

Ron Keller 312501703

This project is based on the paper "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results."  by Tarvainen, Antti, and Harri Valpola. [link to the paper](https://arxiv.org/abs/1703.01780)

The projects contains implamentation of the paper and comparison to a baseline and an improvment of the paper.

This is the final project of the course Machine Learning by SISE departmant at BGU 2021.

## Instructions ##

To execute the code you need to run the file "ml_project.py" that code recicve one parameter from command line. There are three options for running the code:

### Data load ###

By running the command "python ml_project.py data_load" the program will load the 20 datasets as npy files to the "Datasets" folder.

### Experiments ###
By running the command "python ml_project.py experiments <model_name> <dataset_name>" the program will do the training for the chosen model and dataset.

List of models : 'baseline' 'paper' 'improve'

List of datasets: "Beans", "Cifar100_1", "Cifar100_2", "Cifar100_3", 'Cifar100_4', 'Cifar100_5', 'Cmater', 'Ctb_1',
                'Ctb_2', 'Ctb_3',
                'Oxford_1', 'Oxford_2', 'Rps', 'Coloret', 'Casava',
                'stl_1','stl_2','stl_3', 'svhn_1','svhn_2','svhn_3'
                
 Detailed information about the models and dataset is in the pdf report.
 
 ### Evaluation ###
 By running the command "python ml_project.py evaluation <experiments_type>" the program will run statistic tests based on the excel files that are in the 'Results' folder.
 experiments_type can be 'supervised' or 'semisupervised'
 
