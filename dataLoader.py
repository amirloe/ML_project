import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


"""
Data importer - for each dataset there is function called "import_{dataset_name}" 
These functions used tensorflow datasets to load datasets and save them as np array under the Datasets folder
"""


def data_load():
    import_svhn()
    import_stl()
    import_beans()
    import_casava()
    import_cmater()
    import_coloret()
    import_oxford_f()
    import_rps()
    import_ct_birds()
    import_cifar_100()


def import_cifar_100():
    dataset_cifar100 = tf.keras.datasets.cifar100.load_data()
    full_x_cifar100 = np.concatenate((dataset_cifar100[0][0], dataset_cifar100[1][0]), axis=0)
    full_y_cifar100 = np.concatenate((dataset_cifar100[0][1], dataset_cifar100[1][1]), axis=0)
    x_full_1 = np.concatenate([full_x_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(0, 20)])
    y_full_1 = np.concatenate([full_y_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(0, 20)])
    y_full_1 = np.array(y_full_1 % 20)
    x_full_2 = np.concatenate([full_x_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(20, 40)])
    y_full_2 = np.concatenate([full_y_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(20, 40)])
    y_full_2 = np.array(y_full_2 % 20)
    x_full_3 = np.concatenate([full_x_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(40, 60)])
    y_full_3 = np.concatenate([full_y_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(40, 60)])
    y_full_3 = np.array(y_full_3 % 20)
    x_full_4 = np.concatenate([full_x_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(60, 80)])
    y_full_4 = np.concatenate([full_y_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(60, 80)])
    y_full_4 = np.array(y_full_4 % 20)
    x_full_5 = np.concatenate([full_x_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(80, 100)])
    y_full_5 = np.concatenate([full_y_cifar100[np.where(full_y_cifar100 == x)[0]] for x in range(80, 100)])
    y_full_5 = np.array(y_full_5 % 20)
    np.save("Cifar100_1_X.npy", x_full_1)
    np.save("Cifar100_1_Y.npy", y_full_1)
    np.save("Cifar100_2_X.npy", x_full_2)
    np.save("Cifar100_2_Y.npy", y_full_2)
    np.save("Cifar100_3_X.npy", x_full_3)
    np.save("Cifar100_3_Y.npy", y_full_3)
    np.save("Cifar100_4_X.npy", x_full_4)
    np.save("Cifar100_4_Y.npy", y_full_4)
    np.save("Cifar100_5_X.npy", x_full_5)
    np.save("Cifar100_5_Y.npy", y_full_5)


def import_ct_birds():
    ct_data = tfds.load('caltech_birds2010', split='train+test')
    ct_data_np = tfds.as_numpy(ct_data)
    birds_dataset = np.array([x for x in ct_data_np])
    ctbirds_x = np.array([tf.image.resize(x['image'], (64, 64)).numpy() for x in birds_dataset])
    ctbirds_y = np.array([x['label'] for x in birds_dataset])
    ctbirds_y = ctbirds_y.reshape((ctbirds_y.shape[0], 1))
    ctb_x_full_1 = np.concatenate([ctbirds_x[np.where(ctbirds_y == x)[0]] for x in range(0, 70)])
    ctb_y_full_1 = np.concatenate([ctbirds_y[np.where(ctbirds_y == x)[0]] for x in range(0, 70)])
    ctb_x_full_2 = np.concatenate([ctbirds_x[np.where(ctbirds_y == x)[0]] for x in range(70, 140)])
    ctb_y_full_2 = np.concatenate([ctbirds_y[np.where(ctbirds_y == x)[0]] for x in range(70, 140)])
    ctb_y_full_2 = np.array(ctb_y_full_2 % 70)
    ctb_x_full_3 = np.concatenate([ctbirds_x[np.where(ctbirds_y == x)[0]] for x in range(140, 200)])
    ctb_y_full_3 = np.concatenate([ctbirds_y[np.where(ctbirds_y == x)[0]] for x in range(140, 200)])
    ctb_y_full_3 = np.array(ctb_y_full_3 % 70)
    np.save('Ctb_1_X.npy', ctb_x_full_1)
    np.save('Ctb_1_Y.npy', ctb_y_full_1)
    np.save('Ctb_2_X.npy', ctb_x_full_2)
    np.save('Ctb_2_Y.npy', ctb_y_full_2)
    np.save('Ctb_3_X.npy', ctb_x_full_3)
    np.save('Ctb_3_Y.npy', ctb_y_full_3)


def import_rps():
    rps_data = tfds.load('rock_paper_scissors', split='train+test')
    rps_np = tfds.as_numpy(rps_data)
    dataset_rps = np.array([x for x in rps_np])
    rps_x = np.array([tf.image.resize(x['image'], (100, 100)).numpy() for x in dataset_rps])
    rps_y = np.array([x['label'] for x in dataset_rps])
    rps_y = rps_y.reshape((rps_y.shape[0], 1))
    np.save('Rps_X.npy', rps_x)
    np.save('Rps_Y.npy', rps_y)


def import_oxford_f():
    oxford_data = tfds.load('oxford_flowers102', split='train+test+validation')
    oxford_np = tfds.as_numpy(oxford_data)
    dataset_oxford = np.array([x for x in oxford_np])
    oxford_x = np.array([tf.image.resize(x['image'], (64, 64)).numpy() for x in dataset_oxford])
    oxford_y = np.array([x['label'] for x in dataset_oxford])
    oxford_y = oxford_y.reshape((oxford_y.shape[0], 1))
    oxford_x_full_1 = np.concatenate([oxford_x[np.where(oxford_y == x)[0]] for x in range(0, 51)])
    oxford_y_full_1 = np.concatenate([oxford_y[np.where(oxford_y == x)[0]] for x in range(0, 51)])
    oxford_x_full_2 = np.concatenate([oxford_x[np.where(oxford_y == x)[0]] for x in range(51, 102)])
    oxford_y_full_2 = np.concatenate([oxford_y[np.where(oxford_y == x)[0]] for x in range(51, 102)])
    oxford_y_full_2 = np.array(oxford_y_full_2 % 51)
    np.save('Oxford_1_X.npy', oxford_x_full_1)
    np.save('Oxford_1_Y.npy', oxford_y_full_1)
    np.save('Oxford_2_X.npy', oxford_x_full_2)
    np.save('Oxford_2_Y.npy', oxford_y_full_2)


def import_coloret():
    coloret_data = tfds.load('colorectal_histology', split='train')
    coloret_np = tfds.as_numpy(coloret_data)
    dataset_coloret = np.array([x for x in coloret_np])
    coloret_x = np.array([x['image'] for x in dataset_coloret])
    coloret_y = np.array([x['label'] for x in dataset_coloret])
    coloret_y = coloret_y.reshape((coloret_y.shape[0], 1))
    np.save("Datasets/Coloret_X.npy", coloret_x)
    np.save("Datasets/Coloret_Y.npy", coloret_y)


def import_cmater():
    cmater_data = tfds.load('cmaterdb', split='train+test')
    cmater_np = tfds.as_numpy(cmater_data)
    dataset_cmater = np.array([x for x in cmater_np])
    cmater_x = np.array([x['image'] for x in dataset_cmater])
    cmater_y = np.array([x['label'] for x in dataset_cmater])
    cmater_y = cmater_y.reshape((cmater_y.shape[0], 1))
    np.save("Datasets/Cmater_X.npy", cmater_x)
    np.save("Datasets/Cmater_Y.npy", cmater_y)


def import_casava():
    casave_data = tfds.load('cassava', split='train+test+validation')
    casava_np = tfds.as_numpy(casave_data)
    dataset_casava = np.array([x for x in casava_np])
    casava_x = np.array([tf.image.resize(x['image'], (64, 64)).numpy() for x in dataset_casava])
    casava_y = np.array([x['label'] for x in dataset_casava])
    casava_y = casava_y.reshape((casava_y.shape[0], 1))
    np.save("Datasets/Casava_X.npy", casava_x)
    np.save("Datasets/Casava_Y.npy", casava_y)


def import_beans():
    data_test = tfds.load('beans', split='train+test+validation')
    test_np = tfds.as_numpy(data_test)
    dataset_beans = np.array([x for x in test_np])
    beans_x = np.array([x['image'] for x in dataset_beans])
    beans_y = np.array([x['label'] for x in dataset_beans])
    beans_x = tf.image.resize(beans_x, (32, 32)).numpy()
    beans_y = beans_y.reshape((beans_y.shape[0], 1))
    np.save("Datasets/Beans_X.npy", beans_x)
    np.save("Datasets/Beans_Y.npy", beans_y)


def import_stl():
    leaves_data = tfds.load('stl10', split='train+test')
    leaves_np = tfds.as_numpy(leaves_data)
    dataset_leaves = np.array([x for x in leaves_np])
    leaves_x = np.array([x['image'] for x in dataset_leaves])
    leaves_y = np.array([x['label'] for x in dataset_leaves])
    leaves_y = leaves_y.reshape((leaves_y.shape[0], 1))
    x_stl_1 = np.concatenate([leaves_x[np.where(leaves_y == x)[0]] for x in range(0, 4)])
    y_stl_1 = np.concatenate([leaves_y[np.where(leaves_y == x)[0]] for x in range(0, 4)])
    x_stl_2 = np.concatenate([leaves_x[np.where(leaves_y == x)[0]] for x in range(4, 8)])
    y_stl_2 = np.concatenate([leaves_y[np.where(leaves_y == x)[0]] for x in range(4, 8)])
    y_stl_2 = np.array(y_stl_2 % 4)
    x_stl_3 = np.concatenate([leaves_x[np.where(leaves_y == x)[0]] for x in range(6, 10)])
    y_stl_3 = np.concatenate([leaves_y[np.where(leaves_y == x)[0]] for x in range(6, 10)])
    y_stl_3 = np.array(y_stl_3 % 4)
    np.save("Datasets/stl_1_X.npy", x_stl_1)
    np.save("Datasets/stl_1_Y.npy", y_stl_1)
    np.save("Datasets/stl_2_X.npy", x_stl_2)
    np.save("Datasets/stl_2_Y.npy", y_stl_2)
    np.save("Datasets/stl_3_X.npy", x_stl_3)
    np.save("Datasets/stl_3_Y.npy", y_stl_3)


def import_svhn():
    svhn_data = tfds.load('svhn_cropped', split='train+test')
    svhn_np = tfds.as_numpy(svhn_data)
    dataset_svhn = np.array([x for x in svhn_np])
    svhn_x = np.array([x['image'] for x in dataset_svhn])
    svhn_y = np.array([x['label'] for x in dataset_svhn])
    svhn_y = svhn_y.reshape((svhn_y.shape[0], 1))
    x_shvn_1 = np.concatenate([svhn_x[np.where(svhn_y == x)[0]] for x in range(0, 4)])
    y_shvn_1 = np.concatenate([svhn_y[np.where(svhn_y == x)[0]] for x in range(0, 4)])
    x_shvn_2 = np.concatenate([svhn_x[np.where(svhn_y == x)[0]] for x in range(4, 8)])
    y_shvn_2 = np.concatenate([svhn_y[np.where(svhn_y == x)[0]] for x in range(4, 8)])
    y_shvn_2 = np.array(y_shvn_2 % 4)
    x_shvn_3 = np.concatenate([svhn_x[np.where(svhn_y == x)[0]] for x in range(6, 10)])
    y_shvn_3 = np.concatenate([svhn_y[np.where(svhn_y == x)[0]] for x in range(6, 10)])
    y_shvn_3 = np.array(y_shvn_3 % 4)
    np.save("Datasets/svhn_1_X", x_shvn_1)
    np.save("Datasets/svhn_1_Y", y_shvn_1)
    np.save("Datasets/svhn_2_X", x_shvn_2)
    np.save("Datasets/svhn_2_Y", y_shvn_2)
    np.save("Datasets/svhn_3_X", x_shvn_3)
    np.save("Datasets/svhn_3_Y", y_shvn_3)
