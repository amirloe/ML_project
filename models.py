import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, AvgPool2D, Flatten, Dense, Concatenate
from tqdm import tqdm
from tensorflow.keras.applications import VGG16
import time
from sklearn.metrics import roc_auc_score, average_precision_score


class BaselineModel():
    """
    BaseLine model VGG16 network.
    """

    def __init__(self, dataset, classes, learning_rate, batch_size, pooling):
        """
        Constructor to the baseline model.
        :param dataset: The dataset, construct from ((x_train,t_train),(x_test,y_test))
        :param classes: Numer of classes in the dataset
        :param learning_rate: model's learning rate
        :param batch_size: model's batch size
        :param pooling: pooling method for the VGG model. can be MAX AVG or None
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        self.input_shape = self.x_train.shape[1:]
        self.classes = classes
        self.pooling = pooling
        self.full_model = self.bulid_model()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def bulid_model(self):
        '''
        The function builds the baseline keras model
        :return: Built model
        '''
        vgg16 = VGG16(weights=None, pooling=self.pooling, include_top=False, input_shape=self.input_shape,
                      classes=self.classes)
        model = Sequential()
        model.add(vgg16)
        #     model.add(tf.keras.layers.Dropout(0.2))
        model.add(Dense(self.classes, activation='softmax'))
        model.summary()
        return model

    def train(self, number_of_epochs):
        '''
        Run the training process
        :param number_of_epochs: Number of epochs for train
        :return: Metrics of training. loss acc etc.
        '''
        self.full_model.compile(loss="sparse_categorical_crossentropy",
                                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        hist = self.full_model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=number_of_epochs,
                                   verbose=1)
        return hist.history

    def evaluate(self):
        '''
        Evaluate and calculate all metrics according to the assignment requirements
        :return: A dictionary contains all metrices
        '''
        results_dict = {}
        prediction = self.full_model.predict(self.x_test)

        soft = np.reshape(prediction, (self.y_test.shape[0], self.classes))
        classes = np.unique(self.y_test)

        # A. Accuracy
        acc_eval = tf.keras.metrics.SparseCategoricalAccuracy()
        acc_eval.update_state(self.y_test, soft)
        acc = acc_eval.result().numpy()
        results_dict['acc'] = acc

        # B. TPR

        pred_labels = soft.argmax(axis=1)
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        for label in classes:
            for i in range(len(pred_labels)):
                if self.y_test[i][0] == pred_labels[i] == label:
                    total_tp += 1

                if pred_labels[i] == label and self.y_test[i][0] != label:
                    total_fp += 1

                if pred_labels[i] != label and self.y_test[i][0] != label:
                    total_tn += 1

                if pred_labels[i] != label and self.y_test[i][0] == label:
                    total_fn += 1

        results_dict['TPR'] = total_tp / (total_tp + total_fn)

        # C. FPR
        results_dict['FPR'] = total_fp / (total_tn + total_fp)

        # D. Precision
        results_dict['Presicion'] = total_tp / (total_tp + total_fp)

        # E. AUC – Area Under the ROC Curve
        y_true = self.y_test.reshape((self.y_test.shape[0],))
        y_pred = soft
        results_dict['AUC'] = roc_auc_score(y_true, y_pred, 'macro', multi_class='ovr')
        y_oh = tf.keras.utils.to_categorical(y_true)

        # F. Area under the Precision-Recall
        results_dict['Area under PR'] = average_precision_score(y_oh, y_pred, 'macro')

        # H. Inference time for 1000 instances
        if self.x_test.shape[0] < 1000:
            inf_data = self.x_test
        else:
            inf_data = self.x_test[:1000]
        start = time.time()
        self.full_model.predict(inf_data)
        end = time.time()
        results_dict['Inferece time'] = end - start

        return results_dict


class MT_MODEL():
    def __init__(self, dataset, batch_size, teacher_drop_out_rate, student_drop_out_rate, ema_decay):
        """
        Constructor to the paper model.
        :param dataset: The dataset, construct from ((x_train,t_train),(x_test,y_test))
        :param batch_size: Model's batch size
        :param teacher_drop_out_rate: Drop out rate for the teacher model
        :param student_drop_out_rate: Drop out rate for the student model
        :param ema_decay: Exponential moving average decay value
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.teacher_drop_out_rate = teacher_drop_out_rate
        self.student_drop_out_rate = student_drop_out_rate
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        self.input_shape = self.x_train.shape[1:]
        self.num_of_labels = len(np.unique(self.y_train))
        self.student = self.build_base_model(name="student", do_rate=self.student_drop_out_rate)
        self.teacher = self.build_base_model(name="teacher", do_rate=self.teacher_drop_out_rate)
        self.full_model = self.build_mt_model()
        self.ema_decay = ema_decay

    def build_base_model(self, do_rate, name):
        """
        Constructs keras base model for both teacher and student.
        :param do_rate: The drop out rate of the model
        :param name: name of the model - can be teacher or student.
        :return: Keras model
        """
        img_in = Input(self.input_shape, name="Img_input")
        # Data augmentation noise etc
        X = img_in
        X = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal")(X)
        X = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.3),
                                                                         width_factor=(-0.2, 0.3))(X)
        X = tf.keras.layers.GaussianNoise(stddev=0.2)(X)
        # Conv nets
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_1_1_{name}")(X)
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_1_2_{name}")(X)
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_1_3_{name}")(X)
        X = MaxPool2D((2, 2), name=f"max_pool_1_{name}")(X)
        X = Dropout(rate=do_rate, name=f'dropout_1_{name}')(X)

        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_2_1_{name}")(X)
        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_2_2_{name}")(X)
        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_2_3_{name}")(X)
        X = MaxPool2D((2, 2), name=f"max_pool_2_{name}")(X)
        X = Dropout(rate=do_rate, name=f'dropout_2_{name}')(X)

        X = Conv2D(512, (3, 3), (1, 1), padding="SAME", name=f"conv_3_1_{name}")(X)
        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_3_2_{name}")(X)
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_3_3_{name}")(X)
        X = AvgPool2D((6, 6), name=f"avg_pool_{name}")(X)
        X = Flatten()(X)

        primary = Dense(self.num_of_labels, name=f"{name}_primary")(X)
        secondary = Dense(self.num_of_labels, name=f"{name}_seondary")(X)
        out_base = Concatenate(name=f"{name}_Concat")([primary, secondary])

        model = Model(inputs=img_in, outputs=out_base, name=f"{name}_model")
        # model.summary()
        return model

    def build_mt_model(self):
        """
        Builds keras model for teacher and student combined
        :return: Keras model
        """
        self.teacher.trainable = False
        img_in = Input(self.input_shape, name="Img_input")
        out1 = self.student(img_in)
        out2 = self.teacher(img_in)
        out_total = Concatenate()([out1, out2])
        model = Model(inputs=img_in, outputs=out_total)
        # model.summary()
        return model

    def classification_costs(self, logits, labels, name=None):
        """
        THIS CODE IS BASED ON THE PAPER CODE
        Compute classification cost mean and classification cost per sample

        Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
        Compute the mean over all examples.
        Note that unlabeled examples are treated differently in error calculation.
        """
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.cast(tf.shape(per_sample)[0], tf.float32)
        mean = tf.math.divide(labeled_sum, total_count)
        return mean, per_sample

    def consistency_costs(self, logits1, logits2, name=None):
        """
        Calculates MSE between two logits
        :param logits1: logits of output 1
        :param logits2: logits of output 2
        :param name:
        :return: The mse of this two logits
        """
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        def pure_mse():
            costs = tf.losses.mse(softmax1, softmax2)
            return costs

        costs = pure_mse()
        mean_cost = tf.reduce_mean(costs)
        return mean_cost, costs

    def _mt_loss(self):
        """
        :return: Customized loss function contains both classification and consistency costs
        """

        def mt_loss(y_true, y_pred):
            y_true = tf.reshape(y_true, (y_true.shape[0],))  # Need to validate with real data
            y_true = tf.cast(y_true, tf.int32)
            num_of_labels = tf.cast(y_pred.shape[1] / 4, tf.int32)
            prim_stu = y_pred[:, 0:num_of_labels]
            sec_stu = y_pred[:, num_of_labels:2 * num_of_labels]
            prim_teach = y_pred[:, 2 * num_of_labels:3 * num_of_labels]
            sec_teach = y_pred[:, 3 * num_of_labels:4 * num_of_labels]

            class_cost_mean, class_cost_per_sample = self.classification_costs(prim_stu, y_true)
            const_cost_mean, const_cost_per_sample = self.consistency_costs(sec_stu, sec_teach)

            total_costs = tf.reduce_mean([class_cost_per_sample, const_cost_per_sample],
                                         axis=0)  # 0.5*class_loss + 0.5*const_loss

            return total_costs

        return mt_loss

    def apply_metric(self, fn):
        """
        customized metric fn
        :param fn: base function fot the customized metric
        :return:
        """

        def my_metric_fn(y_true, y_pred):
            num_of_labels = tf.cast(y_pred.shape[1] / 4, tf.int32)
            s_prediction = y_pred[:, 0:num_of_labels]
            fn.update_state(y_true, s_prediction)
            return fn.result()

        return my_metric_fn

    def update_ema(self):
        """
        This method the teacher model using ema on student's weights
        :return:
        """
        new_tw = []
        i = 0
        for s_weight, t_weight in zip(self.student.get_weights(), self.teacher.get_weights()):
            tw = self.ema_decay * t_weight + (1 - self.ema_decay) * s_weight
            new_tw.append(tw)
        self.teacher.set_weights(new_tw)

    def train(self, number_of_epochs):
        """
        Run the training process
        :param number_of_epochs: Number of epochs for train
        :return: List of losses per epoch
        """
        self.full_model.compile(loss=self._mt_loss(), optimizer='adam',
                                metrics=[self.apply_metric(tf.keras.metrics.SparseCategoricalAccuracy())])
        loss_list = []
        for epoch in range(number_of_epochs):
            print("epoch number " + str(epoch))
            batch_loss = 0
            number_of_batches = int(round(len(self.x_train) / self.batch_size))
            for i in tqdm(range(0, number_of_batches)):
                x_batch = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_batch = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]

                # ===========TRAIN=================
                # Student train
                info = self.full_model.train_on_batch(x_batch, y_batch)
                loss, acc = info
                batch_loss += loss
                # Teacher train
                self.update_ema()
            loss_list.append(batch_loss / number_of_batches)
            # print("loss per epoch: " + str(loss_list[-1]))
        return loss_list

    def evaluate(self):
        """
        Evaluate and calculate all metrics according to the assignment requirements
        :return: A dictionary contains all metrices
        """
        results_dict = {}
        prediction = self.full_model.predict(self.x_test)
        student_predictions = prediction[:, 0:self.num_of_labels]
        soft = tf.nn.softmax(student_predictions)
        classes = np.unique(self.y_test)

        # A. Accuracy
        acc_eval = tf.keras.metrics.SparseCategoricalAccuracy()
        acc_eval.update_state(self.y_test, soft)
        acc = acc_eval.result().numpy()
        results_dict['acc'] = acc

        # B. TPR
        pred_labels = soft.numpy().argmax(axis=1)
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        for label in classes:
            for i in range(len(pred_labels)):
                if self.y_test[i][0] == pred_labels[i] == label:
                    total_tp += 1

                if pred_labels[i] == label and self.y_test[i][0] != label:
                    total_fp += 1

                if pred_labels[i] != label and self.y_test[i][0] != label:
                    total_tn += 1

                if pred_labels[i] != label and self.y_test[i][0] == label:
                    total_fn += 1

        results_dict['TPR'] = total_tp / (total_tp + total_fn)

        # C. FPR

        results_dict['FPR'] = total_fp / (total_tn + total_fp)

        # D. Precision
        results_dict['Presicion'] = total_tp / (total_tp + total_fp)

        # E. AUC – Area Under the ROC Curve
        y_true = self.y_test.reshape((self.y_test.shape[0],))
        y_pred = soft.numpy()
        results_dict['AUC'] = roc_auc_score(y_true, y_pred, 'macro', multi_class='ovr')

        # F. Area under the Precision-Recall
        y_oh = tf.keras.utils.to_categorical(y_true)
        results_dict['Area under PR'] = average_precision_score(y_oh, y_pred, 'macro')

        # H. Inference time for 1000 instances
        if self.x_test.shape[0] < 1000:
            inf_data = self.x_test
        else:
            inf_data = self.x_test[:1000]
        start = time.time()
        self.full_model.predict(inf_data)
        end = time.time()
        results_dict['Inferece time'] = end - start

        return results_dict


class MT_doubleTeacher_MODEL():
    def __init__(self, dataset, batch_size, teacher_drop_out_rate, student_drop_out_rate, dummy_teacher_dropout,
                 ema_decay):
        '''
        Constructor to the improve model.
        :param dataset: The dataset, construct from ((x_train,t_train),(x_test,y_test))
        :param batch_size:
        :param teacher_drop_out_rate: Drop out rate for the student model
        :param student_drop_out_rate: Drop out rate for the student model
        :param dummy_teacher_dropout: Drop out rate for the student model
        :param ema_decay: Exponential moving average decay value
        '''

        self.dataset = dataset
        self.batch_size = batch_size
        self.teacher_drop_out_rate = teacher_drop_out_rate
        self.student_drop_out_rate = student_drop_out_rate
        self.dummy_teacher_drop_out_rate = dummy_teacher_dropout
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        self.inputshape = self.x_train.shape[1:]
        self.num_of_labels = len(np.unique(self.y_train))
        self.student = self.build_base_model(name="student", do_rate=self.student_drop_out_rate)
        self.EMA_teacher = self.build_base_model(name="EMA_teacher", do_rate=self.teacher_drop_out_rate)
        self.dummy_teacher = self.build_base_model(name="dummy_teacher", do_rate=self.dummy_teacher_drop_out_rate)
        self.full_model = self.build_mt_model()
        self.ema_decay = ema_decay

    def build_base_model(self, do_rate, name):
        """
        Constructs keras base model for both teachers and student.
        :param do_rate: The drop out rate of the model
        :param name: name of the model - can be EMA_teacher dummy_teacher or student.
        :return: Keras model
        """
        img_in = Input(self.inputshape, name="Img_input")
        # Data augmentation noise etc
        X = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal")(img_in)
        X = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.3),
                                                                         width_factor=(-0.2, 0.3))(X)
        X = tf.keras.layers.GaussianNoise(stddev=0.2)(X)
        # Conv nets
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_1_1_{name}")(X)
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_1_2_{name}")(X)
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_1_3_{name}")(X)
        X = MaxPool2D((2, 2), name=f"max_pool_1_{name}")(X)
        X = Dropout(rate=do_rate, name=f'dropout_1_{name}')(X)

        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_2_1_{name}")(X)
        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_2_2_{name}")(X)
        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_2_3_{name}")(X)
        X = MaxPool2D((2, 2), name=f"max_pool_2_{name}")(X)
        X = Dropout(rate=do_rate, name=f'dropout_2_{name}')(X)

        X = Conv2D(512, (3, 3), (1, 1), padding="SAME", name=f"conv_3_1_{name}")(X)
        X = Conv2D(256, (3, 3), (1, 1), padding="SAME", name=f"conv_3_2_{name}")(X)
        X = Conv2D(128, (3, 3), (1, 1), padding="SAME", name=f"conv_3_3_{name}")(X)
        X = AvgPool2D((6, 6), name=f"avg_pool_{name}")(X)
        X = Flatten()(X)

        primary = Dense(self.num_of_labels, name=f"{name}_primary")(X)
        secondary = Dense(self.num_of_labels, name=f"{name}_seondary")(X)
        out_base = Concatenate(name=f"{name}_Concat")([primary, secondary])

        model = Model(inputs=img_in, outputs=out_base, name=f"{name}_model")
        # model.summary()
        return model

    def build_mt_model(self):
        """
        Builds keras model for both teachers and student combined
        :return: Keras model
        """
        self.EMA_teacher.trainable = False
        img_in = Input(self.inputshape, name="Img_input")
        out1 = self.student(img_in)
        out2 = self.EMA_teacher(img_in)
        out3 = self.dummy_teacher(img_in)
        out_total = Concatenate()([out1, out2, out3])
        model = Model(inputs=img_in, outputs=out_total)
        # model.summary()
        return model

    def classification_costs(self, logits, labels, name=None):
        """
        THIS CODE IS BASED ON THE PAPER CODE
        Compute classification cost mean and classification cost per sample

        Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
        Compute the mean over all examples.
        Note that unlabeled examples are treated differently in error calculation.
        """
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.cast(tf.shape(per_sample)[0], tf.float32)
        mean = tf.math.divide(labeled_sum, total_count)
        return mean, per_sample

    def consistency_costs(self, logits1, logits2, name=None):
        """
        Calculates MSE between two logits
        :param logits1: logits of output 1
        :param logits2: logits of output 2
        :param name:
        :return: The mse of this two logits
        """
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        def pure_mse():
            costs = tf.losses.mse(softmax1, softmax2)
            return costs

        costs = pure_mse()
        mean_cost = tf.reduce_mean(costs)
        return mean_cost, costs

    def _mt_loss(self):
        """
        :return: Customized loss function contains both classification and consistency costs
        """
        def mt_loss(y_true, y_pred):
            y_true = tf.reshape(y_true, (y_true.shape[0],))  # Need to validate with real data
            y_true = tf.cast(y_true, tf.int32)
            num_of_labels = tf.cast(y_pred.shape[1] / 6, tf.int32)
            prim_stu = y_pred[:, 0:num_of_labels]
            sec_stu = y_pred[:, num_of_labels:2 * num_of_labels]
            prim_teach = y_pred[:, 2 * num_of_labels:3 * num_of_labels]
            sec_teach = y_pred[:, 3 * num_of_labels:4 * num_of_labels]
            prim_dummy_teach = y_pred[:, 4 * num_of_labels:5 * num_of_labels]
            sec_dummy_teach = y_pred[:, 5 * num_of_labels:6 * num_of_labels]

            class_cost_mean, class_cost_per_sample = self.classification_costs(prim_stu, y_true)
            const_cost_mean, const_cost_per_sample = self.consistency_costs(sec_stu, sec_teach)
            const_cost_mean2, const_cost_per_sample2 = self.consistency_costs(sec_stu, sec_dummy_teach)

            total_costs = tf.reduce_mean([class_cost_per_sample, const_cost_per_sample, const_cost_per_sample2],
                                         axis=0)  # 0.5*class_loss + 0.5*const_loss

            return total_costs

        return mt_loss

    def apply_metric(self, fn):
        """
        customized metric fn
        :param fn: base function fot the customized metric
        :return:
        """

        def my_metric_fn(y_true, y_pred):
            num_of_labels = tf.cast(y_pred.shape[1] / 6, tf.int32)
            s_prediction = y_pred[:, 0:num_of_labels]
            # print(y_true)
            fn.update_state(y_true, s_prediction)
            return fn.result()

        return my_metric_fn

    def update_ema(self):
        """
        This method the teacher model using ema on student's weights
        :return:
        """
        new_tw = []
        i = 0
        for s_weight, t_weight in zip(self.student.get_weights(), self.EMA_teacher.get_weights()):
            tw = self.ema_decay * t_weight + (1 - self.ema_decay) * s_weight
            new_tw.append(tw)
        self.EMA_teacher.set_weights(new_tw)

    def train(self, number_of_epochs):
        """
        Run the training process
        :param number_of_epochs: Number of epochs for train
        :return: List of losses per epoch
        """
        self.full_model.compile(loss=self._mt_loss(), optimizer='adam',
                                metrics=[self.apply_metric(tf.keras.metrics.SparseCategoricalAccuracy())])
        loss_list = []
        for epoch in range(number_of_epochs):
            print("epoch number " + str(epoch))
            batch_loss = 0
            number_of_batches = round(len(self.x_train) / self.batch_size)
            for i in tqdm(range(0, number_of_batches)):
                x_batch = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_batch = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]

                # ===========TRAIN=================
                # Student train
                info = self.full_model.train_on_batch(x_batch, y_batch)
                loss, acc = info
                batch_loss += loss
                # Teachers train
                self.update_ema()
                self.dummy_teacher.set_weights(self.student.get_weights())

            loss_list.append(batch_loss / number_of_batches)

        return loss_list

    def evaluate(self):
        """
        Evaluate and calculate all metrics according to the assignment requirements
        :return: A dictionary contains all metrices
        """
        results_dict = {}
        prediction = self.full_model.predict(self.x_test)
        student_predictions = prediction[:, 0:self.num_of_labels]
        soft = tf.nn.softmax(student_predictions)
        classes = np.unique(self.y_test)
        # A. Accuracy
        acc_eval = tf.keras.metrics.SparseCategoricalAccuracy()
        acc_eval.update_state(self.y_test, soft)
        acc = acc_eval.result().numpy()
        results_dict['acc'] = acc

        # B. TPR
        pred_labels = soft.numpy().argmax(axis=1)
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        for label in classes:
            for i in range(len(pred_labels)):
                if self.y_test[i][0] == pred_labels[i] == label:
                    total_tp += 1

                if pred_labels[i] == label and self.y_test[i][0] != label:
                    total_fp += 1

                if pred_labels[i] != label and self.y_test[i][0] != label:
                    total_tn += 1

                if pred_labels[i] != label and self.y_test[i][0] == label:
                    total_fn += 1

        results_dict['TPR'] = total_tp / (total_tp + total_fn)

        # C. FPR
        results_dict['FPR'] = total_fp / (total_tn + total_fp)

        # D. Precision
        results_dict['Presicion'] = total_tp / (total_tp + total_fp)

        # E. AUC – Area Under the ROC Curve
        y_true = self.y_test.reshape((self.y_test.shape[0],))
        y_pred = soft.numpy()
        results_dict['AUC'] = roc_auc_score(y_true, y_pred, 'macro', multi_class='ovr')

        # F. Area under the Precision-Recall
        y_oh = tf.keras.utils.to_categorical(y_true)
        results_dict['Area under PR'] = average_precision_score(y_oh, y_pred, 'macro')

        # H. Inference time for 1000 instances
        if self.x_test.shape[0] < 1000:
            inf_data = self.x_test
        else:
            inf_data = self.x_test[:1000]
        start = time.time()
        self.full_model.predict(inf_data)
        end = time.time()
        results_dict['Inferece time'] = end - start

        return results_dict
