import keras
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers import Input
from keras import backend as K

import numpy as np
from keras.datasets import cifar10

def scheduler(epoch):
    learning_rate_init = 0.003
    if epoch > 40:
        learning_rate_init = 0.0005
    if epoch > 50:
        learning_rate_init = 0.0001
    return learning_rate_init

class LossWeightsModifier(keras.callbacks.Callback):
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 8:
            K.set_value(self.alpha, 0.1)
            K.set_value(self.beta, 0.8)
            K.set_value(self.gamma, 0.1)
        if epoch == 18:
            K.set_value(self.alpha, 0.1)
            K.set_value(self.beta, 0.2)
            K.set_value(self.gamma, 0.7)
        if epoch == 28:
            K.set_value(self.alpha, 0)
            K.set_value(self.beta, 0)
            K.set_value(self.gamma, 1)


class DataLoader:
    def train_data(self):
        raise NotImplementedError(
            "train_data method is not implemented"
        )

        input_data = None
        target_data = None
        return input_data, target_data

    def validations_data(self):
        raise NotImplementedError(
            "validation_data method is not implemented"
        )
        input_data = None
        target_data = None
        return input_data, target_data

    def test_data(self):
        raise NotImplementedError(
            "test_data method is not implemented"
        )

        input_data = None
        target_data = None
        return input_data, target_data

class CIFAR10(DataLoader):
    def __init__(self, params):
        self.num_classes = 10

        #--- coarse 1 classes ---
        self.num_c_1      = params.num_c_1  # coarse 1 classes
        self.num_c_2      = params.num_c_2  # coarse 2 classes
        self.num_classes  = params.num_classes  # fine classes

    def train_data(self):
        x_train = None
        t_c1_train = None
        t_c2_train = None
        t_train = None

        input_data = x_train
        target_data = [t_c1_train, t_c2_train, t_train]
        return input_data, target_data

    def validation_data(self):
        x_test = None
        t_c1_test = None
        t_c2_test = None
        t_test = None

        input_data = x_test
        target_data = [t_c1_test, t_c2_test, t_test]
        return input_data, target_data

    def tmp(self):
        #-------------------- data loading ----------------------
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        self.y_test = y_test
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_test = x_test

        #---------------- data preprocessiong -------------------
        x_train = (x_train-np.mean(x_train)) / np.std(x_train)
        x_test = (x_test-np.mean(x_test)) / np.std(x_test)

        #---------------------- make coarse 2 labels --------------------------
        parent_f = {
            2:3, 3:5, 5:5, 1:2, 7:6, 4:6,
            0:0, 6:4, 8:1, 9:2
        }
        y_c2_train = np.zeros((y_train.shape[0], self.num_c_2)).astype("float32")
        y_c2_test = np.zeros((y_test.shape[0], self.num_c_2)).astype("float32")
        for i in range(y_c2_train.shape[0]):
            y_c2_train[i][parent_f[np.argmax(y_train[i])]] = 1.0
        for i in range(y_c2_test.shape[0]):
            y_c2_test[i][parent_f[np.argmax(y_test[i])]] = 1.0
        self.y_c2_test = y_c2_test

        #---------------------- make coarse 1 labels --------------------------
        parent_c2 = {
            0:0, 1:0, 2:0,
            3:1, 4:1, 5:1, 6:1
        }
        y_c1_train = np.zeros((y_c2_train.shape[0], self.num_c_1)).astype("float32")
        y_c1_test = np.zeros((y_c2_test.shape[0], self.num_c_1)).astype("float32")
        for i in range(y_c1_train.shape[0]):
            y_c1_train[i][parent_c2[np.argmax(y_c2_train[i])]] = 1.0
        for i in range(y_c1_test.shape[0]):
            y_c1_test[i][parent_c2[np.argmax(y_c2_test[i])]] = 1.0
        self.y_c1_test = y_c1_test

class Trainer:
    def __init__(self, model, params):
        #--- file paths ---
        self.log_path = params.log_path
        self.model_path = params.model_path


        self.batch_size   = params.batch_size
        self.epochs       = params.epochs

        self.model = model

    def train(self, dataloader):
        #-------- dimensions ---------
        img_rows, img_cols = 32, 32
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_rows, img_cols)
        else:
            input_shape = (img_rows, img_cols, 3)
        #-----------------------------

        train_size = 50000

        #----------------------- model definition ---------------------------
        alpha = K.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
        beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
        gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

        img_input = Input(shape=input_shape, name='input')

        from b_cnn.models.bcnn_model import BCNN_Model
        model = self.model(img_input)
        #----------------------- compile and fit ---------------------------
        sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      loss_weights=[alpha, beta, gamma],
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        tb_cb = TensorBoard(log_dir=str(self.log_path), histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        change_lw = LossWeightsModifier(alpha, beta, gamma)
        cbks = [change_lr, tb_cb, change_lw]

        x_train, y_train = dataloader.train_data()
        x_val, y_val = dataloader.validation_data()
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=cbks,
                  validation_data=(x_val, y_val))

        #---------------------------------------------------------------------------------
        # The following compile() is just a behavior to make sure this model can be saved.
        # We thought it may be a bug of Keras which cannot save a model compiled with loss_weights parameter
        #---------------------------------------------------------------------------------
        model.compile(loss='categorical_crossentropy',
                      # optimizer=keras.optimizers.Adadelta(),
                      optimizer=sgd,
                      metrics=['accuracy'])
        model.save(str(self.model_path))
        return model

    def estimate(self):
        from keras.models import load_model
        model = load_model(self.model_path)

        score = model.evaluate(self.x_test, [self.y_c1_test, self.y_c2_test, self.y_test], verbose=0)
        print('score is: ', score)

