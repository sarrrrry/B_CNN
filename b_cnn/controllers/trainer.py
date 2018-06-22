import keras
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers import Input
from keras import backend as K


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


from b_cnn.controllers.dataloader import DataLoader
class Trainer:
    def __init__(self, model, dataloader, params):
        self.dataloader = dataloader
        if not isinstance(dataloader, DataLoader):
            raise AttributeError(
                "dataloader must be instance of DataLoader"
            )


        #--- file paths ---
        self.log_path = params.log_path
        self.model_path = params.model_path

        self.batch_size   = params.batch_size
        self.epochs       = params.epochs

        self.model = model

    def train(self):
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

        x_train, y_train = self.dataloader.train_data()
        x_val, y_val = self.dataloader.validation_data()
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
        # return model

    def estimate(self):
        from keras.models import load_model
        model = load_model(self.model_path)

        x_test, y_test = self.dataloader.test_data()
        score = model.evaluate(x_test, y_test, verbose=0)
        print('score is: ', score)

