from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
class BCNN_Model:
    def __init__(self, params):
        self.num_c_1 = params.num_c_1
        self.num_c_2 = params.num_c_2
        self.num_classes = params.num_classes

    def __call__(self, img_input):

        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        #--- block 2 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #--- coarse 1 branch ---
        c_1_bch = Flatten(name='c1_flatten')(x)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.num_c_1, activation='softmax', name='c1_predictions_cifar10')(c_1_bch)

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x)
        c_2_bch = Dense(512, activation='relu', name='c2_fc_cifar10_1')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(512, activation='relu', name='c2_fc2')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_pred = Dense(self.num_c_2, activation='softmax', name='c2_predictions_cifar10')(c_2_bch)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #--- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc_cifar10_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(self.num_classes, activation='softmax', name='predictions_cifar10')(x)

        model = Model(input=img_input, output=[c_1_pred, c_2_pred, fine_pred], name='medium_dynamic')
        return model
