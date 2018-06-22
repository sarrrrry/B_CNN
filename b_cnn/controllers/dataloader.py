import keras
import numpy as np
from keras.datasets import cifar10


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
        x_train, t_c1_train, t_c2_train, t_train = self.tmp()[:4]

        input_data = x_train
        target_data = [t_c1_train, t_c2_train, t_train]
        return input_data, target_data

    def validation_data(self):
        x_test, t_c1_test, t_c2_test, t_test = self.tmp()[4:]

        input_data = x_test
        target_data = [t_c1_test, t_c2_test, t_test]
        return input_data, target_data

    def test_data(self):
        x_test, t_c1_test, t_c2_test, t_test = self.tmp()[4:]

        input_data = x_test
        target_data = [t_c1_test, t_c2_test, t_test]
        return input_data, target_data

    def tmp(self):
        #-------------------- data loading ----------------------
        (x_train, t_train), (x_test, t_test) = cifar10.load_data()
        t_train = keras.utils.to_categorical(t_train, self.num_classes)
        t_test = keras.utils.to_categorical(t_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #---------------- data preprocessiong -------------------
        x_train = (x_train-np.mean(x_train)) / np.std(x_train)
        x_test = (x_test-np.mean(x_test)) / np.std(x_test)

        #---------------------- make coarse 2 labels --------------------------
        # parent_f = {
        #     2:3, 3:5, 5:5, 1:2, 7:6, 4:6,
        #     0:0, 6:4, 8:1, 9:2
        # }
        #TODO: これどうやってきめたんだろうね
        #TODO: 主成分分析して見てみたい
        #TODO: 上流部分の大クラスみたいなものの定義なはず
        parent_f = {
            0:0, 1:2, 2:3, 3:5, 4:6, 5:5, 6:4, 7:6, 8:1, 9:2
        }
        t_c2_train = np.zeros((t_train.shape[0], self.num_c_2)).astype("float32")
        t_c2_test = np.zeros((t_test.shape[0], self.num_c_2)).astype("float32")
        for i in range(t_c2_train.shape[0]):
            t_c2_train[i][parent_f[np.argmax(t_train[i])]] = 1.0
        for i in range(t_c2_test.shape[0]):
            t_c2_test[i][parent_f[np.argmax(t_test[i])]] = 1.0
        # import pudb;
        # pudb.set_trace()
        print()

        #---------------------- make coarse 1 labels --------------------------
        #TODO: これどうやってきめたんだろうね
        #TODO: 主成分分析して見てみたい
        #TODO: 上流部分の大クラスみたいなものの定義なはず
        parent_c2 = {
            0:0, 1:0, 2:0,
            3:1, 4:1, 5:1, 6:1
        }
        t_c1_train = np.zeros((t_c2_train.shape[0], self.num_c_1)).astype("float32")
        t_c1_test = np.zeros((t_c2_test.shape[0], self.num_c_1)).astype("float32")
        for i in range(t_c1_train.shape[0]):
            t_c1_train[i][parent_c2[np.argmax(t_c2_train[i])]] = 1.0
        for i in range(t_c1_test.shape[0]):
            t_c1_test[i][parent_c2[np.argmax(t_c2_test[i])]] = 1.0

        return x_train, t_c1_train, t_c2_train, t_train,\
               x_test, t_c1_test, t_c2_test, t_test
