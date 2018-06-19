import unittest


from b_cnn.tasks.train import train



# class TestTrain(unittest.TestCase):
#     def test_train_is_runable(self):
#         train()


from b_cnn.controllers.trainer import Trainer
from b_cnn.config.bcnn_params import BCNN_Params


class BCNN_Model:
    def __init__(self):
        pass

    def __call__(self):
        pass


class TestTrainer(unittest.TestCase):
    def test_train_is_runable(self):
        params = BCNN_Params()
        model = BCNN_Model()

        trainer = Trainer(model=model, params=params)
        trainer.train()


if __name__ == "__main__":
    unittest.main()
