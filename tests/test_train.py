import unittest

from b_cnn.models.bcnn_model import BCNN_Model
# class TestTrain(unittest.TestCase):
#     def test_train_is_runable(self):
#         train()


from b_cnn.controllers.trainer import Trainer
from b_cnn.config.bcnn_params import BCNN_Params
from b_cnn.controllers.dataloader import CIFAR10


class TestTrainer(unittest.TestCase):
    def test_train_is_runable(self):
        params = BCNN_Params()
        model = BCNN_Model(params=params)
        dataloader = CIFAR10(params=params)

        trainer = Trainer(model=model, params=params, dataloader=dataloader)
        trainer.train()
        trainer.estimate()


if __name__ == "__main__":
    unittest.main()
