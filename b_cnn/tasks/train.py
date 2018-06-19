from b_cnn.config.mypath import MyPath
from b_cnn.controllers.trainer import Trainer

from b_cnn.config.bcnn_params import BCNN_Params
def train():
    params = BCNN_Params()
    model = None
    trainer = Trainer(model=model, params=params)
    trainer.train()

if __name__ == "__main__":
    train()
