from pathlib import Path

from b_cnn.config.mypath import MyPath


class BCNN_Params:
    def __init__(self, **params):
        self.log_path = Path(params.pop('log_path', MyPath.log_filepath))
        self.model_path = Path(params.pop('model_path', MyPath.model_path))

        #--- coarse 1 classes ---
        self.num_c_1      = 2  # coarse 1 classes
        self.num_c_2      = 7  # coarse 2 classes
        self.num_classes  = 10  # fine classes

        self.batch_size   = 128
        # self.epochs       = 60
        self.epochs       = 1
