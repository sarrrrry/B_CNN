from pathlib import Path

from b_cnn.config.mypath import MyPath


class BCNN_Params:
    def __init__(self, **params):
        self.log_path = Path(params.pop('log_path', MyPath.log_filepath))
        self.model_path = Path(params.pop('model_path', MyPath.model_path))