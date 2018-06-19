from pathlib import Path


class MyPath:

    PROJECT_ROOT = Path('/workspace/B_CNN/b_cnn')
    STORAGE = PROJECT_ROOT/'storage'

    #--- file paths ---
    log_filepath = str(Path('./tb_log_medium_dynamic/'))
    weights_store_filepath = str(Path('./medium_dynamic_weights/'))
    if not Path(weights_store_filepath).exists():
        Path(weights_store_filepath).mkdir()

    train_id = '1'
    model_name = str(Path(
        'weights_medium_dynamic_cifar_10_{}.h5'.format(train_id)
    ))
    model_path = str(Path(weights_store_filepath, model_name))
