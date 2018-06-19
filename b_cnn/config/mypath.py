from pathlib import Path


class MyPath:

    PROJECT_ROOT = Path('/workspace/B_CNN/b_cnn')
    STORAGE = PROJECT_ROOT/'storage'

    #--- file paths ---
    log_filepath = STORAGE/'tb_log_medium_dynamic'
    weights_store_filepath = STORAGE/'medium_dynamic_weights'
    if not weights_store_filepath.exists():
        Path(weights_store_filepath).mkdir()

    train_id = 1
    model_path = weights_store_filepath/f'weights_medium_dynamic_cifar_10_{train_id}.h5'.format(train_id)
