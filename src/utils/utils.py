import os
from typing import Optional

def define_path(data_folder: str,
                dataframe_type: str,
                module_name: Optional[str] = None,
                samples_per_module: int = 0,
                tokenized: bool = False) -> str:
    """
    Constructs the path to the DataFrame based on the provided parameters.

    Args:
        - data_folder (str): The root folder where the data is stored.
        - dataframe_type (str): The type of data. Either 'train' or 'test'.
        - module_name (str, optional): The name of the module. If None, it indicates a mixed dataset.
        - samples_per_module (int): The number of samples per module of the dataframe to be saved/loaded.
        - tokenized (bool): If True, define the path to the tokenized data. Default to False.

    Returns:
        - str: The path to the DataFrame.
    """
    # error checks for the arguments
    if not isinstance(data_folder, str):
        raise TypeError('"data_folder" must be a string.')

    if not isinstance(dataframe_type, str):
        raise TypeError('"dataframe_type" must be a string.')

    if not isinstance(samples_per_module, int):
        raise TypeError('"samples_per_module" must be an integer.')

    if samples_per_module <= 0:
        raise ValueError('"samples_per_module" must be a positive integer.')

    if dataframe_type not in ['train', 'test']:
        raise ValueError(f'Invalid dataframe_type: {dataframe_type}. Expected "train" or "test".')

    if tokenized:
        df_dir = 'tokenized_dataframe'
    else:
        df_dir = 'dataframe'
    # define the path to the dataframe directory
    base_path = os.path.join(data_folder, f'{df_dir}/{dataframe_type}')

    # define the path to the dataframe
    if not module_name:
        sub_folder = 'mixed'
        file_name = 'interpolate' if dataframe_type=='test' else f'mixed_{str(samples_per_module)}'
    else:
        sub_folder = 'unique_module'
        file_name = module_name if dataframe_type=='test' else f'{module_name}_{str(samples_per_module)}'

    return os.path.join(base_path, sub_folder, file_name)
