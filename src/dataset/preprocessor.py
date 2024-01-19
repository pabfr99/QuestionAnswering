import os
from typing import List, Optional, Dict
import logging
import pandas as pd
from tqdm.notebook import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self,
                 directory_path: str,
                 show_progress_bar: bool = False) -> None:
        """
        Initialize the DataPreprocessor module.

        Args:
            - directory_path (str): The path to the directory containing the data files.
            - show_progress_bar (bool): Whether to show a bar during extraction operations
        """

        self._validate_initial_inputs(directory_path, show_progress_bar)
        # set init attributes
        self.directory_path = directory_path
        self.show_progress_bar = show_progress_bar

        logger.info(f'DataPreprocessor initialized with directory: {directory_path}')

    def _validate_initial_inputs(self,
                                 directory_path: str,
                                 show_progress_bar: bool) -> None:
        """Helper function to validate the initial inputs."""

        if not isinstance(directory_path, str):
            raise TypeError('directory_path must be a string.')

        if not isinstance(show_progress_bar, bool):
            raise TypeError('show_progress_bar must be a boolean.')

        if not os.path.exists(directory_path):
            logger.info(f'Directory at {directory_path} not found. Creating the directory.')
            os.makedirs(directory_path)
        else:
            logger.info(f'Found existing directory at {directory_path}.')

    def _process_files(self,
                       file_paths: List[str],
                       samples_per_module: int) -> pd.DataFrame:
        """
        Extract the files and the required information.

        Args:
            - file_paths (List[str]): List of file paths to extract.
            - samples_per_module (int): Number of samples to extract from each module.

        Yields:
            - pd.DataFrame: A DataFrame containing the questions, the answers, the folder and module names.
        """
        logger.info('Starting to process files.')

        if self.show_progress_bar:
            file_paths = tqdm(file_paths, desc="Processing files")

        # extract the data for each file specified
        for file_path in file_paths:
            questions = []
            answers = []
            modules = []
            folders = []
            # extract the folder and module names
            folder_name = os.path.basename(os.path.dirname(file_path))
            module_name = os.path.basename(file_path)[:-4]
            with open(file_path, 'r') as file:
                # read the lines from the file up to the specified count
                count = 0
                for line in file:
                    if count >= samples_per_module:
                        break
                    question = line.strip()
                    answer = next(file).strip()
                    questions.append(question)
                    answers.append(answer)
                    modules.append(module_name)
                    folders.append(folder_name)
                    count += 1
            yield pd.DataFrame({
                'question': questions,
                'answer': answers,
                'folder': folders,
                'module': modules
            })

    def extract_data(self,
                     folders_to_process: Optional[List[str]] = None,
                     files_to_process: Optional[List[str]] = None,
                     samples_per_module: int = 10000) -> pd.DataFrame:
        """
        Extract the data from the specified subdirectories.

        Args:
            - folders_to_process (List[str], optional): List of subdirectories to process. If None, all subdirectories will be processed.
            - files_to_process (List[str], optional): List of files to extract. If None, all files in the specified folders will be extracted.
            - samples_per_module (int): Number of samples to extract from each module, default to 10000.

        Returns:
            - pd.DataFrame: DataFrame containing data from all the specified directories.
        """
        # error checks for the method arguments
        if not isinstance(samples_per_module, int):
            raise TypeError('samples_per_module must be an integer.')
        if samples_per_module <= 0:
            raise ValueError('samples_per_module must be a positive integer.')

        if folders_to_process is not None and not all(isinstance(folder, str) for folder in folders_to_process):
            raise TypeError('All elements in folders_to_process must be strings.')

        if files_to_process is not None and not all(isinstance(file, str) for file in files_to_process):
            raise TypeError('All elements in files_to_process must be strings.')

        logger.info(f'Starting data extraction from {self.directory_path} with {samples_per_module} samples per module.')

        # define the folders' paths
        if folders_to_process:
            folder_paths = [os.path.join(self.directory_path, folder) for folder in folders_to_process if os.path.isdir(os.path.join(self.directory_path, folder))]
        else:
            folder_paths = [os.path.join(self.directory_path, folder) for folder in os.listdir(self.directory_path) if os.path.isdir(os.path.join(self.directory_path, folder))]

        # define the files' paths
        if files_to_process:
            file_paths = [os.path.join(subdir, fl) for subdir in folder_paths for fl in files_to_process if os.path.isfile(os.path.join(subdir, fl))]
        else:
            file_paths = [os.path.join(subdir, file) for subdir in folder_paths for file in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, file))]

        # checks for missing or incorrect paths
        if not folder_paths:
            raise ValueError('None of the specified folders were found.')
        if files_to_process and not file_paths:
            raise ValueError('None of the specified files were found.')

        # concatenate the data from the subdirectories
        data = pd.concat(self._process_files(file_paths, samples_per_module), ignore_index=True)

        logger.info(f'Data extraction completed. Processed {len(file_paths)} files from {len(folder_paths)} folders.')

        return data

    def split_by_folder(self,
                        data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the samples based on the folder type.

        Args:
            - data (pd.DataFrame): The input DataFrame containing the 'folder' column.

        Returns:
            - Dict[str, pd.DataFrame]: A dictionary containing the splitted DataFrames.
        """
        # error checks for the method argument
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data must be a pandas DataFrame.')

        if data.empty:
            logger.warning('The input DataFrame is empty. Returning an empty dictionary.')
            return {}

        if 'folder' not in data.columns:
            raise ValueError('The DataFrame must contain a "folder" column.')

        logger.info('Splitting data by folder.')

        # get the unique folder names
        folders = data['folder'].unique()
        # return the dictionary containing the splitted dataframes
        return {folder: data[data['folder'] == folder] for folder in folders}

    def split_by_module(self,
                        data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the samples based on the module.

        Args:
            - data (pd.DataFrame): The input DataFrame containing the 'module' column.

        Returns:
            - Dict[str, pd.DataFrame]: A dictionary containing the splitted DataFrames.
        """
        # error checks for the method argument
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data must be a pandas DataFrame.')
        if data.empty:
            logger.warning('The input DataFrame is empty. Returning an empty dictionary.')
            return {}
        if 'module' not in data.columns:
            raise ValueError('The DataFrame must contain a "module" column.')

        logger.info('Splitting data by module.')

        # get the unique module names
        modules = data['module'].unique()
        # return the dictionary containing the splitted dataframes
        return {module: data[data['module'] == module] for module in modules}

    def save_to_file(self,
                     data: pd.DataFrame,
                     file_path: str) -> None:
        """
        Save the given DataFrame in a csv file.

        Args:
            - data (pd.DataFrame): The DataFrame to be saved.
            - file_path (str): The destination file path.
        """
        # error checks for the method arguments
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data must be a pandas DataFrame.')

        if not isinstance(file_path, str):
            raise TypeError('file_path must be a string.')

        logger.info(f'Saving data to {file_path}.')

        # get the directory name and eventually create it
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # check if the dataframe is empty and save to csv
        if data.empty:
            logger.warning('The DataFrame is empty. An empty file will be saved.')
        data.to_csv(file_path, index=False)

        logger.info(f'Data successfully saved to {file_path}.')

    def load_from_file(self,
                       file_path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a csv file.

        Args:
            - file_path (str): The path to the csv file to be read.

        Returns:
            - pd.DataFrame: The DataFrame loaded.
        """
        # error checks for the method argument
        if not isinstance(file_path, str):
            raise TypeError('file_path must be a string.')

        logger.info(f'Loading data from {file_path}.')

        # get the dataframe from csv file
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, dtype=str)
            # check if the dataframe is empty
            if data.empty:
                logger.warning('The loaded DataFrame is empty.')
            else:
                logger.info(f'Successfully loaded data from {file_path}. DataFrame has {len(data)} rows.')
            return data
        else:
            raise FileNotFoundError(f'The file at the provided path {file_path} was not found.')
