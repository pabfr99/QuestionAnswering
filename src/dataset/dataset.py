import os
import pickle
import logging
from typing import List, Iterator, Dict, Optional
from tqdm.notebook import tqdm
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):

    def __init__(self,
                 data: pd.DataFrame) -> None:
        """
        Initialize the TextDataset Module.

        Args:
            - data (pd.DataFrame): A Pandas DataFrame containing the 'question' and 'answer' columns.
        """
        # error checks for the method argument
        if data.empty:
            raise ValueError('The input DataFrame is empty. Cannot initialize class attributes.')

        if 'question' not in data.columns or 'answer' not in data.columns:
            raise ValueError('DataFrame must contain "question" and "answer" columns.')

        # initialize the attributes
        self.data = data[['question', 'answer']]
        self.special_tokens = ['<unk>', '<eos>', '<bos>', '<pad>']
        self.tokenized_questions = None
        self.tokenized_answers = None
        self.vocabulary_to_index = None
        self.index_to_vocabulary = None
        self.vocab = None

        logger.info(f'Initialized TextDataset with {len(self.data)} entries.')

    def load_tokenized_data(self,
                            load_path: str) -> None:
        """
        Load tokenized questions and answers from the specified path.

        Args:
            - load_path (str): Prefix path to the tokenized data. The actual file should have "_tokenized_data.pkl" appended to this prefix.
        """
        # error checks for the load_path argument
        if not isinstance(load_path, str):
            raise TypeError('load_path must be a string.')

        file_path = f'{load_path}_tokenized_data.pkl'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'No tokenized data found at {file_path}')

        with open(file_path, 'rb') as f:
            tokenized_data = pickle.load(f)

        # define the tokenized questions and answers
        self.tokenized_questions = tokenized_data['question'].values.tolist()
        self.tokenized_answers = tokenized_data['answer'].values.tolist()

        logger.info('Tokenized data loaded successfully.')

    def save_tokenized_data(self,
                            save_path: str) -> None:
        """
        Save tokenized questions and answers at the specified path.

        Args:
            - save_path (str): Prefix path to the tokenized data. The actual file will have "_tokenized_data.pkl" appended to this prefix.
        """
        # error checks for the argument
        if not isinstance(save_path, str):
            raise TypeError('save_path must be a string.')

        logger.info(f'Attempting to save the tokenized data to {save_path}.')

        save_path = f'{save_path}_tokenized_data.pkl'
        # specify the directory and eventually create it
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # create the dataframe and save it
        tokenized_data = pd.DataFrame({
            'question': self.tokenized_questions,
            'answer': self.tokenized_answers
        })

        with open(save_path, 'wb') as f:
            pickle.dump(tokenized_data, f)

        logger.info('Tokenized data saved successfully.')

    def tokenize_text(self,
                      text: List[str]) -> List[List[str]]:
        """
        Tokenize the sentences using the vocabulary.

        Args:
            - text (List[str]): The list of sentences to tokenize.

        Returns:
            - List[List[str]]: The tokenized sentences.
        """
        # error check for the argument
        if not all(isinstance(t, str) for t in text):
            raise TypeError('text must be a list of strings.')

        # tokenize the data
        tokenized_text = []
        for t in tqdm(text, desc='Tokenizing text'):
            tokenized_example = self.vocab([self.start_token] + list(t) + [self.end_token])
            tokenized_text.append(tokenized_example)
        return tokenized_text

    def tokenize_data(self) -> None:
        """Tokenize the question and answer columns."""
        logger.info('Tokenizing the data.')

        # extract questions and answers
        questions = self.data['question'].values.tolist()
        answers = self.data['answer'].values.tolist()

        # tokenize and save the data
        self.tokenized_questions = self.tokenize_text(questions)
        self.tokenized_answers = self.tokenize_text(answers)

        logger.info('Data tokenized successfully.')

    def load_vocab(self,
                   load_path: str) -> None:
        """
        Load the vocabulary from the specified path prefix. The function expects the vocabulary file to have a "_vocab.pth" suffix.

        Args:
            - load_path (str): The path prefix (without suffix) from which the vocabulary should be loaded.
        """
        logger.info(f'Attempting to load vocabulary from {load_path}_vocab.pth.')

        # error checks for the load_path argument
        if not isinstance(load_path, str):
            raise TypeError('load_path must be a string.')

        # check if the file exists
        if not os.path.exists(f'{load_path}_vocab.pth'):
            raise FileNotFoundError(f'No vocabulary found at {load_path}_vocab.pth')

        # load the vocabulary and set up the parameters
        self.vocab = torch.load(f'{load_path}_vocab.pth')
        self.setup_vocab_params(self.vocab, self.special_tokens)

        logger.info('Vocabulary loaded successfully.')

    def save_vocab(self,
                   save_path: str) -> None:
        """
        Save the vocabulary to the specified path. The file will be saved with a "_vocab.pth" suffix.

        Args:
            - save_path (str): Prefix path to save the vocabulary. The file will be saved with a "_vocab.pth" suffix.
        """
        logger.info(f'Attempting to save vocabulary to {save_path}_vocab.pth.')

        # specify the directory and eventually create it
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save the vocabulary
        torch.save(self.vocab, f'{save_path}_vocab.pth')

        logger.info('Vocabulary saved successfully.')

    def setup_vocab_params(self,
                           vocab,
                           special_tokens) -> None:
        """Set up the vocabulary parameters for tokenization."""
        # set some class attributes from the vocabulary
        self.vocab = vocab
        self.vocabulary_to_index = vocab.get_stoi()
        self.index_to_vocabulary = vocab.get_itos()
        self.unk_token, self.end_token, self.start_token, self.pad_token = special_tokens

        logger.info('Vocabulary parameters set up.')

    def _yield_tokens(self,
                     questions: List[str],
                     answers: List[str]) -> Iterator[List[str]]:
        """
        Generator that yields tokenized sentences.

        Args:
            - questions (List[str]): List of question.
            - answers (List[str]): List of answer.

        Yields:
            - List[str]: The sentence as a list of characters.
        """
        for question in tqdm(questions, desc='Processing questions'):
            yield list(question)

        for answer in tqdm(answers, desc='Processing answers'):
            yield list(answer)

    def build_vocab(self,
                    questions: List[str],
                    answers: List[str]) -> None:
        """
        Build the vocabulary from the given questions and answers.

        Args:
            - questions (List[str]): The list of questions.
            - answers (List[str]): The list of answers.
        """
        # error checks for the arguments
        if not all(isinstance(q, str) for q in questions):
            raise TypeError('questions must be a list of strings.')

        if not all(isinstance(a, str) for a in answers):
            raise TypeError('answers must be a list of strings.')

        # build the vocabulary from the questions and answers
        self.vocab = build_vocab_from_iterator(self._yield_tokens(questions, answers),
                                               specials=self.special_tokens)
        self.vocab.set_default_index(self.vocab['<unk>'])
        # set up the vocabulary parameters
        self.setup_vocab_params(self.vocab, self.special_tokens)

        logger.info("Vocabulary built successfully.")

    def build_and_save_vocab(self,
                             save_path: str=None) -> None:
        """
        Build the vocabulary from the dataset and save it to that path.

        Args:
            - save_path (str): Prefix path to save the vocabulary. The file will be saved with a "_vocab.pth" suffix.
        """
        # error check for the argument
        if not isinstance(save_path, str):
            raise TypeError('save_path must be a string.')

        logger.info('Building and saving the vocabulary.')

        # extract the questions and the answers
        questions = self.data['question'].values.tolist()
        answers = self.data['answer'].values.tolist()

        # build and save the vocabulary
        self.build_vocab(questions, answers)
        self.save_vocab(save_path=save_path)

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            - int: The number of samples.
        """
        if self.tokenized_questions is None:
            raise RuntimeError(
                """
                Tokenized data is missing. Please ensure you've tokenized the data or loaded pre-tokenized data.
                To tokenize the data, call the 'tokenize_and_save_data()' method.
                If you have pre-tokenized data, use 'load_tokenized_data()' to load it.
                """
            )
        return len(self.tokenized_questions)

    def __getitem__(self,
                    idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            - idx (int): The index of the sample.

        Returns:
            - Dict[str, torch.Tensor]: A dictionary containing tokenized question and answer for the given index.
        """
        if self.tokenized_questions is None or self.tokenized_answers is None:
            raise RuntimeError(
                """
                Tokenized data is missing.
                To tokenize the data, call the 'tokenize_and_save_data()' method.
                If you have pre-tokenized data, use 'load_tokenized_data()' to load it."""
            )
        question = torch.LongTensor(self.tokenized_questions[idx])
        answer = torch.LongTensor(self.tokenized_answers[idx])
        return {'questions': question,
              'answers': answer}
              
              
class Collator:
    def __init__(self,
                 pad_token: int) -> None:
        """
        Initialize the Collator.

        Args:
            - pad_token (int): The index corresponding to the padding in the vocabulary. Used to pad the data.
        """
        self.pad_token = pad_token

    def create_src_mask(self,
                        seq: Tensor) -> Tensor:
        """
        Generates the padding mask for the source tensor.

        Args:
            - seq (Tensor): Target tensor of shape (batch_size, batch_max_sequence_length).

        Returns:
            - mask (Tensor): Tensor of shape (batch_size, 1, 1, batch_max_sequence_length).
        """
        return (seq != self.pad_token).unsqueeze(1).unsqueeze(2)

    def create_trg_mask(self,
                        trg: Tensor) -> Tensor:
        """
        Generates the mask for the target tensor.

        Args:
            - trg (Tensor): Target tensor of shape (batch_size, batch_max_sequence_length).

        Returns:
            - mask (Tensor): Tensor of shape (batch_size, 1, batch_max_sequence_length, batch_max_sequence_length)
        """
        # create the padding mask
        trg_pad_mask = (trg != self.pad_token).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        # create the lookahead mask to avoid attending future positions
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool().unsqueeze(0)
        trg_pad_mask = trg_pad_mask.expand(-1, 1, trg_len, trg_len)
        # combine the two masks
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def collate_fn(self,
                   batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Applies the preproccessing operations over the batch.

        Args:
            - batch (List[Dict[str, Tensor]]): List of dictionaries, each containing a single sample 'questions' and 'answers'.

        Returns:
            - dict: A dictionary containing the following keys:
                'questions' (torch.Tensor): Padded questions.
                'answers' (torch.Tensor): Padded answers.
                'questions_mask' (torch.Tensor): Source mask for questions.
                'answers_mask' (torch.Tensor): Target mask for answers.
        """
        questions = [item['questions'] for item in batch]
        answers = [item['answers'] for item in batch]
        # pad the questions and the answers
        questions_padded = pad_sequence(questions, batch_first=True, padding_value=self.pad_token)
        answers_padded = pad_sequence(answers, batch_first=True, padding_value=self.pad_token)
        # create the mask
        questions_mask = self.create_src_mask(questions_padded)
        answers_mask = self.create_trg_mask(answers_padded)
        return {'questions': questions_padded,
                'answers': answers_padded,
                'questions_mask': questions_mask,
                'answers_mask': answers_mask}
                
class TextDataModule(LightningDataModule):
    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 load_path: Optional[str] = None,
                 save_path: Optional[str] = None,
                 batch_size: int = 32) -> None:
        """
        Initialize the DataModule.

        Args:
            - train_data (pd.DataFrame): The DataFrame to be processed for training and validation.
            - test_data (pd.DataFrame): The DataFrame to be processed for testing.
            - load_path (str, optional): The path prefix (without suffix) from which the data and the vocabulary should be loaded.
            - save_path (str, optional): The path prefix (without suffix) to which the data and the vocabulary will be saved.
            - batch_size (int): The batch size, defaults to 32.
        """
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.load_path = load_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.train_val_dataset = TextDataset(self.train_data)
        self.test_dataset = TextDataset(self.test_data)

        if self.save_path:
            self.train_val_dataset.build_and_save_vocab(save_path=self.save_path)
        elif self.load_path:
            self.train_val_dataset.load_vocab(load_path=self.load_path)
        else:
            raise ValueError("Either load_path or save_path should be specified.")
        self.collator = Collator(pad_token=self.train_val_dataset.vocabulary_to_index[self.train_val_dataset.pad_token])

    def prepare_data(self) -> None:
        """
        If a save path exists, saves the tokenized dataset, otherwise loads a tokenized dataset from a specified path.
        """
        if self.save_path:
            self.train_val_dataset.tokenize_data()
            self.train_val_dataset.save_tokenized_data(save_path=self.save_path)
        else:
            self.train_val_dataset.load_tokenized_data(load_path=self.load_path)
    
    def tokenize_test(self) -> None:
        """Utility method to tokenize the test data."""
        self.test_dataset.setup_vocab_params(self.train_val_dataset.vocab, self.train_val_dataset.special_tokens)
        self.test_dataset.tokenize_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the train, validation, and test datasets.

        Args:
            - stage (str, optional): A value in ['fit', 'test', 'predict']. Defaults to None.
        """
        train_len = int(0.8 * len(self.train_val_dataset))
        val_len = len(self.train_val_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, [train_len, val_len])

    def train_dataloader(self) -> DataLoader:
        """
        Builds and returns the train dataloader.

        Returns:
            - DataLoader: Train dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator.collate_fn)

    def val_dataloader(self) -> DataLoader:
        """
        Builds and returns the validation dataloader.

        Returns:
            - DataLoader: Validation dataloader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator.collate_fn)

    def test_dataloader(self) -> DataLoader:
        """
        Builds and returns the test dataloader.

        Returns:
            - DataLoader: Test dataloader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator.collate_fn)
