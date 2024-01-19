import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly.express as px

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self,
                 data: pd.DataFrame) -> None:
        """
        Initialize the Visualizer module.

        Args:
            - data (pandas.DataFrame): A Pandas DataFrame containing the data to visualize.
        """
        # error checks for the method argument
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data must be a pandas DataFrame.')

        if data.empty:
            logger.warning('The input DataFrame is empty.')

        expected_columns = ['question', 'answer', 'folder', 'module']
        for col in expected_columns:
            if col not in data.columns:
                raise ValueError(f"DataFrame is missing the column '{col}'")

        self.data = data

        logger.info(f'Initializing the Visualizer.')

    def print_samples(self,
                      num_samples: int=5,
                      random: bool=False) -> None:
        """
        Prints samples from the DataFrame.

        Args:
            - num_samples (int): The number of samples to print.
            - random (bool): Whether to print random or the first samples.
        """
        # error check for the method arguments
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an integer.')

        if num_samples <= 0:
            raise ValueError('num_samples must be a positive integer.')

        if not isinstance(random, bool):
            raise TypeError('random must be a boolean.')

        logger.info(f'Preparing to print {num_samples} samples.')
        if random:
            logger.info(f'The samples will be selected randomly.')

        # extract the first num_samples samples or random samples
        samples = self.data.sample(num_samples) if random else self.data.head(num_samples)

        # for each sample print the question, the answer, the folder and the module
        for _, row in samples.iterrows():
            question = row['question']
            answer = row['answer']
            folder = row['folder']
            module = row['module']
            print('\n---------------------------------------------------------\n')
            print(f'Question: {question}\n'
                  f'Answer: {answer}\n'
                  f'Folder: {folder}\n'
                  f'Question Type: {module}')

    def display_length_statistics(self) -> None:
        """Displays statistics about question and answer lengths."""
        logger.info('Displaying length statistics.')

        # apply the length to the question and answer columns
        length_data = self.data[['question', 'answer']].applymap(len)

        # rename the columns and print the statistics
        length_data.rename(columns={'question': 'question_length', 'answer': 'answer_length'}, inplace=True)
        print(length_data.describe().round(1))

    def plot_length_distribution(self,
                                 column: str) -> plt.Axes:
        """
        Creates a countplot either of question or answer lengths.

        Args:
            - column (str): The column to plot, either 'question' or 'answer'.
        Returns:
            - ax (matplotlib.axes.Axes): The Axes object with the plot.
        """
        # error check for the column argument
        if column not in ['question', 'answer']:
            raise ValueError('Column must be either "question" or "answer"')

        logger.info(f'Plotting {column} length distribution.')

        fig, ax = plt.subplots(figsize=(8, 6))

        # get the length data from the column
        lengths = self.data[column].apply(len)
        max_length = lengths.max()

        # define a smooth color palette
        n_colors = len(lengths.unique())
        colors = sns.color_palette(palette = 'coolwarm', desat=0.9, n_colors=n_colors)

        # countplot of sentences lengths
        sns.countplot(x=lengths, palette=colors, ax=ax)
        ax.set_title(f'{column} length distribution', fontweight='bold')
        ax.set_xlabel('Length', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        plt.xticks(ticks=np.arange(0, max_length+1, step=10), labels=np.arange(1, max_length+2, step=10), fontsize=10)
        plt.tight_layout()
        return ax

    def boxplot_lengths(self,
                        column: str) -> plt.Axes:
        """
        Creates a boxplot either of question or answer lengths.

        Args:
            - column (str): The column to plot, either 'question' or 'answer'.

        Returns:
            - ax (matplotlib.axes.Axes): The Axes object with the boxplot.
        """
        # error check for the argument
        if column not in ['question', 'answer']:
            raise ValueError('Column must be either "question" or "answer"')

        logger.info(f'Creating boxplot for {column} lengths.')

        fig, ax = plt.subplots(figsize=(8, 6))

        # get the length data from the column
        self.data['length'] = self.data[column].apply(len)

        # boxplot of sentences lengths
        sns.boxplot(x='folder', y='length', data=self.data, ax=ax)
        ax.set_title(f'Boxplot of {column} lengths')
        return ax

    def create_interactive_plot(self,
                                plot_type: str,
                                column: str,
                                *args,
                                **kwargs) -> None:
        """
        Creates an interactive plot of a specified type.

        Args:
            - plot_type (str): Type of plot, either 'bar' or 'pie'.
            - column (str): The column to visualize, either 'folder' or 'module'.
        """
        # error check for the arguments
        if plot_type not in ['bar', 'pie']:
            raise ValueError('Plot type must be either "bar" or "pie".')
        if column not in self.data.columns:
            raise ValueError(f'Column {column} must exist in the DataFrame.')

        logger.info(f'Creating interactive {plot_type} plot for {column}.')

        # count the number of values for each unique value in column
        data = self.data[column].value_counts().reset_index()
        data.columns = [column, 'count']

        # create bar or pie interactive plot
        if plot_type == 'bar':
            fig = px.bar(data, x=column, y='count', *args, **kwargs)
        elif plot_type == 'pie':
            fig = px.pie(data, names=column, values='count', *args, **kwargs)
        fig.show()
