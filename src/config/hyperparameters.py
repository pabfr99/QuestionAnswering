from typing import List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperParams():
    def __init__(self,
                batch_size: int = 32,
                embed_size: int = 512,
                hidden_size: int = 1024,
                sequence_length: int = 200,
                num_layers: int = 6,
                num_heads: int = 8,
                dropout_prob: float = 0.1,
                **kwargs: Any) -> None:
        """
        Initialize the HyperParams module with specified and arbitrary keyword arguments.

        Args:
            - batch_size (int): The batch size, defaults to 32.
            - embed_size (int): The size of the embeddings, defaults to 512.
            - hidden_size (int): The size of the hidden layers, defaults to 2048.
            - sequence_length (int): The maximum length of the sequence, defaults to 200.
            - num_layers (int): The number of layers, defaults to 6.
            - num_heads (int): The number of attention heads, defaults to 8.
            - dropout_prob (float): The dropout probability, defaults to 0.1.
            - **kwargs: Additional keyword arguments representing the parameters of the model.
        """
        # checks for the int arguments
        int_params = [batch_size, embed_size, hidden_size, sequence_length, num_layers, num_heads]
        int_param_names = ['batch_size', 'embed_size', 'hidden_size', 'sequence_length', 'num_layers', 'num_heads']
        for param, name in zip(int_params, int_param_names):
            if not isinstance(param, int):
                raise TypeError(f"{name} must be an integer.")
            if param <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        # checks for the dropout argument
        if not isinstance(dropout_prob, float):
            raise TypeError('dropout_prob must be a float.')
        if not 0 <= dropout_prob < 1:
            raise ValueError("dropout_prob must be between 0 and 1.")
        # check for transformer hyper parameters
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads.")

        # define the vocabulary containing the parameters
        self._properties = {
              'batch_size': batch_size,
              'embed_size': embed_size,
              'hidden_size': hidden_size,
              'sequence_length': sequence_length,
              'num_layers': num_layers,
              'num_heads': num_heads,
              'dropout_prob': dropout_prob
          }
        self._properties.update(kwargs)

        logger.info(f'Initialized HyperParams with properties: {self._properties}')

    def __getattr__(self,
                    name: str) -> Any:
        """
        Custom access method to attributes.

        Args:
            - name (str): the name of the attribute to access.

        Returns:
            - Any: the value of the attribute.
        """
        if name in self._properties:
            return self._properties[name]
        raise AttributeError(f'"HyperParams" object has no attribute "{name}"')

    def __setattr__(self,
                    name: str,
                    value: Any) -> None:
        """
        Custom set method of attributes.

        Args:
            - name (str): the name of the attribute to set.
            - value (Any): the value of the attribute to set.
        """
        if name == '_properties':
            super().__setattr__(name, value)
        elif name in self._properties:
            raise AttributeError(f'"{name}" is a read-only attribute')
        else:
            raise AttributeError(f'"HyperParams" object has no attribute "{name}"')

    def to_dict(self) -> dict:
        """Return the properties as a dictionary."""
        return self._properties

    def keys(self) -> List[str]:
        """Return a list of keys corresponding to the properties."""
        return self._properties.keys()

    def __repr__(self) -> str:
        """Display the object properties in a nice and readable way."""
        items = ('{}={!r}'.format(k, v) for k, v in self._properties.items())
        return '{}:\n\t{}'.format(type(self).__name__, '\n\t'.join(items))

