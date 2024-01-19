import torch
import math
from torch import Tensor
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim
import torch.utils.data
from typing import Optional, Tuple, Dict
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from src.config.hyperparameters import HyperParams


class ScaledDotProductAttention(LightningModule):
    def __init__(self,
                 dropout: float = 0.1) -> None:
        """
        Initialize the ScaledDotProductAttention module.

        Args:
            - dropout (float, optional): Dropout rate, default to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Implements the forward pass of the scaled dot product attention.

        Args:
            q (Tensor): Query tensor of shape (batch_size, num_heads, sequence_length, head_dim)
            k (Tensor): Key tensor of shape (batch_size, num_heads, sequence_length, head_dim)
            v (Tensor): Value tensor of shape (batch_size, num_heads, sequence_length, head_dim)
            mask (Tensor, optional): Mask tensor of shape (batch_size, 1, sequence_length, sequence_length) for preventing attention to certain positions.

        Returns:
            output (Tensor): Output tensor after tensor-product attention of shape (batch_size, num_heads, sequence_length, head_dim)
            attn_weights (Tensor): Attention weights tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
        """
        # get the size of the key
        d_k = q.size(-1)
        # compute attention logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask to the attention logits
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))


        # compute attention weights and apply dropout
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # compute the output tensor
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
        
        
class MultiHeadAttention(LightningModule):
    def __init__(self,
                 params: HyperParams) -> None:
        """
        Initialize the MultiHeadSelfAttention module.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                num_heads: The number of attention heads.
                dropout_prob: The dropout probability.
                batch_size: The batch size.
                sequence_length: The maximum length of the input sequence
        """
        super().__init__()
        assert params.embed_size % params.num_heads == 0, 'Embedding size needs to be divisible by the number of heads'
        # set the attributes with the required parameters
        self.params = params
        self.embed_size = params.embed_size
        self.num_heads = params.num_heads
        self.head_dim = params.embed_size // params.num_heads
        # define linear layers for query, key and value
        self.q_linear = nn.Linear(params.embed_size, params.embed_size, bias=True)
        self.k_linear = nn.Linear(params.embed_size, params.embed_size, bias=True)
        self.v_linear = nn.Linear(params.embed_size, params.embed_size, bias=True)
        # define dropout layer, final linear layer and the dot product attention
        self.dropout = nn.Dropout(p = params.dropout_prob)
        self.out_linear = nn.Linear(params.embed_size, params.embed_size)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout = params.dropout_prob)

    def forward(self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        """
        Implements the forward pass of the traditional Multi-Head Attention.

        Args:
            q (Tensor): Input queries tensor of shape (batch_size, sequence_length, embed_size)
            k (Tensor): Input keys tensor of shape (batch_size, sequence_length, embed_size)
            v (Tensor): Input values tensor of shape (batch_size, sequence_length, embed_size)
            mask (Tensor, optional): Mask tensor of shape (batch_size, 1, sequence_length, sequence_length) for preventing attention to certain positions.

        Returns:
            output (Tensor): Output tensor after multi-head attention of shape (batch_size, sequence_length, embed_size)
            attn_weights (Tensor): Attention weights tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
        """
        batch_size = q.size(0)
        # apply linear layers
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        # reshape and transpose tensors to split by the number of heads
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # compute attention weights and output using scaled dot product attention
        output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # concatenate and pass through the final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        output = self.out_linear(output)
        return output, attn_weights


class PositionalEncoding(LightningModule):
    def __init__(self,
                 embed_size: int = 512,
                 sequence_length: int = 200,
                 dropout: float = 0.1) -> None:
        """
        Initialize the PositionalEncoding module.

        Args:
            - embed_size (int, optional): The size of the input embeddings, default to 512.
            - sequence_length (int, optional): The maximum length of the input sequence, default to 200.
            - dropout (float, optional): Dropout rate, default to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(sequence_length, embed_size)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        # get the values used as divisors in the positional embedding calculation
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        # assign the values to the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x) -> Tensor:
        """
        Implements the forward pass of the positional encoding.

        Args:
            - x (Tensor): Input tensor of shape (batch_size, sequence_length, embed_size).

        Returns:
            - output (Tensor): Output tensor adding positional encoding and dropout.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        
        
class EncoderLayer(LightningModule):
    def __init__(self,
                 params: HyperParams) -> None:
        """
        Initialize the EncoderLayer module.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                hidden_size: The dimension of the feed forward network model.
                dropout_prob: The dropout rate.
        """
        super().__init__()
        self.params = params
        self.attn = MultiHeadAttention(params)
        self.norm1 = nn.LayerNorm(params.embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(params.embed_size, params.hidden_size),
            nn.Dropout(params.dropout_prob),
            nn.ReLU(),
            nn.Linear(params.hidden_size, params.embed_size)
            )
        self.norm2 = nn.LayerNorm(params.embed_size)
        self.dropout = nn.Dropout(params.dropout_prob)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Implements the forward pass of the encoder layer.

        Args:
            - src (Tensor): Source tensor of shape (batch_size, sequence_length, embed_size).
            - src_mask (Optional[Tensor]): Source mask tensor of shape (batch_size, 1, 1, sequence_length).

        Returns:
            - src (Tensor): Transformed tensor of shape (batch_size, sequence_length, embed_size).
        """
        # multi-head attention
        attn_output, _ = self.attn(src, src, src, src_mask)
        # residual connection, layer normalization and dropout
        src = src + self.dropout(self.norm1(attn_output))
        # feed-forward network
        ffn_output = self.ffn(src)
        # residual connection, second layer normalization and dropout
        src = src + self.dropout(self.norm2(ffn_output))
        return src


class Encoder(LightningModule):
    def __init__(self,
                 params: HyperParams) -> None:
        """
        Initialize the Encoder module.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                num_layers: The number of encoder layers.
        """
        super().__init__()
        # generate the required layers
        self.layers = self._get_layers(params)
        self.norm = nn.LayerNorm(params.embed_size)

    @staticmethod
    def _get_layers(params: HyperParams):
        """
        Build the encoder taking all the layers needed.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                num_layers: The number of layers.
        Returns:
            - ModuleList: A ModuleList containing instances of EncoderLayer.
        """
        return nn.ModuleList([EncoderLayer(params) for _ in range(params.num_layers)])

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Implements the forward pass of the encoder.

        Args:
            - src (Tensor): Source tensor of shape (batch_size, sequence_length, embed_size).
            - src_mask (Optional[Tensor]): Source mask tensor of shape (batch_size, 1, 1, sequence_length).

        Returns:
            - Tensor: Transformed tensor of shape (batch_size, sequence_length, embed_size).
        """
        # iterate through the layers
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class DecoderLayer(LightningModule):
    def __init__(self, params: HyperParams) -> None:
        """
        Initialize the DecoderLayer module.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                hidden_size: The dimension of the feed forward network model.
                dropout_prob: The dropout rate.
        """
        super().__init__()
        self.params = params
        # self-attention
        self.self_attn = MultiHeadAttention(params)
        self.norm1 = nn.LayerNorm(params.embed_size)
        # cross-attention
        self.cross_attn = MultiHeadAttention(params)
        self.norm2 = nn.LayerNorm(params.embed_size)
        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(params.embed_size, params.hidden_size),
            nn.Dropout(params.dropout_prob),
            nn.ReLU(),
            nn.Linear(params.hidden_size, params.embed_size)
            )
        self.norm3 = nn.LayerNorm(params.embed_size)
        self.dropout = nn.Dropout(params.dropout_prob)

    def forward(self,
                trg: Tensor,
                memory: Tensor,
                trg_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Implements the forward pass of the decoder layer.

        Args:
            - trg (Tensor): Target tensor of shape (batch_size, sequence_length, embed_size).
            - memory (Tensor): Memory tensor from the encoder of shape (batch_size, sequence_length, embed_size).
            - trg_mask (Optional[Tensor]): Target mask tensor of shape (batch_size, 1, sequence_length, sequence_length).
            - memory_mask (Optional[Tensor]): Memory mask tensor of shape (batch_size, 1, 1, sequence_length).

        Returns:
            - trg (Tensor): Transformed tensor of shape (batch_size, sequence_length, embed_size).
        """
        # self-attention
        self_attn_output, _ = self.self_attn(trg, trg, trg, trg_mask)
        # residual connection, layer normalization and dropout
        trg = trg + self.dropout(self.norm1(self_attn_output))

        # cross-attention
        cross_attn_output, _ = self.cross_attn(trg, memory, memory, memory_mask)
        # residual connection, layer normalization and dropout
        trg = trg + self.dropout(self.norm2(cross_attn_output))
        # feed-forward network
        ffn_output = self.ffn(trg)
        # residual connection, third layer normalization and dropout
        trg = trg + self.dropout(self.norm3(ffn_output))
        return trg
        
        
class Decoder(LightningModule):
    def __init__(self,
                 params: HyperParams) -> None:
        """
        Initialize the Decoder module.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                num_layers: The number of decoder layers.
        """
        super().__init__()
        # generate the required layers
        self.layers = self._get_layers(params)
        self.norm = nn.LayerNorm(params.embed_size)

    @staticmethod
    def _get_layers(params: HyperParams):
        """
        Build the decoder with all the layers needed.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                num_layers: The number of layers.

        Returns:
            - ModuleList: A ModuleList containing instances of DecoderLayer.
        """
        return nn.ModuleList([DecoderLayer(params) for _ in range(params.num_layers)])

    def forward(self,
                trg: Tensor,
                src: Tensor,
                trg_mask: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Implements the forward pass of the decoder.

        Args:
            - trg (Tensor): Target tensor of shape (batch_size, sequence_length, embed_size).
            - src (Tensor): Source tensor of shape (batch_size, sequence_length, embed_size).
            - trg_mask (Optional[Tensor]): Target mask tensor of shape (batch_size, 1, sequence_length, sequence_length).
            - src_mask (Optional[Tensor]): Source mask tensor of shape (batch_size, 1, 1, sequence_length).

        Returns:
            - Tensor: Transformed tensor of shape (batch_size, sequence_length, embed_size).
        """
        # iterate through the layers
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return self.norm(trg)


class Transformer(LightningModule):
    def __init__(self,
                 params,
                 vocab) -> None:
        """
        Initialize the DecoderLayer module.

        Args:
            - params (HyperParams): An object that contains the following properties:
                embed_size: The size of the input embeddings.
                hidden_size: The dimension of the feed forward network model.
                dropout_prob: The dropout rate.
                learning_rate: The learning rate.
            -vocab: the vocabulary containing the mapping between the characters and their corresponding index.
        """
        super().__init__()
        self.vocab = vocab
        self.params = params
        self.save_hyperparameters(params.to_dict())
        self.start_token = vocab['<bos>']
        self.end_token = vocab['<eos>']
        self.pad_token = vocab['<pad>']
        self.vocab_size = len(vocab)
        self.embedding_layer = nn.Embedding(self.vocab_size, params.embed_size)
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        self.pos_encoding = PositionalEncoding(params.embed_size, params.sequence_length, params.dropout_prob)
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.fc_out = nn.Linear(params.embed_size, self.vocab_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor=None,
                trg_mask: Tensor=None) -> Tensor:
        """
        Implements the forward pass of the transformer.

        Args:
            - trg (Tensor): Target tensor of shape (batch_size, sequence_length, embed_size).
            - src (Tensor): Source tensor of shape (batch_size, sequence_length, embed_size).
            - trg_mask (Optional[Tensor]): Target mask tensor of shape (batch_size, 1, sequence_length, sequence_length).
            - src_mask (Optional[Tensor]): Source mask mask tensor of shape (batch_size, 1, 1 sequence_length).

        Returns:
            - output (Tensor): Tensor of shape (batch_size, sequence_length, vocabulary_size)
        """
        # embedding layer and positional encoding
        src = self.pos_encoding(self.embedding_layer(src))
        trg = self.pos_encoding(self.embedding_layer(trg))
        # encoder, decoder and fully connected output
        memory = self.encoder(src, src_mask)
        output = self.decoder(trg, memory, trg_mask, src_mask)
        return self.fc_out(output)

    def training_step(self,
                      batch: Dict[str, Tensor],
                      batch_idx) -> Dict[str, Tensor]:
        """
        Implements the training step of the transformer.

        Args:
            - batch (Dict[str, Tensor]): Dictionary containing the current (questions, answers, questions_mask, answers_mask).

        Returns:
            - loss (Dict[str, Tensor]): The dictionary that stores the loss of the current batch.
        """
        # extract the data from the batch
        questions = batch['questions']
        answers = batch['answers']
        questions_mask = batch['questions_mask']
        answers_mask = batch['answers_mask']
        # model forward method
        logits = self(questions, answers[:, :-1], questions_mask, answers_mask[:, :, :-1, :-1])
        # flattening the logits, the targets and computing the loss
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_trg = answers[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(flat_logits, flat_trg, ignore_index=self.pad_token)
        self.log('train_loss', loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        return {'loss': loss}

    def validation_step(self,
                        batch: Dict[str, Tensor],
                        batch_idx) -> Dict[str, Tensor]:
        """
        Implements the validation step of the transformer.

        Args:
            - batch (Dict[str, Tensor]): Dictionary containing the current (questions, answers, questions_mask, answers_mask)

        Returns:
            - loss (Dict[str, Tensor]): The dictionary that stores the loss of the current batch
        """
        # extract the data from the batch
        questions = batch['questions']
        answers = batch['answers']
        questions_mask = batch['questions_mask']
        answers_mask = batch['answers_mask']
        # model forward method
        logits = self(questions, answers[:, :-1], questions_mask, answers_mask[:, :, :-1, :-1])
        # flattening the logits, the targets and computing the loss
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_trg = answers[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(flat_logits, flat_trg, ignore_index=self.pad_token)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def calc_string_acc(self,
                        trg: Tensor,
                        prediction: Tensor) -> bool:
        """
        Calculates if there is a match between the original answer and the target answer.

        Args:
            - trg (Tensor): Original answer tensor of shape (1, length)
            - prediction (Tensor): answer tensor predicted by the transformer of shape (1, length)

        Returns:
            - exact_match (bool): True if the two string match, False otherwise
        """
        trg = trg.squeeze(0)
        prediction = prediction.squeeze(0)
        exact_match = (torch.eq(trg, prediction) | (trg==self.pad_token)).all().item()
        return exact_match

    def evaluate_greedy(self,
                        test_iterator: torch.utils.data.DataLoader,
                        infer_percent: float) -> None:
        """
        Performs a greedy evaluation of the model, generating the answers character by character.

        Args:
            - test_iterator (torch.utils.data.DataLoader): DataLoader containing the samples to evaluate.
            - infer_percent (float): The percentage of samples to evaluate.
        """
        self.eval()
        infer_raw_match = 0
        infer_raw_count = 0
        infer_count = int(len(test_iterator)*infer_percent)

        with torch.no_grad():
          for idx, batch in enumerate(tqdm(test_iterator, total=infer_count, desc=f'Inference on batch {infer_count}')):
            # extract the data
            batch_src = batch['questions'].to(self.device)
            batch_src_mask = batch['questions_mask'].to(self.device)
            batch_trg = batch['answers'].to(self.device)

            if idx >= infer_count:
                break

            for i, (src, src_mask, trg) in enumerate(zip(batch_src, batch_src_mask, batch_trg)):
                max_len = len(trg)
                src = src.unsqueeze(0)
                src_mask = src_mask.unsqueeze(0)
                trg = trg.unsqueeze(0)
                # get the answer prediction
                prediction = self.forward_inference(src,
                                              src_mask,
                                              max_len=max_len)
                match = self.calc_string_acc(trg, prediction)
                infer_raw_match += match
                infer_raw_count += 1
                # print the predictions for the first element of the batches
                if i==0:
                    print('Question: ', ''.join(self.vocab.lookup_tokens(src.squeeze(0).tolist())), '\n')
                    print('Answer: ', ''.join(self.vocab.lookup_tokens(trg.squeeze(0).tolist())), '\n')
                    print('Predicted Answer: ', ''.join(self.vocab.lookup_tokens(prediction.squeeze(0).tolist())))
                    print('--------------------------------------------------------\n')
          # computing the accuracy
          greedy_acc = infer_raw_match / infer_raw_count
          print(f'Greedy Accuracy: {greedy_acc}')

    def create_trg_mask(self,
                        trg: Tensor) -> Tensor:

        """
        Performs the forward inference, by generating an answer given a question.

        Args:
            - trg (Tensor): Target tensor to mask of shape (1, current_length)
        Returns:
            - trg_mask (Tensor): Generated mask tensor of shape (1, 1, current_length, current_length)
        """
        # padding trg mask
        trg_pad_mask = (trg != self.pad_token).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        # lookahead target mask
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool().unsqueeze(0)
        trg_pad_mask = trg_pad_mask.expand(-1, 1, trg_len, trg_len)
        # combined target mask
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward_inference(self,
                          src: Tensor,
                          src_mask: Tensor,
                          max_len: int) -> Tensor:
        """
        Performs the forward inference, by generating an answer given a question.

        Args:
            - src (Tensor): Source tensor of shape (1, sequence_length)
            - src_mask (Tensor): Source mask tensor of shape (1, 1, 1, sequence_length)
            - max_len (Int): Integer value that defines the max length of the answer

        Returns:
            - trg (Tensor): Generated target tensor of shape (1, max_len)
        """

        self.eval()
        # embedding and positional encoding
        src_emb = self.embedding_layer(src)
        src_pos = self.pos_encoding(src_emb)
        # encoder
        memory = self.encoder(src_pos, src_mask)
        # generate the start token
        trg = torch.ones(src.shape[0], 1, device=self.device).fill_(self.start_token).type(src.dtype)

        for _ in range(max_len-1):
            # generate the next token
            trg_mask = self.create_trg_mask(trg)
            trg_emb = self.embedding_layer(trg)
            trg_pos = self.pos_encoding(trg_emb)
            output = self.decoder(trg=trg_pos, src=memory,
                                  trg_mask=trg_mask, src_mask=src_mask)
            logits = self.fc_out(output)
            pred = torch.argmax(logits[:,[-1],:], dim=-1)
            # concatenate with the previous generated tokens
            trg = torch.cat([trg, pred], dim=1)

        return trg

    def configure_optimizers(self) -> torch.optim.Optimizer:

        """
        Pytorch lightning method for optimizer handling.

        Returns:
            - optimizer (torch.optim.Optimizer): The optimizer to use.
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.params.learning_rate,
                                     betas=(0.9, 0.995))
        return optimizer

