import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type[nn.RNNBase] = nn.RNN, rnn_layers: int = 1, device: str = "cpu"):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        
        self.dtype = torch.float32
        self.device = torch.device(device)

        self.tokenizer = dataset.sp_model       # вместо обращения к датасету
        self.embed_size = embed_size
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        # """
        # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        # Create necessary layers
        # """

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=self.tokenizer.pad_id(),
            device=self.device, dtype=self.dtype
        )
        self.rnn = rnn_type(
            input_size=self.embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            device=self.device, dtype=self.dtype
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=self.vocab_size,
            device=self.device, dtype=self.dtype
        )

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # """
        # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        # Convert indices to embeddings, pass them through recurrent layers
        # and apply output linear layer to obtain the logits
        # """

        embeddings = self.embedding(indices)
        packed = nn.utils.rnn.pack_padded_sequence(
            input=embeddings,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=packed_out,
            batch_first=True,
            padding_value=self.tokenizer.pad_id(),
            total_length=self.max_length
        )

        logits = self.linear(rnn_out)

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()

        # """
        # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        # Encode the prefix (do not forget the BOS token!),
        # pass it through the model to accumulate RNN hidden state and
        # generate new tokens sequentially, sampling from categorical distribution,
        # until EOS token or reaching self.max_length.
        # Do not forget to divide predicted logits by temperature before sampling
        # """

        indices = torch.full(
            size=(1, self.max_length),
            fill_value=self.tokenizer.pad_id(),
            dtype=torch.long, device=self.device
        )

        prefix_tokenized = self.tokenizer.Encode(
            input=prefix,
            add_bos=True,
            emit_unk_piece=True
        )
        indices[0, :len(prefix_tokenized)] = torch.tensor(
            data=prefix_tokenized,
            dtype=torch.long, device=self.device
        )

        hidden = None
        eos_pos = self.max_length - 1

        for i in range(len(prefix_tokenized), self.max_length):
            input_ids = indices[:, :i]              # (1, i)
            embeddings = self.embedding(input_ids)  # (1, i, embed_size)

            rnn_out, hidden = self.rnn(embeddings, hidden)
            last_output = rnn_out[:, -1, :]         # (1, hidden_size)

            logits = self.linear(last_output) / temp
            probs = torch.softmax(logits, dim=-1)

            next_token = int(torch.multinomial(probs, num_samples=1).item())
            indices[0, i] = next_token

            if next_token == self.tokenizer.eos_id():
                eos_pos = i
                break

        tokens = indices[0].tolist()
        tokens = tokens[:eos_pos + 1]

        return self.tokenizer.decode(tokens)
