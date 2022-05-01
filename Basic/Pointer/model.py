from typing import List, Tuple

import torch
from torch import nn


class TokenEmbedder(nn.Module):
    INIT_NORMALIZE_DENOMINATOR = (
        1000
    )  # Same order of magnitude that the width in pixels

    def __init__(self, n_characters: int, embedding_dim: int, position_dim: int):
        super().__init__()
        self.chars_embedding_layer = nn.Embedding(
            num_embeddings=n_characters, embedding_dim=embedding_dim, padding_idx=0
        )

        self.word_embedding_layer = nn.LSTM(
            input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1
        )

        self.dropout_word_embedding = nn.Dropout(0.5)

        self.normalize_position = nn.Parameter(
            self.INIT_NORMALIZE_DENOMINATOR * torch.ones(4), requires_grad=True
        )
        self.position_embedding_layer = nn.Linear(4, position_dim)
        self.dropout_position = nn.Dropout(0.5)

    def forward(self, words: torch.tensor, positions: torch.tensor) -> torch.tensor:
        character_embeddings = self.chars_embedding_layer(words)

        batch_size, n_words, n_characters, dim = character_embeddings.shape
        character_embeddings = character_embeddings.reshape(
            batch_size * n_words, n_characters, dim
        ).transpose(0, 1)

        _, (hn, _) = self.word_embedding_layer(character_embeddings)

        word_embeddings = self.dropout_word_embedding(hn.transpose(0, 1))
        word_embeddings = word_embeddings.reshape(batch_size, n_words, dim)

        positions = positions / self.normalize_position
        positions_embedding = torch.sigmoid(self.position_embedding_layer(positions))
        positions_embedding = self.dropout_position(positions_embedding)

        raw_token_embeddings = torch.cat([positions_embedding, word_embeddings], dim=2)
        return raw_token_embeddings


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, position_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=embedding_dim + position_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, raw_token_embeddings: torch.tensor):
        batch_size, _, _ = raw_token_embeddings.shape
        token_embeddings, (encoded_document, _) = self.lstm(
            raw_token_embeddings.transpose(0, 1)
        )
        token_embeddings = self.dropout(token_embeddings.transpose(0, 1))
        encoded_document = self.dropout(
            encoded_document.reshape(batch_size, 2 * self.embedding_dim)
        )
        return token_embeddings, encoded_document


class PointerNetwork(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.lstm_cell = nn.LSTMCell(
            input_size=2 * embedding_dim, hidden_size=embedding_dim
        )
        self.dropout = nn.Dropout(0.5)

        self.reference = nn.Linear(embedding_dim, 1)
        self.decoder_weights = nn.Linear(embedding_dim, embedding_dim)
        self.encoder_weights = nn.Linear(2 * embedding_dim, embedding_dim)

    def attention(
        self, token_embeddings: torch.tensor, hx: torch.tensor
    ) -> torch.tensor:
        batch_size, n_tokens, hidden_dim = token_embeddings.shape

        decoder_query = self.decoder_weights(hx)
        token_embeddings = self.encoder_weights(token_embeddings)

        decoder_query = decoder_query.repeat(
            n_tokens, 1, 1
        )  # n_token x batch_size x embedding_dim
        decoder_query = decoder_query.transpose(
            0, 1
        )  # batch_size x n_token x embedding_dim
        comparison = torch.tanh(decoder_query + token_embeddings)
        probabilities = torch.log_softmax(
            self.reference(comparison).reshape(batch_size, n_tokens), 1
        )
        return probabilities

    def forward(
        self, token_embeddings: torch.tensor, encoded_document: torch.tensor
    ) -> Tuple[torch.tensor, List[int]]:
        batch_size, _, _ = token_embeddings.shape
        overall_probabilities = []
        batch_identifier = torch.arange(batch_size).type(torch.LongTensor)

        peak_indices = []
        decoder_input = encoded_document
        for step in range(self.max_seq_len):
            hx, cx = self.lstm_cell(decoder_input)
            hx = self.dropout(hx)
            probabilities = self.attention(token_embeddings, hx)

            _, peak_idx = probabilities.max(dim=1)
            decoder_input = token_embeddings[batch_identifier, peak_idx, :]

            overall_probabilities.append(probabilities)
            peak_indices.append(peak_idx)

        overall_probabilities = torch.stack(overall_probabilities).transpose(0, 1)
        peak_indices = torch.stack(peak_indices).t()
        return overall_probabilities, peak_indices


class Model(nn.Module):
    def __init__(
        self, n_characters: int, embedding_dim: int, position_dim: int, max_seq_len: int
    ):
        super().__init__()

        self.token_embedder = TokenEmbedder(n_characters, embedding_dim, position_dim)
        self.encoder = Encoder(embedding_dim, position_dim)
        self.pointer_network = PointerNetwork(embedding_dim, max_seq_len)

    def forward(self, words: torch.tensor, positions: torch.tensor) -> torch.tensor:
        raw_token_embeddings = self.token_embedder(words, positions)
        token_embeddings, encoded_document = self.encoder(raw_token_embeddings)
        overall_probabilities, peak_indices = self.pointer_network(
            token_embeddings, encoded_document
        )

        return overall_probabilities, peak_indices
