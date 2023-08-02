import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            embed_size: int = 512,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dropout: float = 0.1,
            max_seq_len: int = 100,
            batch_first: bool = True,
            device='cpu',

    ) -> None:
        super(Transformer, self).__init__()

        self.device = device

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

        self.src_position_embedding = nn.Embedding(max_seq_len, embed_size)
        self.tgt_posiion_embedding = nn.Embedding(max_seq_len, embed_size)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=batch_first
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_index = src_pad_idx
        self.batch_first = batch_first

    def make_src_mask(self, src):
        if not self.batch_first:
            src = src.transpose(0, 1)
        src_mask = src == self.pad_index
        return src_mask

    def forward(self, src, tgt):
        print(src.shape, tgt.shape)
        if self.batch_first:
            N, src_seq_len = src.shape
            _, tgt_seq_len = tgt.shape
        else:
            src_seq_len, N = src.shape
            tgt_seq_len, _ = tgt.shape
        print(N, src_seq_len)
        # src_positions = (
        #     torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N)
        # ).to(self.device)
        # tgt_positions = (
        #     torch.arange(0, tgt_seq_len).unsqueeze(1).expand(tgt_seq_len, N)
        # ).to(self.device)

        if self.batch_first:
            src_positions = torch.arange(0, src_seq_len).unsqueeze(
                0).expand(N, src_seq_len).to(self.device)
            tgt_positions = torch.arange(0, tgt_seq_len).unsqueeze(
                0).expand(N, tgt_seq_len).to(self.device)
        else:
            src_positions = torch.arange(0, src_seq_len).unsqueeze(
                1).expand(src_seq_len, N).to(self.device)
            tgt_positions = torch.arange(0, tgt_seq_len).unsqueeze(
                1).expand(tgt_seq_len, N).to(self.device)

        src_embedding = self.dropout(self.src_embedding(
            src) + self.src_position_embedding(src_positions))
        tgt_embedding = self.dropout(self.tgt_embedding(
            tgt) + self.tgt_posiion_embedding(tgt_positions))

        src_padding_mask = self.make_src_mask(src)
        tgt_padding_mask = self.transformer.generate_square_subsequent_mask(
            tgt_seq_len).to(self.device)

        output = self.transformer(
            src=src_embedding,
            tgt=tgt_embedding,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_padding_mask
        )

        return self.fc_out(output)
