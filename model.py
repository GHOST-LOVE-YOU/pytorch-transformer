import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from dataset import causal_mask


class Inputembddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)

        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (seq_len, d_model / 2)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, d_model, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = q.shape
        q = self.query(q)  # (B,T,hs)
        k = self.key(k)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))  # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(v)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        head_size = d_model // num_heads
        self.heads = nn.ModuleList([Head(d_model, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_block = FeedForwardBlock(d_model)
        self.residual_connection1 = ResidualConnection(d_model)
        self.residual_connection2 = ResidualConnection(d_model)

    def forward(self, x, src_mask):
        x = self.residual_connection1(
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection2(x, self.feed_forward_block)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, num_heads)
        self.cross_attention_block = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_block = FeedForwardBlock(d_model)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model) for _ in range(3)]
        )

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_out, encoder_out, src_mask),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Transformer(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size, seq_len, d_model, N=6, num_heads=8
    ):
        super().__init__()
        self.src_embed = Inputembddings(d_model, src_vocab_size)
        self.tgt_embed = Inputembddings(d_model, tgt_vocab_size)
        # src and tgt use a single positional encode
        self.positional_encoding = PositionalEncoding(d_model, seq_len)

        # Create the encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads) for _ in range(N)]
        )

        # Create the decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads) for _ in range(N)]
        )

        # layer norm
        self.ec_norm = LayerNormalization(d_model)
        self.dc_norm = LayerNormalization(d_model)

        # pro layer
        self.projection_layer = nn.Linear(d_model, tgt_vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, x, src_mask):
        x = self.src_embed(x)
        x = self.positional_encoding(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        return self.ec_norm(x)

    def decode(self, x, encoder_out, src_mask, tgt_mask):
        x = self.tgt_embed(x)
        x = self.positional_encoding(x)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_out, src_mask, tgt_mask)
        return self.dc_norm(x)

    def forward(self, src, tgt, src_mask, tgt_mask, target=None):
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, src_mask, tgt_mask)
        logits = self.projection_layer(decoder_out)
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target, ignore_index=0)
        else:
            loss = None
        return logits, loss

    def generate(self, src, src_mask, max_len, tgt_tokenizer, device):
        memory = self.encode(src, src_mask)
        memory = memory.to(device)

        decoder_input = (
            torch.ones(1, 1)
            .fill_(tgt_tokenizer.token_to_id("[SOS]"))
            .type(torch.long)
            .to(device)
        )
        for i in range(max_len - 1):
            decoder_mask = causal_mask(i + 1).type(torch.int).to(device)
            out = self.decode(decoder_input, memory, src_mask, decoder_mask)
            prob = self.projection_layer(out[:, i])
            prob = F.softmax(prob, dim=-1)  # (B, C)

            next_word = torch.multinomial(prob, num_samples=1)

            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.ones(1, 1).type_as(src.data).fill_(next_word.item()),
                ],
                dim=1,
            )
            if next_word == tgt_tokenizer.token_to_id("[EOS]"):
                break
        return decoder_input.view(-1)
