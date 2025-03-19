import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbdding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embd = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embd(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float64).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float64)
            / (-d_model)
            * math.log(10000)
        )  # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class Head(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        q = self.query(q)  # (1, 1, 64)
        k = self.key(k)  # (1, 200, 64)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(v)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.heads = nn.ModuleList(
            [Head(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.pro = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return self.dropout(out)


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        # x(batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.linear_2(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.fw = FeedForward(d_model)
        self.residual_connect_1 = ResidualConnection(d_model)
        self.residual_connect_2 = ResidualConnection(d_model)

    def forward(self, x, src_mask):
        x = self.residual_connect_1(x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connect_2(x, self.fw)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.fw = FeedForward(d_model)
        self.residual_connects = nn.ModuleList(
            [ResidualConnection(d_model) for _ in range(3)]
        )

    def forward(self, x, encode_output, src_mask, tgt_mask):
        x = self.residual_connects[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        x = self.residual_connects[1](
            x,
            lambda x: self.cross_attention(x, encode_output, encode_output, src_mask),
        )
        x = self.residual_connects[2](x, self.fw)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size=30000,
        tgt_vocab_size=30000,
        seq_len=200,
        d_model=512,
        N=6,
        n_heads=8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.src_embd = InputEmbdding(src_vocab_size, d_model)
        self.tgt_embd = InputEmbdding(tgt_vocab_size, d_model)
        self.positional = PositionalEncoding(seq_len, d_model)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads) for _ in range(N)]
        )
        self.encoder_norm = LayerNormalization(d_model)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads) for _ in range(N)]
        )
        self.decoder_norm = LayerNormalization(d_model)

        self.projection_layer = nn.Linear(d_model, tgt_vocab_size)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        else:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src_tokens, src_mask):
        x = self.src_embd(src_tokens)  # (Batch, seq_len, d_model)
        x = self.positional(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, x, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embd(x)
        x = self.positional(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src_tokens, tgt_tokens, src_mask, tgt_mask, target=None):
        encoder_out = self.encode(src_tokens, src_mask)
        decoder_out = self.decode(tgt_tokens, encoder_out, src_mask, tgt_mask)
        logits = self.projection_layer(decoder_out)
        logits = F.softmax(logits, dim=-1)

        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target, ignore_index=0)
        else:
            loss = None

        return logits, loss

    def generate(self, src_tokens, src_mask, tgt_tokenizer, device):
        memory = self.encode(src_tokens, src_mask)
        decode_input = (
            torch.ones(1, 1)
            .fill_(tgt_tokenizer.token_to_id("[PAD]"))
            .type(torch.int64)
            .to(device)
        )

        for i in range(self.seq_len - 1):
            decode_mask = (
                torch.triu(torch.ones(i + 1, i + 1), diagonal=1).to(torch.bool) == 0
            )
            out = self.decode(decode_input, memory, src_mask, decode_mask)
            logits = self.projection_layer(out[:, i])
            logits = F.softmax(logits)

            nextword = torch.multinomial(logits, 1)

            decode_input = torch.cat(
                [
                    decode_input,
                    torch.ones(1, 1)
                    .fill_(nextword.item())
                    .type(torch.int64)
                    .to(device),
                ],
                dim=1,
            )
            if nextword.item() == tgt_tokenizer.token_to_id("[EOS]"):
                break
        return decode_input.view(-1)
