import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.pad_token = torch.tensor(
            [src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )
        self.sos_token = torch.tensor(
            [src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )

        self.causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(torch.bool) == 0
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        # transform the text to token
        encode_tokens = self.src_tokenizer.encode(src_text).ids
        decode_tokens = self.tgt_tokenizer.encode(tgt_text).ids
        if len(encode_tokens) > self.seq_len - 2:
            encode_tokens = encode_tokens[: self.seq_len - 2]
        if len(decode_tokens) > self.seq_len - 1:
            decode_tokens = decode_tokens[: self.seq_len - 1]

        num_encode_pad_tokens = self.seq_len - len(encode_tokens) - 2
        num_decode_pad_tokens = self.seq_len - len(decode_tokens) - 1

        encode_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encode_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full(
                    (num_encode_pad_tokens,), self.pad_token.item(), dtype=torch.int64
                ),
            ],
            dim=0,
        )
        decode_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decode_tokens, dtype=torch.int64),
                torch.full(
                    (num_decode_pad_tokens,), self.pad_token.item(), dtype=torch.int64
                ),
            ],
            dim=0,
        )
        label = torch.cat(
            [
                torch.tensor(decode_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full(
                    (num_decode_pad_tokens,), self.pad_token.item(), dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encode_input.size(0) == self.seq_len
        assert decode_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encode_input": encode_input,
            "decode_input": decode_input,
            "encode_mask": (encode_input != self.pad_token)
            .unsqueeze(0)
            .to(torch.int64),  # (1, seq)
            "decode_mask": (decode_input != self.pad_token).to(torch.int64)
            & self.causal_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
