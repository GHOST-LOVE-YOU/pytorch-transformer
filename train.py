from dataset import BilingualDataset
from model import Transformer

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings

warnings.filterwarnings("ignore")

config = {
    "batch_size": 64,
    "max_iters": 10000,
    "eval_interval": 2000,
    "lr": 10**-4,
    "seq_len": 200,
    "d_model": 512,
    "datasource": "Helsinki-NLP/opus-100",
    "lang_src": "en",
    "lang_tgt": "zh",
    "cache_dir": "./data",
    "tokenizer_file": "tokenizer_{0}.json",
    "debug": False,
}


# ---------- check device ----------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.has_mps or torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)
if device == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(
        f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3} GB"
    )
else:
    print("NOTE: If you have a GPU, consider using it for training.")
device = torch.device(device)


# ---------- create dataseter ----------
train_dataset = load_dataset(
    f"{config['datasource']}",
    f"{config['lang_src']}-{config['lang_tgt']}",
    split="train",
    cache_dir=config["cache_dir"],
)
val_dataset = load_dataset(
    f"{config['datasource']}",
    f"{config['lang_src']}-{config['lang_tgt']}",
    split="validation",
    cache_dir=config["cache_dir"],
)
test_dataset = load_dataset(
    f"{config['datasource']}",
    f"{config['lang_src']}-{config['lang_tgt']}",
    split="test",
    cache_dir=config["cache_dir"],
)


# train tokenizer
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )

        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.train_from_iterator(ds, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading tokenizer for {lang}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def extract_text(ds, lang):
    return [item["translation"][lang] for item in ds]


train_src_texts = extract_text(train_dataset, config["lang_src"])
train_tgt_texts = extract_text(train_dataset, config["lang_tgt"])

src_tokenizer = get_or_build_tokenizer(config, train_src_texts, config["lang_src"])
tgt_tokenizer = get_or_build_tokenizer(config, train_tgt_texts, config["lang_tgt"])

train_ds = BilingualDataset(
    train_dataset,
    src_tokenizer,
    tgt_tokenizer,
    config["lang_src"],
    config["lang_tgt"],
    config["seq_len"],
)
val_ds = BilingualDataset(
    val_dataset,
    src_tokenizer,
    tgt_tokenizer,
    config["lang_src"],
    config["lang_tgt"],
    config["seq_len"],
)
test_ds = BilingualDataset(
    test_dataset,
    src_tokenizer,
    tgt_tokenizer,
    config["lang_src"],
    config["lang_tgt"],
    config["seq_len"],
)

train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)


# ---------- create model ----------
model = Transformer(
    src_tokenizer.get_vocab_size(),
    tgt_tokenizer.get_vocab_size(),
    config["seq_len"],
    config["d_model"],
)
m = model.to(device)

print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(m.parameters(), lr=config["lr"])


# ---------- train ----------
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    batch_iterator = tqdm(val_dataloader, desc="Processing eval")
    for batch in batch_iterator:
        encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
        decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
        encoder_mask = batch["encoder_mask"].to(device)  # (B, 1,
        decoder_mask = batch["decoder_mask"].to(device)  # (B,
        label = batch["label"].to(device)  # (B, seq_len)

        # Compute the loss using a simple cross entropy
        logits, loss = model(
            encoder_input, decoder_input, encoder_mask, decoder_mask, label
        )
        losses.append(loss.item())
        if config["debug"]:
            break
    return sum(losses) / len(losses)


@torch.no_grad()
def show_test():
    model.eval()
    count = 0

    for batch in test_dataloader:
        count += 1
        if count > 5:
            break
        encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
        encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, seq_len)

        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = model.generate(
            encoder_input, encoder_mask, config["seq_len"], tgt_tokenizer, device
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

        print(f"---------- {count} ----------")
        print(f"source_text: {source_text}")
        print(f"target_text: {target_text}")
        print(f"model_out_text: {model_out_text}")
        print("")


torch.cuda.empty_cache()
model.train()
batch_iterator = tqdm(range(config["max_iters"]))
for batch_iter in batch_iterator:
    batch = next(iter(train_dataloader))
    encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
    decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
    encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
    decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)
    label = batch["label"].to(device)  # (B, seq_len)

    # Compute the loss using a simple cross entropy
    logits, loss = model(
        encoder_input, decoder_input, encoder_mask, decoder_mask, label
    )
    batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

    # Backpropagate the loss
    loss.backward()

    # Update the weights
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if (
        batch_iter % config["eval_interval"] == 0 and batch_iter != 0
    ) or batch_iter == config["max_iters"] - 1:
        loss = estimate_loss()
        print(f"step {batch_iter}: val loss {loss:.4f}")

show_test()
