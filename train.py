from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import BilingualDataset
from model import Transformer

# ---------- config ----------
config = {"seq_len": 200, "batch_size": 64, "max_iters": 5000, "eval_iters": 2000}

# ---------- check device ----------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("using device:", device)
device = torch.device(device)
if device == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(
        f"Device memory: {torch.cuda.get_device_properties(device.index) / 1024 * 3} GB"
    )
else:
    print("NOTE: If you have a GPU, consider using it for training")

# ---------- create dataset ----------
train_dataset = load_dataset(
    "Helsinki-NLP/opus-100", "en-zh", split="train", cache_dir="./data"
)
val_dataset = load_dataset(
    "Helsinki-NLP/opus-100", "en-zh", split="validation", cache_dir="./data"
)
test_dataset = load_dataset(
    "Helsinki-NLP/opus-100", "en-zh", split="test", cache_dir="./data"
)


def get_or_build_tokenizer(text, lang):
    tokenizer_path = Path(f"./tokenizer_{lang}")
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(text, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading tokenizer for {lang}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


src_text = [item["translation"]["en"] for item in train_dataset]
tgt_text = [item["translation"]["zh"] for item in train_dataset]

src_tokenizer = get_or_build_tokenizer(src_text, "en")
tgt_tokenizer = get_or_build_tokenizer(tgt_text, "zh")


train_ds = BilingualDataset(
    train_dataset, src_tokenizer, tgt_tokenizer, "en", "zh", config["seq_len"]
)
val_ds = BilingualDataset(
    val_dataset, src_tokenizer, tgt_tokenizer, "en", "zh", config["seq_len"]
)
test_ds = BilingualDataset(
    test_dataset, src_tokenizer, tgt_tokenizer, "en", "zh", config["seq_len"]
)

train_dataloader = DataLoader(train_ds, config["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_ds, 1, shuffle=True)
test_dataloader = DataLoader(test_ds, 1, shuffle=True)


# ---------- create model ----------
model = Transformer()
model = model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")


def lr_lambda(current_step):
    warmup_steps = 4000
    d_model = 512

    if current_step < warmup_steps:
        return current_step / warmup_steps
    else:
        return (d_model**-0.5) * min(
            (current_step**-0.5), current_step * (warmup_steps**-1.5)
        )


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-9,
)
scheduler = LambdaLR(optimizer, lr_lambda)


# ---------- train ----------
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    batch_iterator = tqdm(val_dataloader, desc="Processing eval")
    for batch in batch_iterator:
        encode_input = batch["encode_input"].to(device)
        decode_input = batch["decode_input"].to(device)
        encode_mask = batch["encode_mask"].to(device)
        decode_mask = batch["decode_mask"].to(device)
        label = batch["label"].to(device)

        _, loss = model(encode_input, decode_input, encode_mask, decode_mask, label)
        losses.append(loss)
    return sum(losses) / len(losses)


@torch.no_grad()
def show_test():
    model.eval()
    count = 0

    for batch in test_dataloader:
        count += 1
        if count > 5:
            break
        encode_input = batch["encode_input"].to(device)
        encode_mask = batch["encode_mask"].to(device)

        model_out = model.generate(encode_input, encode_mask, tgt_tokenizer, device)

        src_text = batch["src_text"]
        tgt_text = batch["tgt_text"]
        model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

        print(f"# ---------- {count} ----------")
        print(f"source text: {src_text}")
        print(f"target text: {tgt_text}")
        print(f"model prediction text: {model_out_text}")
        print("")


torch.cuda.empty_cache()
model.train()
batch_iterator = tqdm(range(config["max_iters"]))
for batch_iter in batch_iterator:
    batch = next(iter(train_dataloader))
    encode_input = batch["encode_input"].to(device)
    decode_input = batch["decode_input"].to(device)
    encode_mask = batch["encode_mask"].to(device)
    decode_mask = batch["decode_mask"].to(device)
    label = batch["label"].to(device)

    logits, loss = model(encode_input, decode_input, encode_mask, decode_mask, label)
    batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (batch_iter % config["eval_iters"] == 0 and batch_iter != 0) or (
        batch_iter == config["max_iters"] - 1
    ):
        loss = estimate_loss()
        print(f"step {batch_iterator}: val loss {loss:6.4f}")

show_test()
