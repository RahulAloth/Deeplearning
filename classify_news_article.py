
# -----------------------------
# Task: Classify a news article into one of 4 categories:
# World
# Sports
# Business
# Sci/Tech
# Source: Provided by torchtext (automatic download on first run).  
# -----------------------------


import os
import random
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

set_seed(42)

# -----------------------------
# Config
# -----------------------------
class Config:
    batch_size = 128
    lr = 2e-3
    weight_decay = 1e-4
    num_epochs = 6
    embed_dim = 200
    hidden_size = 128
    num_layers = 1
    bidirectional = True
    min_freq = 2           # min token frequency for vocab
    max_vocab_size = 40000 # cap vocab to limit memory
    valid_ratio = 0.05
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_token = "<pad>"
    unk_token = "<unk>"

cfg = Config()

# -----------------------------
# 1) Load dataset
# -----------------------------
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for (label, text) in data_iter:
        yield tokenizer(text)

# Train/test split from AG_NEWS
train_iter = AG_NEWS(split='train')
test_iter  = AG_NEWS(split='test')

# IMPORTANT: train_iter is a generator; materialize to list so we can split/build vocab
train_list = list(train_iter)

# -----------------------------
# 2) Build vocabulary from training set
# -----------------------------
vocab = build_vocab_from_iterator(
    yield_tokens(train_list),
    min_freq=cfg.min_freq,
    specials=[cfg.pad_token, cfg.unk_token],
    max_tokens=cfg.max_vocab_size
)
vocab.set_default_index(vocab[cfg.unk_token])

pad_idx = vocab[cfg.pad_token]
unk_idx = vocab[cfg.unk_token]

num_classes = 4  # AG_NEWS has 4 classes

# -----------------------------
# 3) Prepare Dataset wrappers
# -----------------------------
class AGNewsDataset(Dataset):
    """
    Wraps (label, text) pairs.
    Labels in AG_NEWS are 1..4, we shift to 0..3.
    """
    def __init__(self, examples: List[Tuple[int, str]], vocab, tokenizer):
        self.examples = examples
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def encode_text(self, text: str) -> List[int]:
        tokens = self.tokenizer(text)
        return self.vocab(tokens)

    def __getitem__(self, idx):
        label, text = self.examples[idx]
        label = label - 1  # to 0-based
        ids = self.encode_text(text)
        return torch.tensor(label, dtype=torch.long), torch.tensor(ids, dtype=torch.long)

full_train_ds = AGNewsDataset(train_list, vocab, tokenizer)
test_ds       = AGNewsDataset(list(test_iter), vocab, tokenizer)

# Create train/valid split
n_total = len(full_train_ds)
n_valid = int(cfg.valid_ratio * n_total)
n_train = n_total - n_valid
train_ds, valid_ds = random_split(full_train_ds, [n_train, n_valid], generator=torch.Generator().manual_seed(42))

# -----------------------------
# 4) Collate function (padding per-batch)
# -----------------------------
def collate_fn(batch):
    """
    batch: list of (label_tensor, token_ids_tensor)
    Returns:
      padded_ids: (B, T)
      lengths:    (B,)
      labels:     (B,)
    """
    labels, sequences = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = max(lengths).item()

    padded = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    labels = torch.stack(labels)
    return padded, lengths, labels

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 5) Model: BiLSTM classifier
# -----------------------------
class BiLSTMClassifier(nn.Module):
    """
    Embedding -> BiLSTM -> Concatenate final hidden states -> Dropout -> Linear
    Using pack_padded_sequence to handle variable-length batches.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int,
                 bidirectional: bool, num_classes: int, pad_idx: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        # Initialize embeddings (xavier uniform is common for layers; for embeddings normal works well)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0.0)

    def forward(self, padded_ids, lengths):
        # padded_ids: (B, T)
        embedded = self.embedding(padded_ids)  # (B, T, E)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, hidden_size)
        # take the final layer's hidden states
        if self.num_directions == 2:
            h_fwd = h_n[-2,:,:]  # (B, H)
            h_bwd = h_n[-1,:,:]  # (B, H)
            h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        else:
            h = h_n[-1,:,:]  # (B, H)
        h = self.dropout(h)
        logits = self.fc(h)  # (B, num_classes)
        return logits

vocab_size = len(vocab)
model = BiLSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=cfg.embed_dim,
    hidden_size=cfg.hidden_size,
    num_layers=cfg.num_layers,
    bidirectional=cfg.bidirectional,
    num_classes=num_classes,
    pad_idx=pad_idx,
    dropout=0.4
).to(cfg.device)

# -----------------------------
# 6) Training utilities
# -----------------------------
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))  # mixed precision on GPU

def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    for padded_ids, lengths, labels in loader:
        padded_ids = padded_ids.to(cfg.device)
        lengths    = lengths.to(cfg.device)
        labels     = labels.to(cfg.device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
            logits = model(padded_ids, lengths)
            loss = criterion(logits, labels)

        if train:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc  += accuracy(logits.detach(), labels) * batch_size
        total_count += batch_size

    return total_loss / total_count, total_acc / total_count

best_valid = float("inf")
best_state = None

for epoch in range(1, cfg.num_epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    valid_loss, valid_acc = run_epoch(valid_loader, train=False)
    scheduler.step()

    print(f"[Epoch {epoch:02d}] "
          f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
          f"Valid Loss={valid_loss:.4f} Acc={valid_acc:.4f}")
    if valid_loss < best_valid:
        best_valid = valid_loss
        best_state = {
            "model": model.state_dict(),
            "vocab": vocab.get_stoi(),
            "config": cfg.__dict__
        }

# Save best
os.makedirs("checkpoints", exist_ok=True)
torch.save(best_state, "checkpoints/bilstm_agnews.pt")
print("Saved best model to checkpoints/bilstm_agnews.pt")

# -----------------------------
# 7) Test evaluation
# -----------------------------
# Load best (optional, we already have it in memory)
state = torch.load("checkpoints/bilstm_agnews.pt", map_location=cfg.device)
model.load_state_dict(state["model"])

test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"[TEST] Loss={test_loss:.4f} Acc={test_acc:.4f}")

# -----------------------------
# 8) Inference helper
# -----------------------------
id_to_label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def predict(text: str, topk: int = 1):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        padded = ids.unsqueeze(0)  # (1, T)
        padded = padded.to(cfg.device)
        lengths = lengths.to(cfg.device)
        logits = model(padded, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        topk_probs, topk_idx = torch.topk(probs, k=topk)
        return [(id_to_label[i.item()], topk_probs[j].item()) for j, i in enumerate(topk_idx)]

# Quick demo
examples = [
    "The central bank raised interest rates this morning amid inflation concerns.",
    "The team secured a dramatic victory in the final minutes of the match."
]
for ex in examples:
    print(ex, "->", predict(ex, topk=2))


