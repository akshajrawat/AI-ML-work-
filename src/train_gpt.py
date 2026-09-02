import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from tokenizers import Tokenizer

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION (~29M Parameter Model)
# ==============================================================================

# Data & Context hyperparameters
batch_size = 40       # Testing batch size 40 (aiming for ~4.8GB VRAM)
block_size = 256      # Context length in BPE tokens

# Model architecture hyperparameters (~29.01M parameters)
n_embd = 512          # Token embedding width
n_head = 8            # 8 Attention heads (head size = 512 // 8 = 64)
n_layer = 9           # 9 Sequential Transformer Blocks

# Training & LR Scheduler hyperparameters
max_steps = 100       # Benchmark steps
max_lr = 3e-4         # Peak learning rate (0.0003)
min_lr = 3e-5         # Minimum learning rate after decay
warmup_steps = 10     # Linear warmup steps

# ==============================================================================
# 2. LEARNING RATE SCHEDULE (Warmup + Cosine Decay)
# ==============================================================================

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

# ==============================================================================
# 3. LOAD TRAINING & VALIDATION DATA
# ==============================================================================

train_text = Path("data/train.txt").read_text(encoding="utf-8")
val_text = Path("data/val.txt").read_text(encoding="utf-8")

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

print("Vocabulary size:", vocab_size)
print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

# ==============================================================================
# 4. CREATE TRAINING & VALIDATION BATCHES
# ==============================================================================

def get_batch(split):
    data = train_data if split == "train" else val_data

    positions = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in positions
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in positions
    ])

    return x.cuda(), y.cuda()

# ==============================================================================
# 5. TRANSFORMER ARCHITECTURE (~29M GPT)
# ==============================================================================

class Head(nn.Module):
    """ One single head of Self-Attention """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(
                n_embd=n_embd,
                head_size=head_size,
                block_size=block_size
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        x = torch.cat(outputs, dim=-1)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """ MLP / Feed-Forward Network: 4x expansion with ReLU """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer Block: Pre-LayerNorm + Multi-Head Attention + MLP + Residual Connections """

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """ Full GPT Language Model (~29M Parameters) """

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    num_heads=n_head,
                    block_size=block_size
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        token_embeddings = self.token_embedding_table(x)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=x.device))

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, x, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = self(x_cond)

            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

        return x

# Instantiate model on GPU
model = GPTLanguageModel(vocab_size).cuda()

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

# ==============================================================================
# 6. BENCHMARK RUN (100 Steps)
# ==============================================================================

torch.cuda.reset_peak_memory_stats()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

print(f"\nStarting benchmark run with batch_size = {batch_size} (100 steps)...")

torch.cuda.synchronize()
start_time = time.time()

for step in range(max_steps):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch("train")

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 25 == 0 or step == max_steps - 1:
        print(f"step {step:3d} | loss {loss.item():.4f} | lr {lr:.6f}")

torch.cuda.synchronize()
elapsed = time.time() - start_time
total_tokens = max_steps * batch_size * block_size
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print("\n" + "=" * 50)
print("BENCHMARK RESULTS:")
print("=" * 50)
print(f"Batch Size: {batch_size}")
print(f"Training Time: {elapsed:.2f}s")
print(f"Throughput: {total_tokens / elapsed:,.0f} tokens/sec")
print(f"Peak VRAM: {peak_vram:.2f} GB / 6.00 GB")
