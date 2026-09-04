import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from tokenizers import Tokenizer

# ==============================================================================
# 11M RoPE GPT PRETRAINING EXPERIMENT (15,000 Steps)
# ==============================================================================
# This script trains an 11M Transformer using Rotary Position Embeddings (RoPE)
# to compare head-to-head with our baseline 11M model (learned absolute positions).
# ==============================================================================

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION (11M Model Matching Baseline)
# ==============================================================================

# Data & Context hyperparameters
batch_size = 64       # 64 parallel sequences (64 x 256 = 16,384 tokens/step)
block_size = 256      # Maximum context window length in BPE tokens

# Model architecture hyperparameters (~11.03M parameters with RoPE)
n_embd = 384          # Embedding dimension (width)
n_head = 6            # 6 Attention heads (head size = 384 // 6 = 64)
n_layer = 6           # 6 Sequential Transformer Blocks

# Training & LR Scheduler hyperparameters
max_steps = 15000     # 15,000 steps (~246M token presentations)
max_lr = 3e-4         # Peak learning rate (0.0003)
min_lr = 3e-5         # Minimum learning rate after cosine decay (10% of peak)
warmup_steps = 500    # Linear warmup across the first 500 steps

# ==============================================================================
# 2. LEARNING RATE SCHEDULE (Warmup + Cosine Decay)
# ==============================================================================

def get_lr(step):
    # Phase 1: Linear Warmup (steps 0 -> 500)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 2: Post-schedule clamp
    if step >= max_steps:
        return min_lr

    # Phase 3: Cosine Annealing Decay (steps 500 -> 15000)
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
# 5. ROTARY POSITION EMBEDDINGS (RoPE) HELPER FUNCTIONS
# ==============================================================================

def build_rope_cache(block_size, head_size, base=10000):
    assert head_size % 2 == 0, "RoPE requires an even head dimension!"

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, head_size, 2, dtype=torch.float32)
            / head_size
        )
    )

    positions = torch.arange(block_size, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)

    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: [B, T, head_size]
    T = x.shape[1]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos[:T].unsqueeze(0).to(dtype=x.dtype)
    sin = sin[:T].unsqueeze(0).to(dtype=x.dtype)

    even_rot = x_even * cos - x_odd * sin
    odd_rot = x_even * sin + x_odd * cos

    return torch.stack(
        (even_rot, odd_rot),
        dim=-1
    ).flatten(-2)

# ==============================================================================
# 6. TRANSFORMER ARCHITECTURE WITH RoPE
# ==============================================================================

class Head(nn.Module):
    """ One single head of Self-Attention with Rotary Position Embeddings (RoPE) """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        cos, sin = build_rope_cache(block_size, head_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Rotate Q and K vectors according to token positions
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Causal scaled dot-product attention
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

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
    """ ~11M GPT using Rotary Position Embeddings (RoPE) """

    def __init__(self, vocab_size):
        super().__init__()

        # Token embeddings only (no learned position embeddings table!)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

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

        x = token_embeddings
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

# ==============================================================================
# 7. INITIALIZE MODEL & MEASURE PARAMETERS
# ==============================================================================

model = GPTLanguageModel(vocab_size).cuda()

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

# ==============================================================================
# 8. LOSS ESTIMATION (Measuring Generalization with BF16 Autocast)
# ==============================================================================

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = []
        for _ in range(50):
            x, y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)

    model.train()
    return results

# ==============================================================================
# 9. GENERATE TEXT BEFORE TRAINING (Random Guess Baseline)
# ==============================================================================

start = torch.zeros((1, 1), dtype=torch.long).cuda()
print("\nBEFORE TRAINING:")
print(decode(model.generate(start, 100)[0].tolist()))

# ==============================================================================
# 10. TRAINING LOOP (15,000 Steps with Grad Clipping & Periodic Checkpoints)
# ==============================================================================

torch.cuda.reset_peak_memory_stats()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
Path("checkpoints").mkdir(exist_ok=True)

best_val_loss = float("inf")

print(f"\nStarting 15,000-step 11M RoPE pretraining run...")
start_time = time.time()

for step in range(max_steps):

    # 1. Update learning rate according to Warmup + Cosine schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # 2. Validation evaluation & Logging every 500 steps
    if step % 500 == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(
            f"step {step:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | "
            f"lr {lr:.6f}"
        )

        # Track and save best validation model
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                        "block_size": block_size,
                        "vocab_size": vocab_size,
                        "pos_type": "rope",
                    },
                    "val_loss": best_val_loss,
                },
                "checkpoints/gpt_11m_rope_best.pt"
            )
            print(f"       -> [NEW BEST] Saved best RoPE model (val_loss: {best_val_loss:.4f}) to 'checkpoints/gpt_11m_rope_best.pt'")

    # 3. Checkpointing every 1000 steps
    if step > 0 and step % 1000 == 0:
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "block_size": block_size,
                    "vocab_size": vocab_size,
                    "pos_type": "rope",
                },
            },
            f"checkpoints/gpt_11m_rope_step_{step}.pt"
        )

    # 4. Fetch random training batch
    x, y = get_batch("train")

    # 5. Forward pass with BF16 Mixed Precision
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)

    # 6. Backward pass (Backpropagation)
    loss.backward()

    # 7. Gradient Clipping: prevents exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 8. Optimizer step: update weights
    optimizer.step()

# Compute final benchmark statistics
torch.cuda.synchronize()
elapsed = time.time() - start_time
total_tokens = max_steps * batch_size * block_size
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print("\n" + "=" * 50)
print("11M RoPE TRAINING RUN COMPLETE:")
print("=" * 50)
print(f"Total time: {elapsed / 60:.1f} minutes ({elapsed:.1f}s)")
print(f"Average throughput: {total_tokens / elapsed:,.0f} tokens/sec")
print(f"Peak VRAM: {peak_vram:.2f} GB / 6.00 GB")
print(f"Best Validation Loss: {best_val_loss:.4f}")

# ==============================================================================
# 11. GENERATE TEXT AFTER TRAINING (Sample with Temperature & Top-K)
# ==============================================================================

print("\nAFTER TRAINING (Sample Generation with Temperature=0.8, Top-K=40):")
print("=" * 50)
generated = model.generate(start, max_new_tokens=250, temperature=0.8, top_k=40)
print(decode(generated[0].tolist()))

# ==============================================================================
# 12. SAVE FINAL CHECKPOINT
# ==============================================================================

torch.save(
    {
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "vocab_size": vocab_size,
            "pos_type": "rope",
        },
        "val_loss": losses["val"],
    },
    "checkpoints/gpt_11m_rope_final.pt"
)

print("\nFinal checkpoint successfully saved to 'checkpoints/gpt_11m_rope_final.pt'.")
