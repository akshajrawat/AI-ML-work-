import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# ==============================================================================
# 1. TRANSFORMER ARCHITECTURE (GPT)
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

        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        v = self.value(x)   # [B, T, head_size]
        out = weights @ v   # [B, T, head_size]
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
    """ Full GPT Language Model """

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


# ==============================================================================
# 2. LOAD CHECKPOINT & RECONSTRUCT CONFIGURATION
# ==============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = Path("checkpoints/gpt_11m_best.pt")
if not checkpoint_path.exists():
    checkpoint_path = Path("checkpoints/gpt_11m_final.pt")

if not checkpoint_path.exists():
    # Fallback to any saved checkpoint
    checkpoints = sorted(list(Path("checkpoints").glob("*.pt")))
    if checkpoints:
        checkpoint_path = checkpoints[-1]
    else:
        raise FileNotFoundError("No checkpoint found in checkpoints/. Run training first!")

print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]

n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
block_size = config["block_size"]
vocab_size = config["vocab_size"]

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

model = GPTLanguageModel(vocab_size).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"11M GPT loaded ({num_params:,} parameters).")
print("Type 'exit' or 'quit' to stop.\n")


# ==============================================================================
# 3. GENERATION WITH TEMPERATURE & TOP-K SAMPLING
# ==============================================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
):
    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits, _ = model(x_cond)

        logits = logits[:, -1, :]

        # -------------------------
        # TEMPERATURE SCALING
        # -------------------------
        # Low (<1.0) = sharper/safer/focused, High (>1.0) = flatter/creative/random
        if temperature > 0:
            logits = logits / temperature

        # -------------------------
        # TOP-K FILTERING
        # -------------------------
        # Only keep top-k highest scoring tokens and mask the rest to -inf
        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )
            cutoff = values[:, [-1]]
            logits = logits.masked_fill(
                logits < cutoff,
                float("-inf")
            )

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        x = torch.cat(
            (x, next_token),
            dim=1
        )

    return tokenizer.decode(
        x[0].tolist()
    )


# ==============================================================================
# 4. INTERACTIVE PROMPTING REPL (COMPARING ALL 3 SAMPLING SETTINGS)
# ==============================================================================

sampling_configs = [
    {
        "name": "1. CONSERVATIVE / FOCUSED",
        "temperature": 0.3,
        "top_k": 20,
        "description": "High probability tokens only, safer & repetitive"
    },
    {
        "name": "2. BALANCED (DEFAULT)",
        "temperature": 0.8,
        "top_k": 40,
        "description": "Natural balance of coherence and creativity"
    },
    {
        "name": "3. CREATIVE / DIVERSE",
        "temperature": 1.2,
        "top_k": 100,
        "description": "Higher entropy, more vocabulary diversity"
    },
]

while True:
    try:
        prompt = input("\n" + "=" * 60 + "\nEnter Prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break

    if not prompt:
        continue

    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    for config_item in sampling_configs:
        t = config_item["temperature"]
        k = config_item["top_k"]

        print("\n" + "-" * 60)
        print(f"[{config_item['name']}] -> temperature={t}, top_k={k}")
        print(f"Description: {config_item['description']}")
        print("-" * 60)

        result = generate(
            prompt,
            max_new_tokens=150,
            temperature=t,
            top_k=k,
        )

        print(result)
