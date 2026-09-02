from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ==============================================================================
# 1. CREATE EMPTY BPE TOKENIZER
# ==============================================================================

# Byte-Pair Encoding (BPE) starts from raw bytes and merges frequent byte pairs.
# [UNK] is the fallback token for any completely unseen characters.
tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)


# ==============================================================================
# 2. SPLIT INITIAL TEXT AT BYTE LEVEL
# ==============================================================================

# ByteLevel splits text at individual byte level (handling all Unicode characters cleanly)
tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

# Decoder converts tokens back to human-readable strings
tokenizer.decoder = ByteLevelDecoder()


# ==============================================================================
# 3. CONFIGURE BPE TRAINER
# ==============================================================================

# vocab_size = 512: The maximum number of tokens the vocabulary will hold
# min_frequency = 2: Only merge pairs of characters/subwords that appear at least twice
trainer = BpeTrainer(
    vocab_size=512,
    min_frequency=2,
    special_tokens=["[UNK]"],
    initial_alphabet=ByteLevel.alphabet(),
)


# ==============================================================================
# 4. LEARN VOCABULARY & MERGE RULES
# ==============================================================================

print("Training BPE tokenizer on data/train.txt...")

tokenizer.train(
    files=["data/train.txt"],
    trainer=trainer
)


# ==============================================================================
# 5. SAVE TOKENIZER TO DISK
# ==============================================================================

Path("tokenizer").mkdir(exist_ok=True)

tokenizer.save(
    "tokenizer/tokenizer.json"
)

print("Tokenizer trained and saved to 'tokenizer/tokenizer.json'!\n")


# ==============================================================================
# 6. TEST TOKENIZER ON SAMPLE TEXTS
# ==============================================================================

examples = [
    "computer",
    "The little girl went outside.",
    "learning",
    "mysterious computer",
]

print("=" * 60)
print("Vocabulary size:", tokenizer.get_vocab_size())
print("=" * 60)

for text in examples:
    encoded = tokenizer.encode(text)

    print("\nTEXT:")
    print(text)

    print("TOKENS (Subwords/Words):")
    print(encoded.tokens)

    print("IDS (Integer representation):")
    print(encoded.ids)

    print("DECODED (Reconstructed text):")
    print(tokenizer.decode(encoded.ids))
