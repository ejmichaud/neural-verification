
from neural_verification import GPTConfig

tokenizer_vocabulary = ["."] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["|", "y", "n"]

model_config = GPTConfig(
    vocab_size=len(tokenizer_vocabulary),
    block_size=12,
    n_layer=5,
    n_head=8,
    n_embd=512,
)

