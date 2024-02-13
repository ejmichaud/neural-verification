
from neural_verification import TransformerConfig

tokenizer_vocabulary = ["."] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["|", "y", "n"]

model_config = TransformerConfig(
    vocab_size=len(tokenizer_vocabulary),
    block_size=12,
    n_layer=5,
    n_head=8,
    n_embd=512,
    layer_norm=True,
    tie_weights=False
)

