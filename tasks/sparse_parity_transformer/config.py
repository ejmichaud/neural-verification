
from neural_verification import TransformerConfig

tokenizer_vocabulary = [".", "0", "1", "|"]

model_config = TransformerConfig(
    vocab_size=len(tokenizer_vocabulary),
    block_size=22, # 20 input bits, "|", then the answer
    n_layer=2,
    n_head=4,
    n_embd=128,
    layer_norm=True,
    tie_weights=False
)

