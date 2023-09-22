
from typing import List, Dict, Union
import math
import random
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class HeterogeneousDataLoader:
    """A DataLoader-like object which is not restricted to a 
    NumPy or PyTorch arrays."""
    def __init__(self, *objects, batch_size=32, shuffle=True):
        """
        :param *objects: each object should be a list-like object
        :param batch_size: the batch size
        :param shuffle: whether to shuffle the data
        """
        self.n = len(objects[0])
        assert all(len(obj) == self.n for obj in objects)
        self.points = list(zip(*objects)) # zip the objects together
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.points)
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        batch = self.points[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        return zip(*batch)


class Tokenizer:
    """
    A tokenizer with a custom vocabulary.

    For algorithmic tasks, using pre-existing tokenizers like the gpt-2
    tokenizer will not be ideal due to its extensive vocabulary of
    unneeded tokens. This tokenizer allows defining a custom set of 
    tokens, such as digits 0-9, letters a-z, or any other desired set.
    """
    def __init__(self, vocabulary: List[str]) -> None:
        """Initializes the Tokenizer with a given vocabulary.
        Args:
            vocabulary (List[str]): The list of unique tokens to 
                initialize the tokenizer with.
        Raises:
            AssertionError: If the vocabulary contains duplicate tokens.
        """
        self.vocabulary = list(vocabulary)
        assert len(self.vocabulary) == len(set(vocabulary)), "Vocabulary contains duplicate tokens."
        self._token_to_index: Dict[str, int] = {
            token: index 
            for index, token in enumerate(self.vocabulary)
        }
        # you can override this choice with e.g. tokenizer.pad_token_id = 2
        self.pad_token_id = 0
    
    def encode(self,
            input_string: Union[List[str], str],
            padding: bool=False) -> List[int]:
        """Encodes a string using the tokenizer's vocabulary.
        Args:
            input_string (Union[List[str], str]): The string to tokenize.
            padding (bool, optional): Whether to pad the tokenized 
                sequence to the length of the longest sequence in the 
                batch. Defaults to False. Pads on the right, so that padding 
                tokens aren't seen by the model when predicting the input 
                sequence before the padding. Since we pad from the right, 
                we don't need to worry about passing an attention mask.
        Returns:
            List[int]: List of token indices.
        Raises:
            ValueError: If a substring of the input string cannot be tokenized.
        """
        if not isinstance(input_string, str): # if an iterable of strings
            token_ids_batch = [self.encode(string) for string in input_string]
            if padding:
                max_length = max([len(token_ids) for token_ids in token_ids_batch])
                token_ids_batch = [token_ids + [self.pad_token_id] * (max_length - len(token_ids)) 
                                    for token_ids in token_ids_batch]
            return token_ids_batch
        token_ids: List[int] = []
        start = 0
        while start < len(input_string):
            match_found = False
            for end in range(len(input_string), start, -1):
                substring = input_string[start:end]
                if substring in self.vocabulary:
                    token_ids.append(self._token_to_index[substring])
                    start = end
                    match_found = True
                    break
            if not match_found:
                raise ValueError(f"Token starting at position {start} \
                    cannot be tokenized with the given vocabulary.")
        return token_ids

    def decode(self, token_ids: Union[int, List[int], List[List[int]]]) -> str:
        """Decodes a list of token indices or a single token index into a string.
        Args:
            token_ids (Union[int, List[int], List[List[int]]]): Token indices to decode.
        Returns:
            Union[str, List[str]] Decoded string.
        """
        if isinstance(token_ids, list) and isinstance(token_ids[0], list):
            return [self.decode(token_ids_) for token_ids_ in token_ids]
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join([self.vocabulary[token_id] 
                            for token_id in token_ids])

    def split(self, input_string: str) -> List[str]:
        """Splits an input string into a list of tokens based on the 
        tokenizer's vocabulary.
        Args:
            input_string (str): The string to split.
        Returns:
            List[str]: List of tokens.
        """
        token_ids = self.encode(input_string)
        return [self.decode([token_id]) for token_id in token_ids]

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocabulary)


"""
Transformer based on Andrej Karpathy's nanoGPT implementation:
https://github.com/karpathy/nanoGPT/
"""
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLPLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.layer_norm:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLPLayer(config)
        self.layer_norm = config.layer_norm

    def forward(self, x):
        if self.layer_norm:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x

@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    layer_norm: bool = True
    tie_weights: bool = False

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        if config.layer_norm:
            self.transformer.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.3fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        
        TODO update this depending on whether the the model uses weight tying or not
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the transformer model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        if self.config.layer_norm:
            x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Note that sampling at actually zero temperature is not supported currently. To sample
        at zero temperature, pass in something like temperature=1e-6 instead.
        TODO: implement an option for zero-temperature sampling
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
        
        
@dataclass
class MLPConfig:
    in_dim: int = 2
    out_dim: int = 1
    width: int = 40
    depth: int = 2 # note: depth is the #(linear layers), and #(hidden layers) = #(linear layers) - 1.
    activation = nn.SiLU
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        shp = [config.in_dim] + [config.width]*(config.depth-1) + [config.out_dim]
        layers = []
        for i in range(config.depth):
            layers.append(nn.Linear(shp[i], shp[i+1]))
            if i < config.depth - 1:
                layers.append(config.activation())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
        # input shape = (batch_size, input_dim)
        # define activation here
        #f = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        # f = torch.nn.SiLU()
        # for i in range(self.depth-1):
        #     x = f(self.linears[i](x))
        # x = self.linears[-1](x)
        # # output shape = (batch_size, output_dim)
        # return x
    
@dataclass
class RNNConfig:
    input_dim: int = 2
    output_dim: int = 1
    hidden_dim: int = 40
    
    
class RNN(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.Wh = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.Wx = nn.Linear(config.input_dim, config.hidden_dim)
        self.Wy = nn.Linear(config.hidden_dim, config.output_dim)
        self.act = nn.Sigmoid()
        #self.act = lambda x: x
        self.device = device
    
    def forward(self, x):
        
        # x shape: (batch size, sequence length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)

        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        outs = []

        for i in range(seq_length):
            hidden = self.act(self.Wh(hidden) + self.Wx(x[:,i,:]))
            out = self.Wy(hidden)
            outs.append(out)
            
        # out shape: (batch size, sequence length, output_dim)
        return torch.stack(outs).permute(1,0,2)
