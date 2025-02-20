"""
Decoder-only transformer model for language modeling.
Replicating OpenAI GPT2 model from scratch using PyTorch.
Main difference between nanoGPT and nanoGPT2 is that GPT2 
uses LayerNorm instead of BatchNorm, and also 
moved to the input of each sub-block, and an additional layer normalization 
was added after the final self-attention block.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F



@dataclass
class Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens(50.000 token merges + 256 byte tokens + 1 special token)
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
 
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"`n_embd` must be divisible by n_head (got `n_embd`: {config.n_embd} and `n_head`:"
                f" {config.n_head})."
            )
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) 
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # "bias" used this term by openai / mask 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch, time, channel /or batch, sequence, embedding dim
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        # attention (materializes the large(T, T) matrix for all the queries and keys)
        # in official gpt2 used `torch.baddbmm`- batch matrix-matrix product of matrice (it is a bit more efficient)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs) weighted sum of the tokens that model found interesting
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # fc: fully connected cell
        self.gelu = nn.GELU() # GELU activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # projection layer
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 
    

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # communication operation, where all the tokens can communicate, and exchange information
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x)) 
        return x


class GPT2(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers/transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # final layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classifier
        
    def forward(self, idx, targets=None):
        # idx: [batch_size, sequence_length] = [B, T]
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is exhausted."
        # forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # [T]
        pos_emb = self.transformer.wpe(pos) # positional embeding [T, n_embd]
        tok_emb = self.transformer.wte(idx) # token embedding [B, T, n_embd]
        x = tok_emb + pos_emb # [B, T, n_embd]
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer normalization and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


#----------------------------------------------------------
# Autodetection of device (cuda, mps, cpu)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_initialized():
    device = "mps"
print(f"Using device: {device}")

#----------------------------------------------------------
# get a data batch
import tiktoken
enc = tiktoken.get_encoding("gpt2")
with open("data/input.txt", "r") as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long).to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# get logits
# model = GPT2.from_pretrained('gpt2')
model = GPT2(Config())
model.to(device)
# logits, loss = model(x, y)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
    


import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("The meaning of life is")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate text
# right now x is (B, T), where B is batch = 5, T is time = 8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the nodel to get logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take logits at the last position (we only care about the last token)
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        # do top-k sampling of 50
        # topk_probs (5, 50), topk_indices (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select atoken from top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # (B, T+1)
        
# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)