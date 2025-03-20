"""
Decoder-only transformer model for language modeling.
Replicating OpenAI GPT2 model from scratch using PyTorch.(Inference model, not the training one)
Main difference between nanoGPT and nanoGPT2 is that GPT2 
uses LayerNorm instead of BatchNorm, and also 
moved to the input of each sub-block, and an additional layer normalization 
was added after the final self-attention block.
"""

""" 
**notes** 
on my 3080 ti gpu
-after tests there is no significant boost in perfomance between fp32 and tf32,
fp32 => avg dt:14166.99ms-22000.00ms, tok/sec:1079.75-1156.49
tf32 => avg dt:15153.53ms-22892.89ms, tok/sec:715.68-1081.20
-but using bfloat16 has a significant boost(+30%)
bfloat16 => avg dt: 10888.44ms-12218.30ms, tok/sec:1447.49-1578.88

-after imporoving our attention layer to Flash attention, 
we can clearly see nearly x10 boost in speed of the model computations
FlashAttetion => avg dt: 1289.38ms-1609.96ms, tok/sec:10176.66-12706.90

-after correcting our model configuration to "nice" numbers(can be devided by 2)
because our spread of results in time and tok/sec, we cannot clearly see,
but it seems that there is a small boost(+15%) in speed of the model computations
vocab_size->50304 => avg dt: 1036.34ms-1413.23ms, tok/sec:12646.46-16315.44

-after implementing cosine decay learning rate scheduler 
to automatic calculate learning rate for our model
lr_scheduler => lr: 6.0000e-04 | norm: 2.0514 | dt: 949.72ms | tok/sec: 17251.39

-after adding weight decay regularization and FusedKernel AdamW
weight_decay, FusedAdamW => lr: 4.8000e-04 | norm: 2.5717 | dt: 952.17ms | tok/sec: 17207.01

- with gradient accumulation of total_batch_size=524288, 
with batch_size of 16 with 32 gradient accumulation steps.
we can see that our time perfomance multiplied by 3 for each epochs,
but generalization of our model, also often leads to more stable training 
and can improve the model's performance and generalization
grad_accum => lr: 3.0000e-04 | norm: 0.1435 | dt: 31165.59ms | tok/sec: 16000.82

- after we set up Distrbuted Data Parallel, 
if you have multiple gpu cluster(if you are rich),
of course my results have no upgraded, because i use only 1 gpu,
but if you have multiple gpu cluster, you can see that your time perfomance will be multiplied
DDP =>  lr: 5.4000e-04 | norm: 0.1175 | dt: 32125.44ms | tok/sec: 16320.02
"""

import os
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import tiktoken
import inspect
import platform
import numpy as np
from transformers import GPT2LMHeadModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from hellaswag import iterate_examples, render_example

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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # let our std inside the residual stream normalized
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
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs) weighted sum of the tokens that model found interesting
        
        # instead we implement Flash Attention to improve speed of the model
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd) # fc: fully connected cell
        self.gelu   = nn.GELU() # GELU activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # projection layer
        self.c_proj.NANOGPT_SCALE_INIT = 1 # let our std inside the residual stream normalized

        
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
        
        # weight sharing scheme(from Attention is all you need, where we share same weight matrix between pre-softmax layer(lme_head) and embedding layer(wte))
        self.transformer.wte.weight = self.lm_head.weight
        
        # init parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5 # multiply by 2, because of every single of layers in transformer has 2 blocks that add to residual pathway(MLP and attn)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start all of the candidate parameters(that reqiure grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2d will be weight decayed, otherwise no
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # decay weights that participate in matmuls and embeddings
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # nodecay 1Dim parameters like biases and layernorms
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ] 
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if its available
        # fused - instead of iterating in a for-loop all parameter tensors, we fusing all kernels into a single kernel and update all the params in single time
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters 
        use_fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer
#----------------------------------------------------------
# data loader 

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # Earlier version of PyTorch may have difficulty converting from uint16 to long.
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        
        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        
        # # at init load tokens from disk and store them in memory
        # with open("data/input.txt", "r") as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"Loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
    
    def reset(self):   
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards(self.current_shard))
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading next batch would overflow the buffer, advance to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
#----------------------------------------------------------
# helper function for HellaSwag evaluation to choose option with the lowest loss
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

#----------------------------------------------------------
# Run the training loop

# set up DDP(distributed data parallel) if you have multiple gpu cluster(if you are rich)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1 # check is it a DDP run
if ddp:
    # use of DDP atm demands CUDA, we set device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl") # DDP settings 'nccl', 'gloo', etc.
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # basically how many GPUs we are using 
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else: 
    # vanilla, no-DDP run(if you are typical mortal being, like me)
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Autodetection of device (cuda, mps, cpu)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_initialized():
        device = "mps"
    print(f"Using device: {device}")

# pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"
#----------------------------------------------------------
# reproducibility
torch.manual_seed(42)
if torch.cuda.is_available:
    torch.cuda.manual_seed(42)

#----------------------------------------------------------
total_batch_size = 524288 # 2**19(nice number)tokens per step, ~0.5M, 
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # 524288/(16*1024) = 32 steps forward/backward for a single update
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"-> calculated gradient accumulation step: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train") # train data, B - batch size, T - max sequence length
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") # valid data, B - batch size, T - max sequence length

# transforming datatype of floatpoint(fp32) to tensorfloat(tf32)
torch.set_float32_matmul_precision("high")

#----------------------------------------------------------
# create model
model = GPT2(Config(vocab_size=50304)) # increasing vocab_size to 50304 because its nice num and can be devided by 2(128)
model.to(device)
# identify in which OS model is running on if its Linux, we can use torch.compile()
# cannot use torch.compile on Windows OS, beacause
# running TorchInductor requires Triton(torchtriton), 
# that only support Linux OS, so as compromiss
# we can use Windows Subsystem for Linux
if platform.system() == "Linux":
    model = torch.compile(model)
else:
    print(f"-> Cannot use torch.compile on Windows OS, beacause running TorchInductor requires Triton(torchtriton), that only support Linux OS")
# wrap model into DDP container
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

#----------------------------------------------------------
# cosine decay learning rate scheduler
# we decay the learning rate with a cosine annealing for each batch as follows:
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375e6 warmup lr / 2**19
max_steps = 19073 # 10e9 unique tokens/2**19 | 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iter, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coef start at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


#----------------------------------------------------------
# optimization
# using some OpenAI GPT-3 parameters that was in their released paper 
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
enc = tiktoken.get_encoding("gpt2")

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while evaluate our validation loss
    if step % 250 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model(except step 0, which is noise)
    # have a argues with torch.compile, so if don't use it works fine
    if (step % 250 == 0 or last_step) and (not use_compile):
        # prefix tokens
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("The meaning of life is")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take logits at the last position (we only care about the last token)
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                # do top-k sampling of 50 (hugginface pipeline default)
                # topk_probs (5, 50), topk_indices (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select atoken from top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                x = torch.cat((x, xcol), dim=1) # (B, T+1)
        
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank: {ddp_rank} sample {i}: {decoded}")

        
    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # autocast should wrap only the forward pass(es) of your network, including the loss computation(s). Backward passes under autocast are not recommended.
            logits, loss = model(x, y) 
        # we have to scale the loss to account a gradient accumulation
        # because the gradients just add on each successive backward()
        # addition of gradients corresponds to a SUM in a objective, but
        # instead of SUM we want MEAN. Scale the loss here so it comes out right 
        loss = loss / grad_accum_steps # this is normalizer
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # we clip the global norm of the grad 1.0, to prevent the model getting too big of shocks in terms of gradient magnitude
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize() # Wait for all kernels in all streams on a CUDA device to complete.
    t1 = time.time()
    dt = t1 - t0 # time diff in sec
    tokens_processed = (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
if ddp:
    destroy_process_group()


