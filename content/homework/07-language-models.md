---
title: "Homework 7: Autoregressive Language Models"
description: "Next-token batches, causal multi-head attention, small GPT training, and autoregressive sampling."
publish: true
---

[Back to Module 7 notes](/modules/07-language-models)

## Coding

Complete the coding problems in one Jupyter notebook. The default dataset is the character-level [Tiny Shakespeare corpus](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/data/shakespeare_char), although you may use another text corpus of similar size. A character-level tokenizer is sufficient for this assignment.

You may use standard PyTorch layers and tensor operations. Implement the next-token batches, causal attention calculation, Transformer block, and autoregressive sampling loop yourself. The four coding problems follow the main pieces of Karpathy's [nanoGPT repository](https://github.com/karpathy/nanoGPT). Each problem links to the corresponding source lines for reference. Your notebook should be a small, self-contained implementation of these pieces rather than a copy of the repository.

> [!problem|Data and Next-Token Batches]
> Build the data pipeline for next-token prediction.
>
> **nanoGPT reference:** [character vocabulary, encoding, and data split](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/data/shakespeare_char/prepare.py#L23-L44); [shifted minibatches](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L114-L131).
>
> 1.  Construct a character vocabulary from the corpus and implement `encode` and `decode`. Check that decoding an encoded string recovers the original string.
> 2.  Split the token sequence into training and validation portions. Implement `get_batch(split)` so that it returns integer tensors `x` and `y` of shape $(B,T)$, where $B$ is the batch size and $T$ is the context length.
> 3.  For a sampled chunk $(s_0,s_1,\ldots,s_T)$, the input and target should be
>
>     $$
>     x=(s_0,s_1,\ldots,s_{T-1}),
>     \qquad
>     y=(s_1,s_2,\ldots,s_T).
>     $$
>
>     Verify that `torch.equal(x[0, 1:], y[0, :-1])` is true. Then decode one row of `x` and `y` and check the shift in readable form.

> [!solution]- Solution
>
> Assign one integer to each character. Sorting the characters makes the encoding reproducible.
>
> ```python
> from pathlib import Path
> import torch
>
> text = Path("input.txt").read_text(encoding="utf-8")
>
> chars = sorted(set(text))
> stoi = {ch: i for i, ch in enumerate(chars)}
> itos = {i: ch for ch, i in stoi.items()}
> vocab_size = len(chars)
>
> def encode(s):
>     return [stoi[ch] for ch in s]
>
> def decode(ids):
>     if isinstance(ids, torch.Tensor):
>         ids = ids.tolist()
>     return "".join(itos[int(i)] for i in ids)
>
> test_string = text[:100]
> assert decode(encode(test_string)) == test_string
> ```
>
> Encode the corpus, then split the token sequence into training and validation sets. Each target is the same chunk shifted by one token.
>
> ```python
> data = torch.tensor(encode(text), dtype=torch.long)
> split = int(0.9 * len(data))
> train_data = data[:split]
> val_data = data[split:]
>
> batch_size = 12
> block_size = 64
> device = "cuda" if torch.cuda.is_available() else "cpu"
>
> def get_batch(split_name):
>     source = train_data if split_name == "train" else val_data
>     starts = torch.randint(len(source) - block_size, (batch_size,))
>     x = torch.stack([source[i : i + block_size] for i in starts])
>     y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
>     return x.to(device), y.to(device)
>
> x, y = get_batch("train")
> assert x.shape == y.shape == (batch_size, block_size)
> assert x.dtype == y.dtype == torch.long
> assert torch.equal(x[0, 1:], y[0, :-1])
>
> print("x:", repr(decode(x[0].cpu())))
> print("y:", repr(decode(y[0].cpu())))
> ```
>
> The assertion checks the shift directly. The two printed strings show the same shift in text form.

> [!problem|Causal Multi-Head Self-Attention]
> Implement causal multi-head self-attention. For hidden states $H\in\mathbb R^{B\times T\times d}$, use learned linear maps to form queries, keys, and values with shape $(B,h,T,d_h)$, where $h$ is the number of heads and $d_h=d/h$. For each head, compute
>
> $$
> A=\frac{QK^\top}{\sqrt{d_h}},
> \qquad
> P=\operatorname{softmax}(A+M),
> \qquad
> O=PV,
> $$
>
> where $M_{ij}=-\infty$ when $j>i$ and $M_{ij}=0$ otherwise. Combine the heads and apply an output projection.
>
> **nanoGPT reference:** [`CausalSelfAttention`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L29-L76), including both the fused and explicit attention paths.
>
> nanoGPT uses PyTorch's fused scaled dot-product attention when it is available and also includes an explicit implementation of the calculation above. For this assignment, implement the explicit version rather than calling `F.scaled_dot_product_attention`.
>
> Before training, check the implementation in two ways:
>
> 1.  In evaluation mode, temporarily return or save the attention probabilities $P$. Inspect `P[0, 0]` and verify that entries above the diagonal are zero. The normal forward method only needs to return the attention output.
> 2.  Create hidden states `H1` and `H2` with the same prefix, `H1[:, :m] == H2[:, :m]`, but different values after position `m`. Verify that the corresponding outputs agree on `[:, :m]` up to numerical precision.

> [!solution]- Solution
>
> Register the lower-triangular mask as a buffer so that it moves with the module but is not trained. In evaluation mode, save the attention probabilities before dropout for inspection.
>
> ```python
> import math
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
>
> class CausalSelfAttention(nn.Module):
>     def __init__(self, n_embd, n_head, block_size, dropout=0.0, bias=True):
>         super().__init__()
>         assert n_embd % n_head == 0
>
>         self.n_head = n_head
>         self.head_dim = n_embd // n_head
>         self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
>         self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
>         self.attn_dropout = nn.Dropout(dropout)
>         self.resid_dropout = nn.Dropout(dropout)
>
>         mask = torch.tril(
>             torch.ones(block_size, block_size, dtype=torch.bool)
>         )
>         self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))
>         self.last_attn = None
>
>     def forward(self, x):
>         B, T, C = x.shape
>         q, k, v = self.qkv(x).chunk(3, dim=-1)
>
>         q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
>         k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
>         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
>
>         scores = q @ k.transpose(-2, -1)
>         scores = scores / math.sqrt(self.head_dim)
>         mask = self.causal_mask[:, :, :T, :T]
>         scores = scores.masked_fill(~mask, float("-inf"))
>
>         probs = F.softmax(scores, dim=-1)
>         self.last_attn = probs.detach() if not self.training else None
>         probs = self.attn_dropout(probs)
>
>         out = probs @ v
>         out = out.transpose(1, 2).contiguous().view(B, T, C)
>         return self.resid_dropout(self.out_proj(out))
> ```
>
> Check both the upper triangle of the attention matrix and the output on a fixed prefix.
>
> ```python
> torch.manual_seed(0)
> attn = CausalSelfAttention(
>     n_embd=32, n_head=4, block_size=8, dropout=0.0
> ).eval()
>
> H1 = torch.randn(2, 8, 32)
> Y1 = attn(H1)
> P = attn.last_attn
> assert torch.count_nonzero(P[0, 0].triu(diagonal=1)) == 0
>
> m = 5
> H2 = H1.clone()
> H2[:, m:] = torch.randn_like(H2[:, m:])
> Y1 = attn(H1)
> Y2 = attn(H2)
> assert torch.allclose(Y1[:, :m], Y2[:, :m], atol=1e-6, rtol=1e-5)
> ```
>
> The second test changes only positions $m,\ldots,T-1$. The first $m$ outputs remain unchanged because they cannot attend to those positions.

> [!problem|Train a Small GPT]
> Build a decoder-only Transformer with the following structure:
>
> $$
> \text{token embedding}+\text{position embedding}
> \longrightarrow
> L\text{ Transformer blocks}
> \longrightarrow
> \text{layer normalization}
> \longrightarrow
> \text{linear vocabulary head}.
> $$
>
> **nanoGPT model reference:** [`MLP` and `Block`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L78-L106); [`GPT` architecture and `forward`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L118-L193).
>
> Use nanoGPT's pre-normalization block with residual connections:
>
> $$
> H\leftarrow H+\operatorname{Attention}(\operatorname{LayerNorm}(H)),
> \qquad
> H\leftarrow H+\operatorname{MLP}(\operatorname{LayerNorm}(H)).
> $$
>
> As in nanoGPT, use an MLP that expands each hidden state from dimension $d$ to $4d$, applies GELU, and projects it back to $d$. Implement `forward(idx, targets=None)` so that it returns `(logits, loss)`:
>
> 1.  When `targets` are provided, compute logits of shape $(B,T,|\mathcal V|)$ and the next-token cross-entropy loss.
> 2.  When `targets=None`, follow nanoGPT's inference path: compute vocabulary logits only at the final position, with shape $(B,1,|\mathcal V|)$, and return `loss=None`.
>
> Train the model with next-token cross-entropy and AdamW. As in nanoGPT, evaluate training and validation loss periodically; store these logged values and plot them in the notebook. See [`estimate_loss`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L214-L228) and the main [training loop](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L249-L314).
>
> For a light first run, you can start with `block_size=64`, `batch_size=12`, `n_layer=4`, `n_head=4`, and `n_embd=128`, following nanoGPT's [small CPU configuration](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/README.md#L82-L88). Report the model size and main training choices, plot the losses, and show a text sample from the final model. There is no required validation-loss target.

> [!solution]- Solution
>
> Using the attention layer from Problem 7.2, define the MLP, Transformer block, and GPT model as follows.
>
> ```python
> from dataclasses import dataclass
> import matplotlib.pyplot as plt
>
> @dataclass
> class GPTConfig:
>     vocab_size: int
>     block_size: int
>     n_layer: int = 4
>     n_head: int = 4
>     n_embd: int = 128
>     dropout: float = 0.1
>     bias: bool = True
>
> class MLP(nn.Module):
>     def __init__(self, config):
>         super().__init__()
>         self.net = nn.Sequential(
>             nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
>             nn.GELU(),
>             nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
>             nn.Dropout(config.dropout),
>         )
>
>     def forward(self, x):
>         return self.net(x)
>
> class Block(nn.Module):
>     def __init__(self, config):
>         super().__init__()
>         self.ln_1 = nn.LayerNorm(config.n_embd)
>         self.attn = CausalSelfAttention(
>             config.n_embd,
>             config.n_head,
>             config.block_size,
>             dropout=config.dropout,
>             bias=config.bias,
>         )
>         self.ln_2 = nn.LayerNorm(config.n_embd)
>         self.mlp = MLP(config)
>
>     def forward(self, x):
>         x = x + self.attn(self.ln_1(x))
>         x = x + self.mlp(self.ln_2(x))
>         return x
>
> class GPT(nn.Module):
>     def __init__(self, config):
>         super().__init__()
>         self.config = config
>         self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
>         self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
>         self.dropout = nn.Dropout(config.dropout)
>         self.blocks = nn.ModuleList(
>             [Block(config) for _ in range(config.n_layer)]
>         )
>         self.ln_f = nn.LayerNorm(config.n_embd)
>         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
>
>     def forward(self, idx, targets=None):
>         B, T = idx.shape
>         if T > self.config.block_size:
>             raise ValueError("sequence is longer than block_size")
>
>         positions = torch.arange(T, device=idx.device)
>         x = self.token_embedding(idx) + self.position_embedding(positions)
>         x = self.dropout(x)
>
>         for block in self.blocks:
>             x = block(x)
>         x = self.ln_f(x)
>
>         if targets is None:
>             logits = self.lm_head(x[:, [-1], :])
>             return logits, None
>
>         logits = self.lm_head(x)
>         loss = F.cross_entropy(
>             logits.reshape(-1, self.config.vocab_size),
>             targets.reshape(-1),
>         )
>         return logits, loss
> ```
>
> Check the training and inference branches before training.
>
> ```python
> torch.manual_seed(0)
> config = GPTConfig(vocab_size=vocab_size, block_size=block_size)
> model = GPT(config).to(device)
>
> xb, yb = get_batch("train")
> train_logits, train_loss = model(xb, yb)
> test_logits, test_loss = model(xb)
>
> assert train_logits.shape == (batch_size, block_size, vocab_size)
> assert train_loss.ndim == 0
> assert test_logits.shape == (batch_size, 1, vocab_size)
> assert test_loss is None
>
> n_params = sum(p.numel() for p in model.parameters())
> print(f"parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
> print(config)
> ```
>
> Estimate each loss by averaging over several minibatches. Restore the previous training mode afterward.
>
> ```python
> @torch.no_grad()
> def estimate_loss(model, eval_iters=50):
>     was_training = model.training
>     model.eval()
>     result = {}
>
>     for split_name in ("train", "val"):
>         losses = []
>         for _ in range(eval_iters):
>             x, y = get_batch(split_name)
>             _, loss = model(x, y)
>             losses.append(loss.item())
>         result[split_name] = sum(losses) / len(losses)
>
>     model.train(was_training)
>     return result
>
> optimizer = torch.optim.AdamW(
>     model.parameters(), lr=3e-4, weight_decay=0.01
> )
>
> max_steps = 3000
> eval_interval = 200
> steps, train_losses, val_losses = [], [], []
>
> model.train()
> for step in range(max_steps + 1):
>     if step % eval_interval == 0:
>         losses = estimate_loss(model)
>         steps.append(step)
>         train_losses.append(losses["train"])
>         val_losses.append(losses["val"])
>         print(
>             f"step {step:4d}: "
>             f"train {losses['train']:.3f}, val {losses['val']:.3f}"
>         )
>
>     if step == max_steps:
>         break
>
>     x, y = get_batch("train")
>     _, loss = model(x, y)
>     optimizer.zero_grad(set_to_none=True)
>     loss.backward()
>     optimizer.step()
>
> plt.plot(steps, train_losses, label="train")
> plt.plot(steps, val_losses, label="validation")
> plt.xlabel("training step")
> plt.ylabel("cross-entropy")
> plt.legend()
> plt.show()
> ```
>
> Record the printed configuration and parameter count together with the loss plot. In a successful run, both losses fall early in training. If the training loss later falls while the validation loss rises, the model is overfitting. Use Problem 7.4 to generate the final text sample.

> [!problem|Autoregressive Sampling]
> Implement nanoGPT's autoregressive sampling loop and run it with `model.eval()` inside `torch.no_grad()`. At each step, keep only the most recent $T$ tokens if the sequence is longer than the context length, run the model, and select the logits from the final position. Divide the logits by `temperature`, optionally set all but the largest `top_k` logits to $-\infty$, apply softmax, and sample one token with `torch.multinomial`.
>
> **nanoGPT reference:** [`GPT.generate`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L305-L330); [sampling settings](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/sample.py#L12-L19); [prompt encoding and generation](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/sample.py#L51-L88).
>
> Use the same checkpoint and prompt to generate text at two different temperatures, for example $0.7$ and $1.2$. Then fix the temperature and compare generation with and without a `top_k` cutoff. Choose `top_k` smaller than the vocabulary size; values such as 10 or 20 work for Tiny Shakespeare. Show representative samples and briefly describe how the generated text changes. Set the same random seed before each run so that the comparison is easier to interpret.

> [!solution]- Solution
>
> At each step, crop only the model input. Keep the full sequence in `idx`, and sample the next token from the final-position logits.
>
> ```python
> @torch.no_grad()
> def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
>     if temperature <= 0:
>         raise ValueError("temperature must be positive")
>
>     model.eval()
>     for _ in range(max_new_tokens):
>         idx_cond = idx[:, -model.config.block_size :]
>         logits, _ = model(idx_cond)
>         logits = logits[:, -1, :] / temperature
>
>         if top_k is not None:
>             if top_k <= 0:
>                 raise ValueError("top_k must be positive")
>             k = min(top_k, logits.size(-1))
>             cutoff = torch.topk(logits, k).values[:, [-1]]
>             logits = logits.masked_fill(logits < cutoff, float("-inf"))
>
>         probs = F.softmax(logits, dim=-1)
>         idx_next = torch.multinomial(probs, num_samples=1)
>         idx = torch.cat((idx, idx_next), dim=1)
>
>     return idx
> ```
>
> Reset the random seed for each comparison so that the prompt, checkpoint, and initial random state are the same.
>
> ```python
> prompt = "ROMEO:"
> context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
>
> def sample_text(temperature, top_k, seed=1234):
>     torch.manual_seed(seed)
>     tokens = generate(
>         model,
>         context,
>         max_new_tokens=300,
>         temperature=temperature,
>         top_k=top_k,
>     )
>     return decode(tokens[0].cpu())
>
> print("temperature 0.7")
> print(sample_text(temperature=0.7, top_k=None))
>
> print("temperature 1.2")
> print(sample_text(temperature=1.2, top_k=None))
>
> print("no top-k")
> print(sample_text(temperature=0.8, top_k=None))
>
> print("top-k = 10")
> print(sample_text(temperature=0.8, top_k=10))
> ```
>
> With the same checkpoint, the lower-temperature sample chooses high-logit characters more often. The higher-temperature sample draws from a flatter distribution and therefore varies more. With `top_k=10`, only the ten largest logits can be sampled at each step. The exact strings depend on the trained checkpoint.

## Theory

Let $(w_1,\ldots,w_L)$ be a token sequence with $w_i\in\mathcal V$. An autoregressive language model represents

$$
p_\theta(w_1,\ldots,w_L)
=
\prod_{i=1}^{L}p_\theta(w_i\mid w_{<i}).
$$

> [!problem|Next-Token Training Objective]
> Write the negative log-likelihood of the sequence $(w_1,\ldots,w_L)$. If the training input is $(w_1,\ldots,w_{L-1})$ and the labels are $(w_2,\ldots,w_L)$, which terms of the negative log-likelihood are included by the shifted cross-entropy loss? Explain how prepending a beginning-of-sequence token allows the model to include the term for $w_1$ as well.

> [!solution]- Solution
>
> The sequence negative log-likelihood is
>
> $$
> -\log p_\theta(w_1,\ldots,w_L)
> =
> -\sum_{i=1}^{L}
> \log p_\theta(w_i\mid w_{<i}).
> $$
>
> With input $(w_1,\ldots,w_{L-1})$ and labels $(w_2,\ldots,w_L)$, position $i$ predicts $w_{i+1}$ from $w_{\leq i}$. Thus the shifted loss is
>
> $$
> -\sum_{i=1}^{L-1}
> \log p_\theta(w_{i+1}\mid w_{\leq i})
> =
> -\sum_{j=2}^{L}
> \log p_\theta(w_j\mid w_{<j}).
> $$
>
> This includes the terms for $w_2,\ldots,w_L$ but not the term for $w_1$. To include every token, use
>
> $$
> \text{input}=(\mathtt{BOS},w_1,\ldots,w_{L-1}),
> \qquad
> \text{targets}=(w_1,w_2,\ldots,w_L).
> $$
>
> The first position then contributes $-\log p_\theta(w_1\mid\mathtt{BOS})$.

> [!problem|Causal Masking and Parallel Training]
> For a sequence of length four, write the $4\times4$ attention mask, using one when position $i$ may attend to position $j$ and zero otherwise. Explain why entries above the diagonal must be masked. Since the whole training sequence is available, why can the model still compute the outputs at all four positions in one forward pass?

> [!solution]- Solution
>
> The binary mask is
>
> $$
> \begin{bmatrix}
> 1&0&0&0\\
> 1&1&0&0\\
> 1&1&1&0\\
> 1&1&1&1
> \end{bmatrix}.
> $$
>
> Row $i$ may use positions up to and including $i$. For example, position $i$ predicts $w_{i+1}$, so allowing it to attend to position $i+1$ would reveal the target token.
>
> During training, the complete sequence is already available. The model computes all queries, keys, values, and attention scores in one set of batched matrix operations. Applying the mask before softmax removes the forbidden entries without requiring a separate forward pass for each position.

> [!problem|Why Attention Logits Are Scaled]
> Let $q,k\in\mathbb R^{d_h}$ be a query and key whose coordinates are independent, with mean zero and variance one. Compute the variance of $q^\top k$. Use the result to explain why attention divides this dot product by $\sqrt{d_h}$ before applying softmax.

> [!solution]- Solution
>
> The dot product is
>
> $$
> q^\top k=\sum_{r=1}^{d_h}q_rk_r.
> $$
>
> Each product has mean zero and
>
> $$
> \operatorname{Var}(q_rk_r)
> =
> \mathbb E[q_r^2]\mathbb E[k_r^2]
> =1.
> $$
>
> The products are independent across coordinates. Therefore,
>
> $$
> \operatorname{Var}(q^\top k)=d_h.
> $$
>
> The unscaled dot product has standard deviation $\sqrt{d_h}$. After scaling,
>
> $$
> \operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_h}}\right)=1.
> $$
>
> The scaled logits have standard deviation $1$, independent of the head dimension. Without the scaling, increasing $d_h$ makes softmax more saturated and its gradients smaller.

> [!problem|Temperature and Top-k Sampling]
> Given next-token logits $z\in\mathbb R^{|\mathcal V|}$, temperature sampling uses
>
> $$
> p_\tau(w=j)
> =
> \frac{\exp(z_j/\tau)}{\sum_{\ell\in\mathcal V}\exp(z_\ell/\tau)},
> \qquad \tau>0.
> $$
>
> Explain how decreasing or increasing $\tau$ changes the next-token distribution. Then explain what top-$k$ sampling changes before a token is sampled. Relate both choices to the differences you observed in Problem 7.4.

> [!solution]- Solution
>
> For two tokens $i$ and $j$,
>
> $$
> \frac{p_\tau(i)}{p_\tau(j)}
> =
> \exp\!\left(\frac{z_i-z_j}{\tau}\right).
> $$
>
> Decreasing $\tau$ magnifies logit differences and assigns more probability to the largest logits. If the largest logit is unique, the distribution approaches greedy selection as $\tau\to0$. Increasing $\tau$ flattens the distribution; as $\tau\to\infty$, it approaches the uniform distribution over the vocabulary.
>
> Top-$k$ keeps the $k$ largest logits, sets all others to $-\infty$, and renormalizes over the remaining tokens. Temperature changes all nonzero probabilities, whereas top-$k$ gives the removed tokens probability zero. This matches Problem 7.4: lower temperature and smaller $k$ select from high-logit continuations more often, while higher temperature and no cutoff allow a wider range of characters.
