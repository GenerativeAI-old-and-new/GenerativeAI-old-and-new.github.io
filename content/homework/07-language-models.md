---
title: "Homework 7: Autoregressive Language Models"
description: "Homework problems covering next-token prediction, causal self-attention, small GPT training, and text generation."
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

> [!problem|Autoregressive Sampling]
> Implement nanoGPT's autoregressive sampling loop and run it with `model.eval()` inside `torch.no_grad()`. At each step, keep only the most recent $T$ tokens if the sequence is longer than the context length, run the model, and select the logits from the final position. Divide the logits by `temperature`, optionally set all but the largest `top_k` logits to $-\infty$, apply softmax, and sample one token with `torch.multinomial`.
>
> **nanoGPT reference:** [`GPT.generate`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L305-L330); [sampling settings](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/sample.py#L12-L19); [prompt encoding and generation](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/sample.py#L51-L88).
>
> Use the same checkpoint and prompt to generate text at two different temperatures, for example $0.7$ and $1.2$. Then fix the temperature and compare generation with and without a `top_k` cutoff. Choose `top_k` smaller than the vocabulary size; values such as 10 or 20 work for Tiny Shakespeare. Show representative samples and briefly describe how the generated text changes. Set the same random seed before each run so that the comparison is easier to interpret.

## Theory

Let $(w_1,\ldots,w_L)$ be a token sequence with $w_i\in\mathcal V$. An autoregressive language model represents

$$
p_\theta(w_1,\ldots,w_L)
=
\prod_{i=1}^{L}p_\theta(w_i\mid w_{<i}).
$$

> [!problem|Next-Token Training Objective]
> Write the negative log-likelihood of the sequence $(w_1,\ldots,w_L)$. If the training input is $(w_1,\ldots,w_{L-1})$ and the labels are $(w_2,\ldots,w_L)$, which terms of the negative log-likelihood are included by the shifted cross-entropy loss? Explain how prepending a beginning-of-sequence token allows the model to include the term for $w_1$ as well.

> [!problem|Causal Masking and Parallel Training]
> For a sequence of length four, write the $4\times4$ attention mask, using one when position $i$ may attend to position $j$ and zero otherwise. Explain why entries above the diagonal must be masked. Since the whole training sequence is available, why can the model still compute the outputs at all four positions in one forward pass?

> [!problem|Why Attention Logits Are Scaled]
> Let $q,k\in\mathbb R^{d_h}$ be a query and key whose coordinates are independent, with mean zero and variance one. Compute the variance of $q^\top k$. Use the result to explain why attention divides this dot product by $\sqrt{d_h}$ before applying softmax.

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
