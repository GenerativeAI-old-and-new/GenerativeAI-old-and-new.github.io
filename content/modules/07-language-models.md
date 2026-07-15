---
title: "Module 7: Language Models"
description: "Autoregressive language models, causal Transformers, next-token training, and text generation."
publish: true
---

<!-- prettier-ignore-start -->

## Language Modeling

Text is represented as a sequence of discrete tokens

$$
W=(w_1,w_2,\ldots,w_L),
\qquad
w_i\in\mathcal V,
$$

where $\mathcal V$ is a finite vocabulary. A token may be a character, a word, or, more commonly in large language models, a subword unit. Character tokenization uses a small vocabulary but produces long sequences; subword tokenization makes the opposite tradeoff.

> [!example|Character and Subword Tokenization]
> A character tokenizer represents `unhelpful` as nine tokens: `u`, `n`, `h`, `e`, `l`, `p`, `f`, `u`, `l`. A subword tokenizer might produce `un`, `help`, `ful`. The exact split depends on the tokenizer and its vocabulary.

Special tokens can mark structure that is not contained in ordinary text. A beginning-of-sequence token provides context for predicting the first token, and an end-of-sequence token lets the model assign probability to when a sequence stops.

A language model assigns a probability to a token sequence, or equivalently, a next-token distribution to every prefix.

## Autoregressive Factorization

The probability chain rule gives

$$
\begin{aligned}
p_\theta(w_1,\ldots,w_L)
&=
p_\theta(w_1)
p_\theta(w_2\mid w_1)
\cdots
p_\theta(w_L\mid w_{1:L-1}) \\
&=
\prod_{i=1}^{L}
p_\theta(w_i\mid w_{<i}).
\end{aligned}
$$

Here $w_{<i}=(w_1,\ldots,w_{i-1})$; for $i=1$, the context is empty or contains a start token. The factorization is exact. The modeling choice is how to parameterize each conditional distribution, usually with one neural network shared across positions.

Suppose the tokens are $(\mathtt{BOS},\mathtt{how},\mathtt{are},\mathtt{you},\mathtt{EOS})$. Conditioning on the start token,

$$
\begin{aligned}
p(\mathtt{how},\mathtt{are},\mathtt{you},\mathtt{EOS}\mid\mathtt{BOS})
&=
p(\mathtt{how}\mid\mathtt{BOS}) \\
&\quad\cdot p(\mathtt{are}\mid\mathtt{BOS},\mathtt{how}) \\
&\quad\cdot p(\mathtt{you}\mid\mathtt{BOS},\mathtt{how},\mathtt{are}) \\
&\quad\cdot p(\mathtt{EOS}\mid\mathtt{BOS},\mathtt{how},\mathtt{are},\mathtt{you}).
\end{aligned}
$$

Generation follows the same factorization: sample one token from $p_\theta(\cdot\mid w_{<i})$, append it to the prefix, and repeat.

<figure class="image-figure">
  <img src="/assets/modules/07-language-models/auto_llm.png" alt="An autoregressive sequence model predicting one output token after another from a start token." />
  <figcaption>An autoregressive model predicts each token using only the tokens to its left.</figcaption>
</figure>

## Next-Token Training

Maximum likelihood training minimizes the negative log-likelihood

$$
\mathcal L(\theta)
=
-\mathbb E_{W\sim P_{\mathrm{data}}}
\left[
  \sum_{i=1}^{L}
  \log p_\theta(w_i\mid w_{<i})
\right].
$$

Training examples are formed by shifting a token sequence by one position. Given a chunk $(s_0,s_1,\ldots,s_T)$, use

$$
x=(s_0,s_1,\ldots,s_{T-1}),
\qquad
y=(s_1,s_2,\ldots,s_T).
$$

At position $i$, the model sees $x_{\le i}=(s_0,\ldots,s_i)$ and predicts $y_i=s_{i+1}$. If the sequence begins with a special start token, this same shift also trains the model to predict the first ordinary token.

> [!example|Shifted Input and Target]
> For the chunk $(\mathtt{BOS},\mathtt{the},\mathtt{cat},\mathtt{sat},\mathtt{EOS})$,
>
> $$
> x=(\mathtt{BOS},\mathtt{the},\mathtt{cat},\mathtt{sat}),
> \qquad
> y=(\mathtt{the},\mathtt{cat},\mathtt{sat},\mathtt{EOS}).
> $$
>
> One forward pass trains four predictions:
>
> 1.  $(\mathtt{BOS})\rightarrow\mathtt{the}$
> 2.  $(\mathtt{BOS},\mathtt{the})\rightarrow\mathtt{cat}$
> 3.  $(\mathtt{BOS},\mathtt{the},\mathtt{cat})\rightarrow\mathtt{sat}$
> 4.  $(\mathtt{BOS},\mathtt{the},\mathtt{cat},\mathtt{sat})\rightarrow\mathtt{EOS}$

### Logits and Cross-Entropy

For every position $i$, the network produces logits $z_i\in\mathbb R^{|\mathcal V|}$. The softmax converts them into a categorical distribution:

$$
p_\theta(w_{i+1}=k\mid w_{\le i})
=
\frac{\exp(z_{i,k})}
{\sum_{\ell\in\mathcal V}\exp(z_{i,\ell})}.
$$

The token-level cross-entropy is

$$
\ell_i
=
-\log p_\theta(y_i\mid x_{\le i}).
$$

Summing or averaging $\ell_i$ across tokens and batches gives the training loss. Perplexity is the exponential of the average token loss,

$$
\operatorname{PPL}=\exp\!\left(\frac{1}{N}\sum_{i=1}^{N}\ell_i\right).
$$

Perplexity comparisons require the same tokenizer and evaluation data. Character-level and subword-level perplexities are not directly comparable.

## Decoder-Only Transformers

A GPT-style language model is a decoder-only Transformer. For a batch of token indices $x\in\{0,\ldots,|\mathcal V|-1\}^{B\times T}$, the initial hidden states are

$$
H^{(0)}_{b,i}
=
E_{\mathrm{tok}}[x_{b,i}]
+
E_{\mathrm{pos}}[i],
\qquad
H^{(0)}\in\mathbb R^{B\times T\times d}.
$$

$E_{\mathrm{tok}}$ encodes the token identity, while $E_{\mathrm{pos}}$ encodes its location in the context. In nanoGPT, both are learned embedding tables. The sequence length must satisfy $T\le T_{\max}$, where $T_{\max}$ is the model's configured context length.

### Causal Self-Attention

For one attention head, linear projections produce

$$
Q=HW_Q,
\qquad
K=HW_K,
\qquad
V=HW_V.
$$

With head dimension $d_h$, scaled dot-product attention is

$$
A
=
\frac{QK^\top}{\sqrt{d_h}},
\qquad
P=\operatorname{softmax}(A+M),
\qquad
O=PV.
$$

The softmax is applied to each row, so each row of $P$ is a probability distribution over source positions.

The additive causal mask is

$$
M_{ij}
=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
$$

For a length-four sequence, its allowed attention pattern is lower triangular:

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}.
$$

The binary matrix above shows which positions are allowed; the additive mask $M$ uses $0$ and $-\infty$. Row $i$ can use positions $0$ through $i$, but no future position. For $(\mathtt{The},\mathtt{cat},\mathtt{sat},\mathtt{.})$, the representation at $\mathtt{sat}$ may attend to $\mathtt{The}$, $\mathtt{cat}$, and itself, but not to the period.

After softmax, every entry of $P$ above the diagonal is zero. The factor $1/\sqrt{d_h}$ keeps the scale of the attention logits roughly stable as the head dimension grows.

Each attention head has its own query, key, and value projections. With $h$ heads and $d_h=d/h$, the projected tensors are reshaped to $(B,h,T,d_h)$. The head outputs are concatenated to shape $(B,T,d)$ and passed through an output projection.

| Tensor | Shape | Role |
| --- | --- | --- |
| Token indices | $(B,T)$ | Input and shifted targets |
| Hidden states | $(B,T,d)$ | Representation at each position |
| $Q,K,V$ | $(B,h,T,d_h)$ | Per-head queries, keys, and values |
| Attention probabilities $P$ | $(B,h,T,T)$ | Causal mixing across positions |
| Logits | $(B,T,\lvert\mathcal V\rvert)$ | Next-token scores |

### The GPT Block

nanoGPT uses a pre-normalization block. Given hidden states $H$, one block computes

$$
\begin{aligned}
U
&=
H+\operatorname{Attention}(\operatorname{LayerNorm}(H)),\\
H'
&=
U+\operatorname{MLP}(\operatorname{LayerNorm}(U)).
\end{aligned}
$$

The MLP acts independently at each position and shares parameters across positions. In nanoGPT, it maps $d$ to $4d$, applies GELU, and maps $4d$ back to $d$. Layer normalization is applied before each sublayer, and each residual connection adds the sublayer output to its input.

After $L$ blocks, a final layer normalization and linear vocabulary head produce the logits:

$$
H^{(0)}
\longrightarrow
\operatorname{Block}^{\times L}
\longrightarrow
\operatorname{LayerNorm}
\longrightarrow
Z\in\mathbb R^{B\times T\times|\mathcal V|}.
$$

The vocabulary head can share its weight matrix with the token embedding table, a parameter-sharing choice known as weight tying.

For a compact implementation, see nanoGPT's [`CausalSelfAttention`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L29-L76), [`Block`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L78-L106), and [`GPT.forward`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L118-L193).

## Parallel Training and Sequential Generation

During training, the entire token sequence is known. The causal mask prevents information leakage, while matrix operations compute the logits for all positions in parallel. This is often called teacher forcing: each position receives the true preceding tokens rather than tokens sampled from the model.

Generation is sequential because token $w_{i+1}$ is not available until it has been sampled from the distribution predicted at position $i$. A minimal generation loop is:

1.  Keep the most recent $T_{\max}$ tokens from the current sequence.
2.  Run the model and take the logits at the final position.
3.  Select a token from the final logits, using greedy decoding or sampling.
4.  Append the selected token to the sequence.
5.  Repeat until the desired length or an end-of-sequence token is reached.

The simple nanoGPT loop recomputes the retained prefix at every step. A KV cache stores keys and values from earlier positions; the model then computes new projections only for the latest token and attends to the cached history.

## Decoding

The model defines next-token probabilities, but generation also requires a rule for selecting a token.

Greedy decoding always selects the largest-logit token. It is deterministic, but choosing the best token at each step does not guarantee the most probable complete sequence. Sampling instead draws from the next-token distribution, allowing different continuations from the same prefix. Temperature and top-$k$ modify this distribution before sampling.

### Temperature

For temperature $\tau>0$,

$$
p_\tau(w=j)
=
\frac{\exp(z_j/\tau)}
{\sum_{\ell\in\mathcal V}\exp(z_\ell/\tau)}.
$$

$\tau<1$ sharpens the distribution and favors high-logit tokens. $\tau>1$ flattens it and increases randomness. Temperature changes the sampling distribution; it does not retrain the model.

### Top-$k$ Sampling

Top-$k$ sampling keeps the $k$ largest logits, sets every other logit to $-\infty$, and then applies softmax. Tokens outside the top $k$ receive zero probability. If $k\ge|\mathcal V|$, top-$k$ has no effect.

> [!example|Temperature and Top-k]
> Suppose three tokens have logits $(2,1,0)$. Their approximate probabilities are
>
> - $\tau=1$: $(0.67,0.24,0.09)$,
> - $\tau=0.5$: $(0.87,0.12,0.02)$,
> - $\tau=2$: $(0.51,0.31,0.19)$.
>
> Lower temperature concentrates probability on the first token; higher temperature spreads it across the three choices. With $\tau=1$ and $k=2$, top-$k$ removes the third token and gives probabilities $(0.73,0.27,0)$.

Generation should use evaluation mode and disabled gradients. nanoGPT's complete loop, including context cropping, temperature, and top-$k$, is shown in [`GPT.generate`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L305-L330).

## Pretraining and Instruction Tuning

GPT pretraining applies the next-token objective to a large text corpus. Instruction tuning keeps the same architecture and trains on prompt-response sequences; the loss is often computed only on the response tokens. Generation remains autoregressive in both cases.

## Homework

[Homework 7: Autoregressive Language Models](/homework/07-language-models)

<!-- prettier-ignore-end -->
