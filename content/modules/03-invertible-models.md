---
title: "Module 3: Invertible Models"
description: "Invertible models, normalizing flows, likelihoods, and practical flow architectures."
publish: true
---

<!-- prettier-ignore-start -->

## Invertible Models and Normalizing Flows

Let $\mathcal{D}=\{x_i\}_{i=1}^n$ be samples from an unknown distribution $P^\star$ on $\mathbb{R}^d$. A generative model specifies a measurable map $$X = T_\theta(\xi),\qquad \xi\sim \pi_0,$$ where $\pi_0$ is a simple base distribution (e.g., standard Gaussian), and $T_\theta:\mathbb{R}^d\to\mathbb{R}^d$ is a neural network parameterized by $\theta$.

The standard learning principle is maximum likelihood. If $p_\theta$ denotes the density of $X = T_\theta(\xi)$ (when it exists), then we estimate $\theta$ by $$\hat\theta \in \operatorname*{arg\,max}_{\theta}\;\hat\ell(\theta),
\qquad
\hat\ell(\theta):= \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i).$$

For a generic simulator, this density may be hard to evaluate: we can sample $X$, but we may not know how much probability mass the simulator puts near a particular data point $x$. Invertible models are the main tractable exception. If $T_\theta:\mathbb{R}^d\to\mathbb{R}^d$ is invertible and continuously differentiable, the density follows from the change-of-variables formula.

Assume $T_\theta:\mathbb{R}^d\to\mathbb{R}^d$ is **invertible** and continuously differentiable, with Jacobian $\nabla T_\theta(\cdot)$. Let $x = T_\theta(\xi)$, where $\xi \sim \pi_0$ with density $\pi_0(\xi)$. Then the density of $X$ is

$$
\begin{aligned}
p_\theta(x)
&= \pi_0\bigl((T_\theta)^{-1}(x)\bigr)\;
\Bigl|\det\bigl(\nabla_x (T_\theta)^{-1}(x)\bigr)\Bigr|.
\end{aligned}
$$

Here, $\nabla_x (T_\theta)^{-1}(x) = [\partial_{x_j} (T_\theta)^{-1}(x)_i]$ is the $\mathbb{R}^d \times \mathbb{R}^d$ Jacobian of the inverse map. The base-density term evaluates how likely the preimage $\xi = (T_\theta)^{-1}(x)$ is under $\pi_0$; the determinant corrects for local volume change. If the map expands volume near $\xi$, density decreases after transformation, and vice versa.

> [!remark] Remark
> For an invertible function, we have $$\nabla_x (T_\theta)^{-1}(x) = \bigl(\nabla_\xi T_\theta(\xi)\bigr)^{-1},$$ where $\xi = (T_\theta)^{-1}(x)$. Hence, we can also write $$p_\theta(x)
> = \pi_0\bigl((T_\theta)^{-1}(x)\bigr)\;
> \Bigl|\det\bigl(\nabla_\xi T_\theta(\xi)\bigr)\Bigr|^{-1}, \qquad \text{with } \xi = (T_\theta)^{-1}(x).$$

> [!proof] Proof
> Recall that $q(x)$ is the density of a random variable $X$ if and only if, for every measurable test function $h$, $$\mathbb{E}[h(X)] = \int q(x)h(x)\,\mathrm{d}x.$$ We compute $\mathbb{E}[h(X)]$ and then read off the density: $$\begin{aligned}
> \mathbb{E}[h(X)]
> &= \mathbb{E}_{\xi\sim \pi_0}[h(T_\theta(\xi))] \\
> &= \int h(T_\theta(\xi))\, \pi_0(\xi)\,\mathrm{d}\xi \\
> &\overset{\xi = (T_\theta)^{-1}(x)}{=} \int h(x)\, \pi_0((T_\theta)^{-1}(x))\, \left|\det\left(\nabla_x (T_\theta)^{-1}(x)\right)\right|\,\mathrm{d}x.
> \end{aligned}$$ The last step uses the integration rule $\mathrm{d}\xi = \left|\det(\nabla_x T_\theta^{-1}(x))\right|\,\mathrm{d}x$. ◻

> [!example]
> Assume $\xi \sim \mathcal{N}(0, I_d)$ and define the affine transformation $$X = T(\xi) = W\xi + \mu,$$ where $\theta = \{\mu, W\}$, with $\mu \in \mathbb{R}^d$ and $W \in \mathbb{R}^{d \times d}$ an invertible matrix. Then $T^{-1}(x) = W^{-1}(x - \mu)$, and $\nabla T^{-1}(x) = W^{-1}$. By the change of variables formula, $$\begin{aligned}
> p_{\mu,W}(x)
> &= \pi_0\!\left(W^{-1}(x - \mu)\right)\, \bigl|\det W^{-1}\bigr| \\
> &= \frac{1}{(2\pi)^{d/2}} \exp\Bigl(-\frac{1}{2} \bigl\|W^{-1}(x - \mu)\bigr\|^2\Bigr)\, \frac{1}{|\det W|}.
> \end{aligned}$$ Hence, $$\log p_{\mu,W}(x)
> = -\frac{1}{2} \bigl\|W^{-1}(x - \mu)\bigr\|^2 - \frac{d}{2} \log(2\pi) - \log|\det W|.$$ This is exactly the density function of a Gaussian random variable $X \sim \mathcal{N}(\mu, \Sigma)$ with covariance $\Sigma = WW^\top$.

##### MLE for Invertible Models

Consequently, the empirical log-likelihood becomes $$\begin{aligned}
\hat\ell(\theta)
&= \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i) \\
&= \frac{1}{n} \sum_{i=1}^n
\Bigl[
\log \pi_0\bigl(T_\theta^{-1}(x_i)\bigr)
+ \log \bigl|\det\bigl(\nabla_x T_\theta^{-1}(x_i)\bigr)\bigr|
  \Bigr].
  \end{aligned}$$

To make maximum likelihood practical, the architecture for $T_\theta$ should give us three things at once:

1.  $T_\theta$ is invertible for all allowed parameters $\theta$.

2.  Sampling $T_\theta(\xi)$ and density evaluation through $T_\theta^{-1}(x)$ are both efficient.

3.  The log-determinant $\log|\det J|$ and its gradients are cheap and numerically stable.

## Normalizing Flows

Normalizing flows build $T_\theta$ by composing small invertible transforms whose inverses and log-determinants are easy: $$T_\theta = T_{K,\theta} \circ T_{K-1,\theta} \circ \dots \circ T_{1,\theta}.$$ A single block is usually too limited; the composition is what lets the model gradually reshape a simple base distribution into a complicated data distribution.

The Jacobian of the composition can be written explicitly as a matrix product: $$\nabla_\xi T_\theta(\xi)
= \nabla T_{K,\theta}(z_{K-1}) \cdot \nabla T_{K-1,\theta}(z_{K-2}) \cdot \dots \cdot \nabla T_{1,\theta}(z_0),$$ where $z_0 = \xi$ and $z_k = T_{k,\theta}(z_{k-1})$ for $k = 1, \dots, K$.

Correspondingly, the determinant is a product of individual determinants: $$\det\left(\nabla_\xi T_\theta(\xi)\right) = \prod_{k=1}^K \det\left(\nabla T_{k,\theta}(z_{k-1})\right).$$

Hence, for a data point $x=z_K$, the log-likelihood is $$\log p_\theta(x)
= \log \pi_0(z_0) - \sum_{k=1}^K \log \left|\det\left(\nabla T_{k,\theta}(z_{k-1})\right)\right|,$$ where $z_0 = \xi = T_\theta^{-1}(x)$ and $z_k = T_{k,\theta}(z_{k-1})$. In practice, likelihood evaluation means running the data backward to recover $z_0$, evaluating the base density, and adding the layer-wise log-determinant corrections.

The main design question is therefore local: how should each block be made invertible without making the determinant expensive?

##### Triangular Maps

One common design is based on triangular maps. Let $x = [x_1, \dots, x_d] \in \mathbb{R}^d$. A triangular map takes the form $$\begin{aligned}
x_1' &= T_1(x_1),\\
x_2' &= T_2(x_1, x_2),\\
&\;\;\vdots\\
x_d' &= T_d(x_1, \dots, x_{d-1}, x_d),
\end{aligned}$$ where each $T_k$ is invertible in its last argument. The Jacobian is lower triangular: $$\nabla_x T(x) =
\begin{bmatrix}
\frac{\partial x_1'}{\partial x_1} & 0 & \cdots & 0 \\

\ast & \frac{\partial x_2'}{\partial x_2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
\ast & \ast & \cdots & \frac{\partial x_d'}{\partial x_d}
  \end{bmatrix},$$ so $$\left|\det\left(\nabla T(x)\right)\right| = \prod_{k=1}^d \left|\frac{\partial x_k'}{\partial x_k}\right|,$$ and the log-determinant becomes a cheap sum of elementwise terms. This structure underlies masked autoregressive flows and related coupling-based designs.

##### Additive Coupling Layers

Another useful design is a coupling layer. Split $x = (x_1, x_2)$ and update the two parts in sequence: $$\begin{aligned}
x_1' &= x_1 + F(x_2),\\
x_2' &= x_2 + G(x_1'),
\end{aligned}$$ where $F$ and $G$ are arbitrary neural networks.

The transformation is invertible via $$\begin{aligned}
x_2 &= x_2' - G(x_1'), \qquad
x_1 = x_1' - F(x_2).
\end{aligned}$$

The Jacobian has the block form $$\nabla T(x) =
\begin{bmatrix}
I & \nabla F(x_2)\\[2pt]
\nabla G(x_1') & I + \nabla G(x_1')\, \nabla F(x_2)
\end{bmatrix}.$$

It may not be immediately obvious, but the determinant of this Jacobian is always one. This follows from the block matrix determinant formula: $$\det\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
= \det(A) \cdot \det(D - C A^{-1} B),$$ which holds when $A$ is invertible. In our case, $A = I$, so $$\det(\nabla T(x)) = \det\big(I + \nabla G(x_1')\, \nabla F(x_2) - \nabla G(x_1')\, \nabla F(x_2)\big) = \det(I) = 1.$$

This is the basic idea behind reversible coupling blocks: $F$ and $G$ can be large neural networks, but the overall map is still exactly invertible and volume-preserving.

## Architectures with Efficient Log-Determinant

### Coupling Layers (NICE/RealNVP/Glow)

Partition the input $x=(x_A,x_B)$ using channels, checkerboard masks, or other masks. An affine coupling layer keeps one part fixed and uses it to scale and shift the other part: $$y_A = x_A,\qquad
y_B = x_B \odot \exp\!\big(s_\theta(x_A)\big) + t_\theta(x_A).$$ The subnetworks $s_\theta,t_\theta$ can be flexible because they only define the scale and shift, not the inverse itself. The Jacobian is block lower-triangular, so $$\log\Bigl|\det \nabla_x T_\theta(x)\Bigr|=\sum_j s_\theta(x_A)_j,$$ and inversion is elementwise: $$x_A=y_A,\qquad
x_B=\bigl(y_B-t_\theta(y_A)\bigr)\odot \exp\!\bigl(-s_\theta(y_A)\bigr).$$ Alternating masks or inserting permutations between coupling layers lets later blocks modify coordinates that earlier blocks left unchanged. The additive NICE layer $y_B=x_B+t_\theta(x_A)$ is the volume-preserving special case with $\log|\det|=0$.

##### Invertible $1\times1$ Convolution (Glow)

On images, Glow applies a learned invertible matrix $A\in\mathbb{R}^{C\times C}$ to the channel vector at each spatial location: $$y_{h,w,:}=A\,x_{h,w,:}.$$ If the feature map has height $H$ and width $W$, then $$\log|\det J|=H W\cdot \log|\det A|.$$ The matrix is typically stored through an LU-style parameterization, which makes the determinant stable and keeps inversion cheap. In Glow, this channel mixing is combined with ActNorm and multi-scale squeeze/factor-out operations.

### Autoregressive Flows (MAF/IAF)

Autoregressive parameterization yields a strictly triangular Jacobian. A common MAF transform is $$y_k=\frac{x_k-\mu_k(x_{1:k-1})}{\sigma_k(x_{1:k-1})},\qquad
\log\Bigl|\det J\Bigr|=-\sum_{k}\log \sigma_k(\cdot).$$ MAF gives fast likelihood evaluation because all $\mu_k,\sigma_k$ can be produced by one masked network pass, but sampling requires sequential inversion. IAF reverses this trade-off: sampling is parallel, while likelihood evaluation becomes sequential.

##### Monotone Spline Couplings (Neural Spline Flows)

Affine couplings can only scale and shift each transformed coordinate. Neural Spline Flows replace that coordinate-wise affine map with a monotone invertible spline, often rational--quadratic. The inverse and log-det remain closed-form, but each coupling layer can now express nonlinear one-dimensional warps.

## Likelihood, Dequantization, and Reporting

##### Exact Likelihood and Gradients

The gradient of the log-likelihood decomposes into a base-density term and a log-determinant term: $$\nabla_\theta \log p_\theta(x)=
\nabla_\theta \log \pi_0\!\big(T_\theta^{-1}(x)\big)
+\nabla_\theta \log\Bigl|\det \nabla_x T_\theta^{-1}(x)\Bigr|.$$ For coupling and autoregressive layers, this expression is differentiable through ordinary neural-network operations. There is no discriminator and no variational bound in the basic continuous case: the training objective is exact maximum likelihood.

##### Dequantization for Discrete Pixels

Images live on a discrete grid, while ordinary flows define continuous densities. The standard fix is dequantization: replace an integer pixel vector $x$ by $x+u$, with $u\sim\mathrm{Unif}[0,1)^d$. Jensen's inequality gives $$\log p_{\mathrm{disc}}(x) \ge \mathbb{E}_{u}\bigl[\log p_\theta(x+u)\bigr],$$ so the continuous model optimizes a lower bound on the discrete log-likelihood. Variational dequantization improves this by learning $q_\phi(u\mid x)$ instead of using uniform noise.

##### Bits-Per-Dimension (bpd)

Bits-per-dimension is the negative log-likelihood measured in bits per scalar dimension: $$\mathrm{bpd}(x)= -\frac{1}{d\log 2}\,\log p_\theta(x),$$ up to the usual constant introduced by pixel scaling/dequantization. Lower bpd means better likelihood, and the normalization makes models comparable across image resolutions.

##### Conditional Flows

For $p_\theta(x\mid c)$, inject condition $c$ into $s_\theta,t_\theta$ (or autoregressive nets) via concatenation, FiLM, or attention. The transform remains invertible for each fixed $c$, so exact conditional likelihoods and fast conditional sampling are retained.

## Continuous-Time Flows and Transport View

##### CNF / Neural ODE

A continuous-time flow replaces a stack of discrete layers with an ODE: $$\frac{\mathrm{d}x_t}{\mathrm{d}t}=v_\theta(x_t,t),\qquad x_0\sim p_0,\quad x_1\sim p_1.$$ Along a trajectory, the instantaneous change-of-variables formula is $$\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(x_t)= -\,\mathrm{div}_x\, v_\theta(x_t,t),$$ so if the trajectory ending at $x_1=x$ starts at $x_0=z$, then $$\log p_1(x)=\log p_0(z)-\int_0^1 \mathrm{div}\, v_\theta(x_t,t)\,\mathrm{d}t.$$ The divergence can be estimated with Hutchinson's trace estimator, avoiding explicit Jacobian matrices. The price is numerical integration: accuracy and runtime both depend on the ODE solver and its tolerances.

##### Positioning vs. Diffusion/Score Models

Discrete flows give exact likelihoods, exact inverses, and one-shot sampling, but their expressivity is constrained by the choice of invertible layer. CNFs remove the layer-by-layer restriction and learn a continuous velocity field, at the cost of ODE solves. Diffusion and score models also describe transport through time, but usually trade exact likelihood and one-step sampling for easier training and stronger sample quality. The common theme is mass transport; the difference is how the transport is parameterized and computed.

## Practical Design and Stability

##### Stable Parameterizations

- Keep scale outputs bounded, for example with $\tanh$ or clamping, so $\exp s_\theta(\cdot)$ cannot explode.

- Use ActNorm or data-dependent affine initialization to avoid a badly scaled first few optimization steps.

- Mix coordinates between coupling layers with permutations or invertible $1\times1$ convolutions; otherwise some dimensions may be updated too indirectly.

- In image flows, use squeeze and factor-out operations to move between spatial and channel structure and to shorten long dependencies.

##### Diagnostics

Useful diagnostics are usually simple: bpd curves, histograms of $\log|\det J|$, and the range of scale outputs. Very negative log-determinants, saturated scales, or sudden bpd spikes often point to the exact subnetwork that needs rescaling, regularization, or a smaller learning rate.

> [!remark] Remark
> Maximum likelihood is mode-covering: it heavily penalizes under-estimating density on data regions. This complements adversarial (often mode-seeking) training and partly explains empirical differences in sample diversity.

## Limitations and Trade-offs

##### Dimensionality and Discreteness

A standard flow is a bijection, so the input and output dimensions must match. Discrete data also needs care: either use specialized discrete invertible layers or relax the data into a continuous space through dequantization.

##### Expressivity vs. Efficiency

The tractable determinant is not free. Coupling and autoregressive layers restrict what each layer can do, then recover flexibility through depth, masking, mixing, and spline nonlinearities. Better expressivity usually means more layers, more memory, or slower sampling/evaluation.

##### MAF/IAF Speed Asymmetry

Use MAF when density evaluation is the main workload, such as anomaly detection or likelihood-based modeling. Use IAF when fast sampling is more important, such as generation or latent-variable inference.

## Key Derivations

##### Change-of-Variables in Log Form

Let $x=T_\theta(z)$, where $z\sim \pi_0$ and $T_\theta$ is invertible. The most useful form of the change-of-variables formula is the log-density form $$\log p_\theta(x)
= \log \pi_0(z) - \log\left|\det \nabla_z T_\theta(z)\right|,\qquad z=T_\theta^{-1}(x).$$

This equation is the accounting rule for every flow model. The first term asks whether the inverse image $z$ is plausible under the base distribution. The second term corrects for local volume change: if $T_\theta$ expands a small region around $z$, the same probability mass is spread over more volume in $x$-space, so the density goes down.

For a composition $T=T_K\circ\cdots\circ T_1$, with $z_k=T_k(z_{k-1})$ and $z_K=x$, the determinant becomes a product, so the log-determinant becomes a sum: $$\log p_\theta(x)
= \log \pi_0(z_0)-\sum_{k=1}^K \log\left|\det \nabla T_k(z_{k-1})\right|.$$ This is why flows are built from layers with cheap log-determinants.

##### Continuous-Time Limit

For a continuous-time flow, the state follows $$\frac{\mathrm{d}x_t}{\mathrm{d}t}=v_\theta(x_t,t).$$ Over a tiny interval $\Delta t$, the map is approximately $$x_{t+\Delta t}\approx x_t+v_\theta(x_t,t)\Delta t.$$ Its Jacobian is approximately $$I+\Delta t\,\nabla_x v_\theta(x_t,t),$$ so $$\log\left|\det\bigl(I+\Delta t\,\nabla_x v_\theta(x_t,t)\bigr)\right|
\approx \Delta t\,\operatorname{tr}\bigl(\nabla_x v_\theta(x_t,t)\bigr)
= \Delta t\,\nabla\!\cdot v_\theta(x_t,t).$$

Thus the log-density changes according to $$\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(x_t)
=-\nabla\!\cdot v_\theta(x_t,t),$$ and integrating along the trajectory gives $$\log p_1(x_1)=\log p_0(x_0)-\int_0^1 \nabla\!\cdot v_\theta(x_t,t)\,\mathrm{d}t.$$ In discrete flows we design triangular or block-triangular Jacobians. In continuous flows we instead estimate a trace/divergence term, often with Hutchinson's estimator.

## Log-Det Patterns

##### Affine Coupling

An affine coupling layer splits $x=(x_A,x_B)$ and defines $$y_A=x_A,\qquad
y_B=x_B\odot \exp s_\theta(x_A)+t_\theta(x_A).$$ Its inverse is elementwise once $y_A$ is known: $$x_A=y_A,\qquad
x_B=\bigl(y_B-t_\theta(y_A)\bigr)\odot \exp\bigl(-s_\theta(y_A)\bigr).$$

The Jacobian has the block form $$\nabla_x y=
\begin{bmatrix}
I & 0\\
\ast & \operatorname{diag}\bigl(\exp s_\theta(x_A)\bigr)
\end{bmatrix}.$$ The $\ast$ block can be complicated, but it does not affect the determinant because the matrix is block triangular. Therefore $$\log|\det \nabla_x y|
=\sum_j s_\theta(x_A)_j.$$ This is the main trick in RealNVP/Glow-style flows: the neural network can be expressive, while the determinant stays cheap.

##### Autoregressive Flow

For a MAF-style transform, $$y_k=\frac{x_k-\mu_k(x_{1:k-1})}{\sigma_k(x_{1:k-1})}.$$ The key constraint is that $\mu_k$ and $\sigma_k$ depend only on earlier coordinates. Hence $$\frac{\partial y_k}{\partial x_j}=0\qquad\text{for }j>k,$$ so the Jacobian is lower triangular. The diagonal entries are $$\frac{\partial y_k}{\partial x_k}=\frac{1}{\sigma_k(x_{1:k-1})},$$ which gives $$\log|\det J|
=-\sum_k \log \sigma_k(x_{1:k-1}).$$

This triangular structure is also the source of the speed trade-off. In MAF, all parameters can be produced by one masked-network pass for likelihood evaluation, but sampling must recover coordinates sequentially. IAF reverses the direction so sampling is fast and likelihood evaluation is sequential.

##### Invertible 1x1 Convolution

For an image feature map $x\in\mathbb{R}^{H\times W\times C}$, Glow applies the same invertible channel-mixing matrix $A\in\mathbb{R}^{C\times C}$ at every spatial position: $$y_{h,w,:}=A\,x_{h,w,:}.$$ If the feature map is flattened over all pixels and channels, the full Jacobian is block diagonal with $H W$ identical blocks. Therefore $$\det J=(\det A)^{H W},\qquad
\log|\det J|=H W\log|\det A|.$$ This layer is small but important: coupling layers update only part of the variables at a time, while the invertible 1x1 convolution mixes channels between coupling layers.

## Further Reading (Pointers)

For the classical architecture story, a natural reading order is: [NICE](https://arxiv.org/abs/1410.8516) for coupling layers, [RealNVP](https://arxiv.org/abs/1605.08803) for affine couplings on images, [Glow](https://arxiv.org/abs/1807.03039) for invertible 1x1 convolutions and multi-scale image flows, [MAF](https://arxiv.org/abs/1705.07057) / [IAF](https://arxiv.org/abs/1606.04934) for the density-evaluation versus sampling-speed trade-off, [Neural Spline Flows](https://arxiv.org/abs/1906.04032) for more expressive monotone coordinate transforms, and [FFJORD](https://arxiv.org/abs/1810.01367) for continuous-time flows.

Recent exact-flow image generation revisits the same likelihood objective with newer scaling tools: autoregressive Transformers, latent image spaces, guidance, and sampling refinements.

- [Normalizing Flows are Capable Generative Models](https://arxiv.org/abs/2412.06329) introduces TarFlow, a Transformer-based autoregressive flow over image patches, and reports strong image likelihood and generation results.

- [JetFormer](https://arxiv.org/abs/2411.19722) uses a normalizing flow as an invertible image representation inside a decoder-only multimodal model, connecting image likelihood modeling with text-image generation.

- [STARFlow](https://arxiv.org/abs/2506.06276) scales TarFlow in latent image space for high-resolution class-conditional and text-conditional synthesis.

- [FARMER](https://arxiv.org/abs/2510.23588) explores the pixel-space route by placing an invertible autoregressive flow before a sequence model, preserving an exact-likelihood path for raw-pixel generation.

- [SimFlow](https://arxiv.org/abs/2512.04084) simplifies latent-flow training by jointly training the representation and the flow.

- [Normalizing Flows with Iterative Denoising](https://arxiv.org/abs/2604.20041) keeps likelihood-based training and adds an iterative refinement step during sampling.

- [STARFlow2](https://arxiv.org/abs/2605.08029) studies unified multimodal generation, placing text generation and image generation under a shared causal Transformer-style interface.

<!-- prettier-ignore-end -->

## Homework

[Homework 3: Invertible Models and Normalizing Flows](/homework/03-invertible-models)
