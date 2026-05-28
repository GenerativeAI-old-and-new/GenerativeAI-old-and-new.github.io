---
title: "Module 3: Invertible Models"
description: "Invertible models, normalizing flows, likelihoods, and practical flow architectures."
publish: true
---

<!-- prettier-ignore-start -->

## Invertible Models and Normalizing Flows

Let $\mathcal{D}=\{x_i\}_{i=1}^n$ be samples from an unknown distribution $P^\star$ on $\mathbb{R}^d$. A generative model specifies a measurable map $$X = T_\theta(\xi),\qquad \xi\sim \pi_0,$$ where $\pi_0$ is a simple base distribution (e.g., standard Gaussian), and $T_\theta:\mathbb{R}^d\to\mathbb{R}^d$ is a neural network parameterized by $\theta$.

The standard learning principle is maximum likelihood. If $p_\theta$ denotes the density of $X = T_\theta(\xi)$ (when it exists), then we estimate $\theta$ by $$\hat\theta \in \argmax_{\theta}\;\hat\ell(\theta),
\qquad
\hat\ell(\theta):= \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i).$$

However, when $T_\theta$ is implemented via a generic simulator rather than defined through a closed-form expression, computing $p_\theta$ can be difficult. A key tractable case is when $T_\theta:\mathbb{R}^d\to\mathbb{R}^d$ is an invertible and continuously differentiable map, in which case we can apply the change-of-variables formula.

Assume $T_\theta:\mathbb{R}^d\to\mathbb{R}^d$ is **invertible** and continuously differentiable, with Jacobian $\nabla T_\theta(\cdot)$. Let $x = T_\theta(\xi)$, where $\xi \sim \pi_0$ with density $\pi_0(\xi)$. Then the density of $X$ is

$$
\begin{aligned}
p_\theta(x)
&= \pi_0\bigl((T_\theta)^{-1}(x)\bigr)\;
\Bigl|\det\bigl(\nabla_x (T_\theta)^{-1}(x)\bigr)\Bigr|.
\end{aligned}
$$

Here, $\nabla_x (T_\theta)^{-1}(x) = [\partial_{x_j} (T_\theta)^{-1}(x)_i]$ is the $\mathbb{R}^d \times \mathbb{R}^d$ Jacobian matrix of the inverse mapping $(T_\theta)^{-1}$. The formula has two parts: the first term, $\pi_0((T_\theta)^{-1}(x))$, accounts for the change of variable via $\xi = (T_\theta)^{-1}(x)$, and the second term is a scaling factor introduced by the distortion of the mapping.

> [!remark] Remark
> For an invertible function, we have $$\nabla_x (T_\theta)^{-1}(x) = \bigl(\nabla_\xi T_\theta(\xi)\bigr)^{-1},$$ where $\xi = (T_\theta)^{-1}(x)$. Hence, we can also write $$p_\theta(x)
> = \pi_0\bigl((T_\theta)^{-1}(x)\bigr)\;
> \Bigl|\det\bigl(\nabla_\xi T_\theta(\xi)\bigr)\Bigr|^{-1}, \qquad \text{with } \xi = (T_\theta)^{-1}(x).$$

> [!proof] Proof
> _Proof._ Recall that a function $q(x)$ is the density of a random variable $X$ if and only if the following holds for all measurable functions $h$: $$\mathbb{E}[h(X)] = \int q(x) h(x)\, \,d x.$$ We compute $\mathbb{E}[h(X)]$ and express it in integral form to identify the density function of $X$: $$\begin{aligned}
> \mathbb{E}[h(X)]
> &= \mathbb{E}_{\xi\sim \pi_0}[h(T_\theta(\xi))] \\
> &= \int h(T_\theta(\xi))\, \pi_0(\xi)\, \,d \xi \\
> &\overset{\xi = (T_\theta)^{-1}(x)}{=} \int h(x)\, \pi_0((T_\theta)^{-1}(x))\, \left|\det\left(\nabla_x (T_\theta)^{-1}(x)\right)\right|\, \,d x.
> \end{aligned}$$ The last step uses the change-of-variables formula in integration: $\,d \Phi(x) = \left|\det(\nabla_x \Phi(x))\right|\, \,d x$. ◻

> [!example]
> Assume $\xi \sim \mathcal{N}(0, I_d)$ and define the affine transformation $$X = T(\xi) = W\xi + \mu,$$ where $\theta = \{\mu, W\}$, with $\mu \in \mathbb{R}^d$ and $W \in \mathbb{R}^{d \times d}$ an invertible matrix. Then $T^{-1}(x) = W^{-1}(x - \mu)$, and $\nabla T^{-1}(x) = W^{-1}$. By the change of variables formula, $$\begin{aligned}
> p_W(x)
> &= \pi_0\!\left(W^{-1}(x - \mu)\right)\, \bigl|\det W^{-1}\bigr| \\
> &= \frac{1}{(2\pi)^{d/2}} \exp\Bigl(-\frac{1}{2} \bigl\|W^{-1}(x - \mu)\bigr\|^2\Bigr)\, \frac{1}{|\det W|}.
> \end{aligned}$$ Hence, $$\log p_W(x)
> = -\frac{1}{2} \bigl\|W^{-1}(x - \mu)\bigr\|^2 - \frac{d}{2} \log(2\pi) - \log|\det W|.$$ This is exactly the density function of a Gaussian random variable $X \sim \mathcal{N}(\mu, \Sigma)$ with covariance $\Sigma = WW^\top$.

##### MLE for Invertible Models

Consequently, the empirical log-likelihood becomes $$\begin{aligned}
\hat\ell(\theta)
&= \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i) \\
&= \frac{1}{n} \sum_{i=1}^n
\Bigl[
\log \pi_0\bigl(T_\theta^{-1}(x_i)\bigr)

- \log \bigl|\det\bigl(\nabla_x T_\theta^{-1}(x_i)\bigr)\bigr|
  \Bigr].
  \end{aligned}$$

To make this MLE computationally feasible in practice, we aim to design the model architecture of $T_\theta$ (which is typically modeled as a neural network in modern generative modeling) such that:

1.  $T_\theta$ is invertible for all $\theta$, and both $T_\theta(x)$ and its inverse $(T_\theta)^{-1}(x)$ can be computed efficiently.

2.  The determinant of the Jacobian matrix, as well as its derivative, can be evaluated efficiently and stably.

## Normalizing Flows

One general approach to designing tractable, invertible models is to define the map $T_\theta$ as a composition of many simple, tractable, and invertible transforms: $$T_\theta = T_{K,\theta} \circ T_{K-1,\theta} \circ \dots \circ T_{1,\theta},$$ where each $T_{k,\theta} \colon \mathbb{R}^d \to \mathbb{R}^d$ is a transformation for which both the inverse and the Jacobian determinant can be computed efficiently.

The Jacobian of the composition can be written explicitly as a matrix product: $$\nabla_\xi T_\theta(\xi)
= \nabla T_{K,\theta}(z_{K-1}) \cdot \nabla T_{K-1,\theta}(z_{K-2}) \cdot \dots \cdot \nabla T_{1,\theta}(z_0),$$ where $z_0 = \xi$ and $z_k = T_{k,\theta}(z_{k-1})$ for $k = 1, \dots, K$.

Correspondingly, the determinant is a product of individual determinants: $$\det\left(\nabla_\xi T_\theta(\xi)\right) = \prod_{k=1}^K \det\left(\nabla T_{k,\theta}(z_{k-1})\right).$$

Hence, the log-likelihood is given by $$\log p_\theta(x)
= \log \pi_0(z_0) - \sum_{k=1}^K \log \left|\det\left(\nabla T_{k,\theta}(z_{k-1})\right)\right|,$$ where $z_0 =\xi = (T_\theta)^{-1}(x), z_k = T_{k,\theta}(z_{k-1})$. Therefore, maximum likelihood estimation reduces to evaluating the base density at $z_K$ and summing the log-determinants of the Jacobians across layers.

Each $T_{k,\theta}$ can be viewed as a building block. Different methods vary in how these blocks are designed. To ensure flexibility, we require a sufficient number of expressive blocks such that their composition can model complex distributions.

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

##### Additive Coupling Layers.

Another design is based on coupling layers. Split $x = (x_1, x_2)$ and define a transformation of the form: $$\begin{aligned}
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

This design is also known as a reversible residual layer. Its key advantage is that it permits the use of arbitrary functions $F$ and $G$ while maintaining tractable invertibility.

---

## Architectures with Efficient Log-Determinant

### Coupling Layers (NICE/RealNVP/Glow)

Partition the input $x=(x_A,x_B)$ (by channels, checkerboard, or masks), and define an affine coupling transform $$y_A = x_A,\qquad
y_B = x_B \odot \exp\!\big(s_\theta(x_A)\big) + t_\theta(x_A),$$ with flexible subnetworks $s_\theta,t_\theta$. The Jacobian is block lower-triangular: $$\log\Bigl|\det J_{f_\theta}(x)\Bigr|=\sum_j s_\theta(x_A)_j,$$ and inversion is closed-form: $$x_A=y_A,\qquad
x_B=\bigl(y_B-t_\theta(y_A)\bigr)\odot \exp\!\bigl(-s_\theta(y_A)\bigr).$$ Stacking multiple layers with alternating masks (and permutations) yields full-dimensional mixing. The additive special case $y_B=x_B+t_\theta(x_A)$ (NICE) is volume-preserving ($\log|\det|=0$) and extremely stable but less expressive per layer.

##### Invertible $1\times1$ Convolution (Glow)

On images, apply a learned invertible $W\in\mathbb{R}^{C\times C}$ to channels at each spatial location: $$y_{h,w,:}=W\,x_{h,w,:}.$$ Then $\log|\det J|=H\,W\cdot \log|\det W|.$ Parameterize $W$ via PLU/LU to guarantee invertibility and make $\log|\det W|$ stable. Combined with ActNorm (data-dependent affine normalization) and multi-scale "squeeze/factor-out", Glow achieves strong likelihoods with fast inversion.

### Autoregressive Flows (MAF/IAF)

Autoregressive parameterization yields a strictly triangular Jacobian. A common MAF forward transform is $$y_k=\frac{x_k-\mu_k(x_{1:k-1})}{\sigma_k(x_{1:k-1})},\qquad
\log\Bigl|\det J\Bigr|=-\sum_{k}\log \sigma_k(\cdot).$$ MAF offers fast density evaluation (one pass), but sampling requires sequential inversion. Inverse autoregressive flows (IAF) swap the roles to make sampling fast (parallel forward of a masked network) at the cost of slower likelihood evaluation. This MAF/IAF duality lets us pick the right trade-off for density estimation versus generation speed.

##### Monotone Spline Couplings (Neural Spline Flows)

Replacing the affine coordinate-wise map by a monotone, invertible spline (e.g., rational--quadratic) improves expressivity while preserving closed-form inverse and exact log-det. This often narrows the gap to more flexible generative families while retaining the computational advantages of couplings.

## Likelihood, Dequantization, and Reporting

##### Exact Likelihood and Gradients

By Eq. `eq:cov`, gradients decompose into a base-density term and a log-det term: $$\nabla_\theta \log p_\theta(x)=
\nabla_\theta \log p_Z\!\big(f_\theta(x)\big)
+\nabla_\theta \log\Bigl|\det J_{f_\theta}(x)\Bigr|.$$ Because coupling/autoregressive layers keep $\log|\det|$ analytic, flows train with standard first-order optimizers and are generally well-behaved.

##### Dequantization for Discrete Pixels

Images live on a discrete grid. To fit continuous flows, one dequantizes $x$ via $y=x+u$, $u\sim\mathrm{Unif}[0,1)^d$. Then Jensen's inequality shows $$\log p_{\mathrm{disc}}(x)\ \ge\
\mathbb{E}_{u}\bigl[\log p_\theta(x+u)\bigr],$$ so maximizing the RHS tightens a valid lower bound on the discrete log-likelihood. Variational dequantization further learns $q_\phi(u\mid x)$ to tighten the bound.

##### Bits-Per-Dimension (bpd)

We report bpd as $\mathrm{bpd}(x)= -\frac{1}{d\log 2}\,\log p_\theta(x)$ (with a dataset-specific constant for dequantization). This unit normalizes across resolutions and enables fair model comparisons.

##### Conditional Flows

For $p_\theta(x\mid c)$, inject condition $c$ into $s_\theta,t_\theta$ (or autoregressive nets) via concatenation, FiLM, or attention. The transform remains invertible for each fixed $c$, so exact conditional likelihoods and fast conditional sampling are retained.

## Continuous-Time Flows and Transport View

##### CNF / Neural ODE

A continuous-time flow evolves by an ODE $$\frac{d x_t}{dt}=v_\theta(x_t,t),\qquad x_0\sim p_0,\quad x_1\stackrel{d}{=}x,$$ and the instantaneous change-of-variables formula states $$\frac{d}{dt}\log p_t(x_t)= -\,\mathrm{div}_x\, v_\theta(x_t,t)
\quad\Rightarrow\quad
\log p_1(x)=\log p_0(z)-\int_0^1 \mathrm{div}\, v_\theta(x_t,t)\,dt.$$ The divergence can be stochastically estimated (Hutchinson trace), avoiding explicit Jacobians. CNFs offer fine-grained flexibility but require numerical integration; training and evaluation times thus hinge on solver tolerances.

##### Positioning vs. Diffusion/Score Models

Discrete flows: exact likelihoods, exact inverses, one-shot sampling; expressivity is governed by layer design. CNFs: flexible dynamics, exact likelihood via Eq. `eq:icov`, but integration cost. Diffusion/score: superb sample quality and simple training, yet likelihoods are inexact (or expensive) and sampling is multi-step. All can be unified under mass transport and continuity equations, differing in parameterization and numerical pathways.

## Practical Design and Stability

##### Stable Parameterizations

- Scale control: Bound $s_\theta(\cdot)$ (e.g., $\tanh$, clamping) to prevent exploding $\exp s$.

- Normalization: ActNorm or data-dependent affine initialization improves early stability.

- Permutation/mixing: Use invertible $1\times1$ conv or channel permutations between couplings.

- Multi-scale: Squeeze (space$\to$channels) and factor-out latents to shorten dependencies and ease optimization.

##### Diagnostics

Monitor the distribution of $\log|\det J|$, bpd curves, and intermediate activations. Pathologies (e.g., overly negative $\log|\det|$ or saturated scales) often pinpoint subnetworks that need regularization or rescaling.

> [!remark] Remark
> Maximum likelihood is mode-covering: it heavily penalizes under-estimating density on data regions. This complements adversarial (often mode-seeking) training and partly explains empirical differences in sample diversity.

## Limitations and Trade-offs

##### Dimensionality and Discreteness

A bijection demands equal input--output dimension; truly discrete variables necessitate specialized invertible discrete layers or relaxation via dequantization.

##### Expressivity vs. Efficiency

Couplings/autoregressive layers restrict per-layer transforms to keep $\log|\det|$ closed-form. Expressivity is then accrued via depth, mixing, and spline nonlinearity---each adds cost.

##### MAF/IAF Speed Asymmetry

Choose MAF when likelihood evaluation dominates (density modeling, anomaly detection); choose IAF when sampling speed is paramount (real-time generation, compression).

## Mathematical Underpinnings (Sketches)

##### Change-of-Variables

For a $C^1$ diffeomorphism $f_\theta$, any integrable $\phi$ satisfies $$\int \phi(x)\,p_\theta(x)\,dx=\int \phi\!\bigl(f_\theta^{-1}(z)\bigr)\,p_Z(z)\,dz,$$ yielding Eq. `eq:cov` by taking $\phi\equiv 1$ and then logs.

##### Instantaneous Formula

From the continuity equation $\partial_t p_t+\nabla\!\cdot(p_t v)=0$, evaluating along characteristics $x_t$ gives $\frac{d}{dt}\log p_t(x_t)=-\nabla\!\cdot v(x_t,t)$, and integrating over $t\in[0,1]$ gives Eq. `eq:icov`.

##### Knothe--Rosenblatt Rearrangement

For sufficiently regular $p_X,p_Z$, there exists a monotone triangular map $T$ with $T_\# p_X=p_Z$. Autoregressive and monotone-spline layers approximate such transport maps numerically, supporting the expressivity of triangular-Jacobian flows.

## Worked Log-Det Examples

##### Affine Coupling $\Rightarrow$ Sum of Scales

With Eq. `eq:affine-coupling`, the Jacobian is block lower-triangular with diagonal $\exp s_\theta(x_A)$, hence $\log|\det J|=\sum_j s_\theta(x_A)_j$ and inversion is elementwise.

##### MAF $\Rightarrow$ Sum of Log-Scales

For Eq. `eq:maf`, $\partial y_k/\partial x_k=\sigma_k^{-1}$ and $\partial y_k/\partial x_j=0\ (j>k)$. Thus $\log|\det J|=-\sum_k \log \sigma_k(\cdot).$

##### Invertible $1\times1$ Convolution

Treating each spatial location independently, $J$ is block-diagonal with $H\!W$ copies of $W$. Therefore $\log|\det J|=H\,W\cdot \log|\det W|$, computable efficiently via LU with stable sign handling.

## Further Reading (Pointers)

NICE/RealNVP (coupling; tractable log-det), Glow (invertible $1\times1$ conv; multi-scale), MAF/IAF (autoregressive duality), Neural Spline Flows (monotone splines), FFJORD/CNF (continuous-time with trace estimators). Each navigates the triangle of invertibility--expressivity--efficiency differently, and the right choice depends on whether likelihood accuracy, sampling speed, or representational power is the primary design goal.

<!-- prettier-ignore-end -->
