---
title: "Module 5: Autoencoder Models"
description: "Autoencoders, adversarial autoencoders, VAEs, ELBOs, and latent variable modeling."
publish: true
---

<!-- prettier-ignore-start -->

Recall that we aim to learn a mapping $X = T^\theta(Z)$ that transforms noise $Z$ into data-like samples $X$. The classical approach based on maximum likelihood estimation (MLE) requires $T^\theta$ to be invertible and have a tractable Jacobian, so that the change-of-variables formula can be computed explicitly. This restriction motivates specialized architectures such as normalizing flows, which guarantee invertibility but limit flexibility. Generative adversarial networks (GANs) lift this restriction by allowing arbitrary, non-invertible mappings $T^\theta$, but replace likelihood maximization with a challenging minimax game between a generator and a discriminator, often leading to instability in training.

We now turn to an alternative, autoencoder-based approach that relies on approximate invertibility. Instead of enforcing $T^\theta$ to be exactly invertible, this framework introduces an auxiliary neural network $E^\phi$ that approximates its inverse: $$E^\phi(x) \approx (T^\theta)^{-1}(x).$$ The two networks are trained jointly to achieve approximate reconstruction, $$T^\theta(E^\phi(x)) \approx x,$$ allowing $T^\theta$ to remain flexible and expressive while preserving a weak notion of invertibility sufficient for learning meaningful latent representations.

We start with introducing the idea of autoencoders, which is general idea of using pairs of encoders and decoders for representation, and then discuss how to use autoencoders to build generative models such as variational encoders and adverairal antocoders.

### Autoencoders: Non-Generative

_TikZ diagram omitted; see source notes for the original figure._

_Structure of an autoencoder. The encoder (left) compresses input data $X$ into a latent code $Z$, and the decoder (right) reconstructs $\hat X$ from it._

> [!example]
> In linear encoder--decoder models, we have $$Z = W_e X, \qquad \hat X = W_d Z, \qquad d_z < d_x,$$ with $W_e \in \mathbb{R}^{d_z \times d_x}$ and $W_d \in \mathbb{R}^{d_x \times d_z}$, where $d_x,d_z$ are the dimensions of $X$ and $Z$, respectively. The autoencoder minimizes the expected squared reconstruction error: $$\min_{W_e, W_d} \; \mathbb{E}\!\left[ \| X - W_d W_e X \|^2 \right].$$ Let $A = W_d W_e \in \mathbb{R}^{d_x \times d_x}$, which has rank at most $d_z$. Then the problem becomes $$\min_{A:\,\operatorname{rank}(A) \le d_z} \; \mathbb{E}\!\left[ \| X - A X \|^2 \right].$$ This optimization has a closed form solution related to PCA/SVD.
>
> Define $\Sigma := \mathbb{E}[X X^\top]$, and let $\Sigma = U \Lambda U^\top$ be the eigendecomposition of the covariance matrix, where $$\Lambda = \mathrm{diag}(\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_{d_x} \ge 0),
> \quad U = [u_1, \ldots, u_{d_x}].$$ By the Eckart--Young--Mirsky theorem, the optimal solution above is given by the orthogonal projector onto the top-$d_z$ eigenspace: $$A^\star = U_{1:d_z} U_{1:d_z}^\top,
> \qquad U_{1:d_z} := [u_1, \ldots, u_{d_z}].$$ A valid factorization achieving this $A^\star$ is obtained via: $$W_e^\star = U_{1:d_z}^\top,
> \qquad W_d^\star = U_{1:d_z},$$ leading to the reconstruction $$\hat X = U_{1:d_z} U_{1:d_z}^\top X.$$ This is exactly the **PCA reconstruction** from the top $d_z$ principal components.
>
> The minimum achievable loss equals the sum of the discarded eigenvalues (residual variance): $$\min_{W_e, W_d}\; \mathbb{E}\!\left[ \| X - W_d W_e X \|^2 \right]
> = \sum_{i=d_z+1}^{d_x} \lambda_i.$$
>
> Equivalently, if the centered data matrix $X_{\text{data}} \in \mathbb{R}^{d_x \times n}$ has the singular value decomposition $X_{\text{data}} = U S V^\top$, the optimal linear autoencoder spans the same subspace as the top-$d_z$ left singular vectors $U_{1:d_z}$.

> [!remark] Remark
> Note that the solution of the autoencoder is not unique by definition. Given any encoder--decoder pair $(E^\phi, D^\theta)$, we can construct another pair with the same reconstruction mapping and hence the same loss. Specifically, for any invertible matrix $R \in \mathbb{R}^{d_z \times d_z}$, define $$\tilde E^\phi = R \circ  E^\phi, \qquad
> \tilde D^\theta = D^\theta \circ  R^{-1},$$ where $\circ$ denotes function composition. Then $\tilde D^\theta \circ \tilde E^\phi = D^\theta \circ E^\phi$, so the reconstruction $\hat X$ and the loss remain unchanged. This shows that the latent representation $Z = E^\phi(X)$ is only defined up to an arbitrary invertible linear transformation of the latent space.

An autoencoder (AE) learns to represent data through a pair of neural networks: an encoder $E^\phi$ that maps the input $X$ to a low-dimensional latent code $Z = E^\phi(X)$, and a decoder $D^\theta$ that reconstructs the input from the code, $\hat{X} = D^\theta(Z)$. Typically, the latent dimension satisfies $\text{dim}(Z) < \text{dim}(X)$, enforcing information compression.

The training objective minimizes the reconstruction error: $$\min_{\theta, \phi}\, \mathbb{E}_{X \sim P_\text{data}}
\left[ \| X - D^\theta(E^\phi(X)) \|^2 \right].$$

Autoencoders are widely used for dimensionality reduction, denoising, and pretraining in deep learning. However, they are not inherently generative: the latent representation $Z$ is unconstrained and does not necessarily follow a known prior distribution, such as a standard Gaussian. As a result, the learned latent space may have a complex or irregular structure, making it difficult to sample valid $Z$ values for generating new data. In other words, without an explicit prior on $Z$, generation from an autoencoder remains ill-defined.

### Generative Autoencoders

Autoencoders can be made generative by introducing a regularization term that constrains the distribution of the latent variable $Z$. The idea is to encourage the encoder outputs $Z = E^\phi(X)$, where $X \sim P_\text{data}$, to follow a simple noise prior $P_\text{noise}$, such as a standard Gaussian. The training objective becomes: $$\min_{\theta, \phi} \;
\mathbb{E}_{X \sim P_\text{data}}
\left[ \| X - T^\theta(E^\phi(X)) \|^2 \right]

+ \beta\, D(P_Z^\phi,\, P_\text{noise}),$$ where $P_Z^\phi$ denotes the distribution of encoded latents, and $D(\cdot,\cdot)$ measures their discrepancy. This regularization bridges the gap between deterministic autoencoders and true generative models, enabling sampling of new data from the learned latent distribution.

Different generative autoencoder variants are distinguished by the choice of the divergence $D$:

1.  **Adversarial Autoencoders (AAE)** use a GAN-based adversarial loss to match $P_Z^\phi$ to $P_\text{noise}$.

2.  **Variational Autoencoders (VAE)** use the Kullback--Leibler (KL) divergence, with a stochastic encoder that enables a meaningful KL divergence computation.

### Adversarial Autoencoders (AAE)

The Adversarial Autoencoder (AAE) combines the reconstruction principle of autoencoders with the adversarial training mechanism of GANs to regularize the latent space. It enforces the encoded latent distribution $P_Z^\phi$ to match a simple prior $P_\text{noise}$ by adding a GAN-style divergence term to the reconstruction loss:

$$
\min_{\theta, \phi}\;
\mathbb{E}_{X \sim P_\text{data}}
\left[\|X - T^\theta(E^\phi(X))\|^2\right]

+ \beta\, D_{\text{GAN}}(P_Z^\phi, P_\text{noise}),
$$

where $D_{\text{GAN}}$ measures the discrepancy between the encoded distribution $P_Z^\phi$ and the prior $P_\text{noise}$. It can be implemented using any off-the-shelf GAN method, which uses a discriminator (or critic) $h_\beta$ that attempts to distinguish encoded samples from true noise samples:

$$
D_{\text{GAN}}(P_Z^\phi, P_\text{noise})
  = \max_{h}\!
  \Big\{
  \mathbb{E}_{X\sim P_\text{data}}[h(E^\phi(X))]

- \mathbb{E}_{Z\sim P_\text{noise}}[h(Z)]
- \Phi(h)
  \Big\},
$$

where $\Phi(h)$ depends on the specific GAN formulation (e.g., Wasserstein, Jensen--Shannon, or f-GAN).

##### Full Objective.

The overall optimization becomes a min--max game: $$\min_{\theta, \phi} \max_h
\mathbb{E}_{X \sim P_\text{data}}
\left[\|X - T^\theta(E^\phi(X))\|^2\right]

+ \beta
  \Big(
  \mathbb{E}_{X\sim P_\text{data}}[h(E^\phi(X))]

- \mathbb{E}_{Z\sim P_\text{noise}}[h(Z)]
- \Phi(h)
  \Big).$$ This formulation integrates the reconstruction power of autoencoders with the distributional matching capability of GANs, yielding a model that can both encode data and generate new samples.

![Architecture of an Adversarial Autoencoder (AAE), combining an autoencoder with a GAN-style discriminator to align the latent distribution $P_Z^\phi$ with a prior $P_\text{noise}$. *Source:* Makhzani et al., Adversarial Autoencoders (2015).](/assets/modules/05-autoencoder-models/aae.png)

### Variational Autoencoders (VAE)

The Variational Autoencoder (VAE) \[Kingma & Welling, 2014\] introduces a probabilistic formulation of the autoencoder by combining reconstruction with a KL divergence penalty: $$\min_{\theta, \phi}
\; \mathbb{E}_{X \sim P_\text{data}}
\!\left[ \| X - T^\theta(E^\phi(X)) \|^2 \right]

+ \beta\, D_{\text{KL}}(P_Z^\phi \,\|\, P_\text{noise}).$$ Here, $P_Z^\phi$ is the distribution of latent variables induced by the encoder $Z = E^\phi(X)$, and $P_\text{noise}$ is a simple prior, typically $\mathcal{N}(0, I)$.

##### Motivation.

The deterministic encoder $Z = E^\phi(X)$ makes $P_Z^\phi$ an implicit distribution, for which the KL divergence cannot be computed analytically. VAE resolves this issue by introducing a stochastic encoder that yields an explicit Gaussian form.

##### Stochastic Encoder.

Each input $X$ is encoded as a Gaussian distribution rather than a point: $$Z = \mu^\phi(X) + \sigma^\phi(X) \odot \xi,
\qquad \xi \sim \mathcal{N}(0, I),$$ where $\odot$ denotes elementwise product. Thus, the conditional distribution of the latent variable becomes $$P(Z \mid X) = \mathcal{N}\!\left(\mu^\phi(X),\, \mathrm{diag}(\sigma^\phi(X)^2)\right).$$ To align with the prior $P_\text{noise} = \mathcal{N}(0,I)$, we want encourage $\mu^\phi(X)\!\approx\!0$ and $\sigma^\phi(X)\!\approx\!1$. Although it is possible to use a vanilla square loss $\left\lVert \mu - 1\right\rVert^2$ and $\left\lVert \sigma - 1\right\rVert^2$, it is more natural to use KL divergence as follows.

> [!remark] Remark
> The divergence between two Gaussian distributions admits a closed-form expression: $$\mathrm{KL}\!\left(\mathcal{N}(\mu, \sigma)\,\|\,\mathcal{N}(0,1)\right)
> = \frac{\mu^2}{2} + \left(\frac{\sigma^2}{2} - \log\sigma\right) - \frac{1}{2}.$$ Minimizing this yields $\mu=0$ and $\sigma=1$.

Applying the formula above, the overall KL regularization between the approximate posterior and the prior is $$L_{\text{KL}}(\phi)

=
\frac{1}{2}\,
\mathbb{E}_{X \sim P_\text{data}}
\!\left[
\sum_{i=1}^{\mathrm{dim}(Z)}
\big(
\mu_{\phi,i}(X)^2 + \sigma_{\phi,i}(X)^2 - 2\log\sigma_{\phi,i}(X) - 1
\big)
\right].$$

##### Reconstruction Term.

Given this stochastic encoder, the reconstruction loss becomes $$L_{\text{AE}}(\theta, \phi)

=
\mathbb{E}_{X \sim P_\text{data},\, \xi \sim \mathcal{N}(0, I)}
\!\left[\|X - T^\theta(\mu^\phi(X) + \sigma^\phi(X)\!\odot\!\xi)\|^2\right],$$ which measures how well the decoder $T^\theta$ can reconstruct the data from the sampled latent code.

##### Overall Objective.

The VAE jointly minimizes the reconstruction error and the KL divergence: $$\min_{\theta, \phi}\;
L_{\text{AE}}(\theta, \phi)

+ \beta\, L_{\text{KL}}(\phi),$$ where $\beta$ balances reconstruction fidelity against latent regularization.

##### The $\boldsymbol{\beta}$ Trade-off.

The coefficient $\beta$ controls the strength of the KL term: higher $\beta$ promotes more disentangled latent representations but can degrade reconstruction quality. This idea leads to the **$\beta$-VAE** framework \[Higgins et al., 2017\], interpreting the VAE as learning a constrained variational representation of data.

See example code [here](https://colab.research.google.com/github/lqiang67/generative-models-on-toys/blob/main/vae_2D_toy.ipynb).

## VAE: Probabilistic View

We develop a probabilistic perspective of the Variational Autoencoder (VAE), viewing it as a latent variable model defined by joint densities over observed and hidden variables. The key challenge---marginalizing over latent variables---is addressed through variational inference, which converts intractable integrations into tractable optimization problems.

### Latent Variable Models

##### Setup.

We observe data samples $\mathcal{D} = \{X_i\}_{i=1}^n$ drawn from an unknown distribution $P^*$. To model such data, we introduce an unobserved (latent) variable $Z$ and define a generative process through a joint density over $(X, Z)$: $$p^\theta(x, z) = p^\theta(x \mid z)\, p^\theta(z),$$ where $p^\theta(z)$ is the prior distribution over the latent space, and $p^\theta(x \mid z)$ is the conditional likelihood of data given $z$.

##### Marginal Likelihood.

Since the latent variable $Z$ is unobserved, the model defines a marginal density for $X$ by integrating out $Z$: $$p^\theta(x) = \int p^\theta(x \mid z)\, p^\theta(z)\, \mathrm{d}z.$$ If both $(x_i, z_i)$ pairs were observed, one could directly maximize the joint log-likelihood: $$\max_\theta \sum_{i=1}^n \log p^\theta(x_i, z_i).$$ However, in practice only $\{x_i\}$ are observed, so we maximize the marginal log-likelihood: $$\max_\theta \sum_{i=1}^n \log p^\theta(x_i)
= \max_\theta \sum_{i=1}^n \log \int p^\theta(x_i \mid z)\, p^\theta(z)\, \mathrm{d}z.$$

In words, we maximize the log-likelihood of what we observe, and integrate (marginalize) over what we do not.

However, for deep generative models, the integral in $p^\theta(x)$ is typically intractable, since the latent variable appears inside a nonlinear neural network. Variational inference provides an elegant solution: it replaces the difficult integration over $Z$ with an optimization problem over a tractable family of distributions. This leads naturally to the Variational Autoencoder, where the encoder network serves as an approximate inference model for $Z$.

### VAE as a Latent Variable Model

Consider a latent variable model where the latent variable $Z$ follows a standard Gaussian prior, and the data $X$ is generated from a conditional Gaussian distribution given $Z$: $$Z \sim \mathcal{N}(0, I),
\qquad
X \mid Z \sim \mathcal{N}(T^\theta(Z),\, \beta^2 I),$$ or equivalently, $$X = T^\theta(Z) + \beta\, \xi,
\qquad \xi \sim \mathcal{N}(0, I).$$ This stochastic formulation corresponds to a decoder network $T^\theta$ with additive Gaussian noise of variance $\beta^2$.

##### Model Densities.

The corresponding densities are: $$p^\theta(z) \propto \exp\!\left(-\tfrac{1}{2}\|z\|^2\right),
\qquad
p^\theta(x \mid z)
\propto \exp\!\left(-\tfrac{1}{2\beta^2}\|x - T^\theta(z)\|^2\right).$$ Hence, the joint distribution factorizes as $$p^\theta(x, z)
= p^\theta(x \mid z)\, p^\theta(z)
\propto
\exp\!\left(
-\tfrac{1}{2\beta^2}\|x - T^\theta(z)\|^2
-\tfrac{1}{2}\|z\|^2
\right).$$

##### Learning.

If both $\{x_i, z_i\}$ were observed, maximum likelihood estimation (MLE) would maximize the joint log-likelihood: $$\log p^\theta(x_i, z_i)
= -\tfrac{1}{2\beta^2}\|x_i - T^\theta(z_i)\|^2 + \text{const.}$$ However, in practice we only observe $\{x_i\}$, and must maximize the marginal log-likelihood: $$\begin{aligned}
\log p^\theta(x_i)
& = \log \int p^\theta(x_i \mid z)\, p^\theta(z)\, \mathrm{d}z \\
& = \log \int
\exp\!\left(
-\tfrac{1}{2\beta^2}\|x_i - T^\theta(z)\|^2
-\tfrac{1}{2}\|z\|^2
\right)
\mathrm{d}z

- \text{const.}
  \end{aligned}$$ The integral over $z$ is generally intractable for neural decoders $T^\theta$, motivating the use of variational inference, as employed in the Variational Autoencoder.

### Variational Inference: Integration $\to$ Optimization

Quantities like marginal likelihood above requires to compute an integral of the form $$\int f(z)\, \mathrm{d}z, \qquad f(z) > 0,$$ which is typically intractable in high-dimensional latent-variable models.

##### Introducing an Auxiliary Distribution.

Let $q(z)$ be any valid probability density. We can rewrite the integral as an expectation under $q$: $$\int f(z)\, \mathrm{d}z
= \int \frac{f(z)}{q(z)}\, q(z)\, \mathrm{d}z
= \mathbb{E}_{q}\!\left[\frac{f(z)}{q(z)}\right].$$ This simple identity forms the basis of importance sampling and variational inference.

##### Applying Jensen's Inequality.

Taking the logarithm of both sides and applying Jensen's inequality gives $$\begin{aligned}
\log \int f(z)\, \mathrm{d}z
= \log \mathbb{E}_q\!\left[\frac{f(z)}{q(z)}\right]
\ge
\mathbb{E}_q\!\left[\log \frac{f(z)}{q(z)}\right],
\end{aligned}$$ where the inequality holds because the logarithm is a concave function. The lower bound $\mathbb{E}_q\!\left[\log \frac{f(z)}{q(z)}\right]$ is called the Evidence Lower Bound (ELBO) in variational inference. Optimizing it over $q$ transforms the original integration problem into an optimization problem.

> [!remark] Remark
> For a random variable $Z$ and a concave function $f$, $$f(\mathbb{E}[Z]) \ge \mathbb{E}[f(Z)],$$ and the inequality reverses for convex functions: $$f(\mathbb{E}[Z]) \le \mathbb{E}[f(Z)].$$
> ![Jensen inequality for a convex quadratic function.](/assets/modules/05-autoencoder-models/jensen_x2.png)
> ![Jensen inequality for a concave logarithm function.](/assets/modules/05-autoencoder-models/jensen_log.png)
> _Illustration of Jensen's inequality for convex (left) and concave (right) functions._

##### Gibbs (or Donsker--Varadhan) Variational Principle

The inequality in Eq. `equ:logfineq` becomes an equality when $q$ exactly matches the normalized version of $f$, that is, $$q^*(z) = \frac{f(z)}{\int f(z)\, \mathrm{d}z}.$$ To see this, note that for this optimal choice, $$\begin{aligned}
\mathbb{E}_{q^*}\!\left[
\log \frac{f(z)}{q^*(z)}
\right]
=
\mathbb{E}_{q^*}\!\left[
\log \int f(z)\, \mathrm{d}z
\right]
=
\log \int f(z)\, \mathrm{d}z.
\end{aligned}$$ Combining Eq. `equ:logfineq` and Eq. `equ:logfeq`, we can express the logarithm of an integral as an optimization problem: $$\begin{aligned}
\log \int f(z)\, \mathrm{d}z
&=
\max_{q \in \mathcal Q}
\mathbb{E}_{q}\!\left[\log \frac{f(z)}{q(z)}\right],
\end{aligned}$$ where $q$ is maximized in the set of all possible distributions $\mathcal Q$.

Further reformulation yields $$\begin{aligned}
\log \int f(z)\, \mathrm{d}z
&=
\max_{q}
\mathbb{E}_{q}\!\left[\log f(z) - \log q(z)\right] \\[0.75em]
&=
\max_{q}
\Big\{
\mathbb{E}_{q}\![\log f(z)] + H(q)
\Big\},
\end{aligned}$$ where $H(q)$ is the entropy of distribution $q$: $$H(q) = -\mathbb{E}_{z\sim q}[\log q(z)].$$ This identity is known as the Gibbs variational principle, or Donsker--Varadhan representation, and is fundamental to variational inference. It shows that integration can be replaced by an optimization over distributions $q$, combining two opposing forces: the expected log-likelihood term $\mathbb{E}_{q}[\log f(z)]$ and the entropy term $H(q)$ encouraging diversity in $q$.

### Variational Inference for VAE

From the Gibbs variational principle, we have:

$$
\log \int f(z)\, \mathrm{d}z

=
\max_{q}\,
\mathbb{E}_{q}\!\left[
\log f(z) - \log q(z)
\right].
$$

Applying this identity to the marginal likelihood of a latent variable model with $f(z) = p^\theta(x, z)$ (viewing $x$ as fixed), we obtain:

$$
\begin{aligned}
\log p^\theta(x)
&= \log \int p^\theta(x, z)\, \mathrm{d}z \\[0.5em]
&= \max_{q(z|x)}
\mathbb{E}_{q(z|x)}\!\left[
\log p^\theta(x, z) - \log q(z|x)
\right].
\end{aligned}
$$

The distribution $q(z|x)$ serves as an approximate posterior or a probabilistic encoder, typically parameterized by a neural network with parameters $\phi$.

The expression inside the maximization is known as the Evidence Lower Bound (ELBO):

$$
\mathcal{L}_{\text{ELBO}}(\theta, \phi)
=
\mathbb{E}_{q^\phi(z|x)}\!\left[
\log p^\theta(x, z) - \log q^\phi(z|x)
\right].
$$

Maximizing the ELBO jointly over $(\theta, \phi)$ provides a tractable surrogate for maximizing the intractable marginal likelihood $\log p^\theta(x)$.

##### Expanded Form.

Since $p^\theta(x, z) = p^\theta(x|z)\, p^\theta(z)$, the ELBO can be rewritten as:

$$
\begin{aligned}
\mathcal{L}_{\text{ELBO}}(\theta, \phi)
&=
\mathbb{E}_{q^\phi(z|x)}\!\left[
\log p^\theta(x|z)
\right]

- \mathrm{KL}\!\left(q^\phi(z|x)\,\|\,p^\theta(z)\right),
  \end{aligned}
$$

where the first term encourages accurate reconstruction (expected log-likelihood) and the second term regularizes the latent posterior to stay close to the prior.

##### Joint Optimization.

In practice, both $\theta$ and $\phi$ are learned together via:

$$
\max_{\theta, \phi}\;
\mathcal{L}_{\text{ELBO}}(\theta, \phi)
=
\mathbb{E}_{q^\phi(z|x)}\!\left[
\log p^\theta(x|z)
\right]

- \mathrm{KL}\!\left(q^\phi(z|x)\,\|\,p^\theta(z)\right).
$$

This variational reformulation converts the intractable integral in $\log p^\theta(x)$ into an optimization problem over the encoder distribution $q^\phi(z|x)$---the core idea behind the Variational Autoencoder.

<!-- prettier-ignore-end -->
