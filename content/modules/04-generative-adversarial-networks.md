---
title: "Module 4: Generative Adversarial Networks"
description: "GANs, likelihood-free training, IPMs, regularization, and minimax optimization."
publish: true
---

<!-- prettier-ignore-start -->

## Generative Adversarial Networks (GANs)

Given a dataset $\mathcal{D} = \{x_i\}_{i=1}^n$ of i.i.d. samples from a target distribution $P^*$ on $\mathbb{R}^d$, our goal is to learn a parameterized generative model $T_\theta : \mathbb{R}^m \to \mathbb{R}^d$ such that $$X_\theta = T_\theta(\xi), \qquad \xi \sim \pi_0,$$ produces samples that closely approximate $P^*$. Here, $\pi_0$ is a simple reference distribution (e.g., standard Gaussian or uniform).

If the density function $p_\theta$ of the model output $X_\theta$ can be evaluated explicitly, we may estimate $\theta$ via maximum likelihood estimation. However, computing $p_\theta$ typically requires $T_\theta$ to be invertible and to have a tractable Jacobian determinant. This constraint motivates the use of invertible architectures, such as normalizing flows, specifically designed to ensure both invertibility and efficient density computation.

##### Generative Adversarial Networks (GANs).

An alternative approach is to give up the architectural constraint that the transformation $T_\theta$ must be invertible. In a GAN, $T_\theta$ can be any general neural network that maps latent samples $\xi \sim \pi_0$ to data samples $X_\theta = T_\theta(\xi)$. Since $T_\theta$ is typically non-invertible, the induced distribution of $X_\theta$ does not generally have a tractable or even well-defined density function. In fact, the probability mass of $X_\theta$ may concentrate on a low-dimensional manifold in the data space when $T_\theta$ is not invertible, making likelihood-based training infeasible.

To train such models, we must resort to likelihood-free approaches to estimate $\theta$. A key idea is to train $T_\theta$ so that the generated samples match the statistics (or moments) of the real data. This leads to a class of moment-matching or adversarial training methods, which can be formulated as a minimax optimization problem, as we explain next.

### Likelihood-Free Training via Moment Matching

Given a generative model $X_\theta = T_\theta(\xi)$, we can simulate a set of "fake" data points by pushing forward samples from the base noise distribution: $$x_\theta^{(i)} = T_\theta(\xi^{(i)}),
\qquad
\xi^{(i)} \sim \pi_0,
\quad i = 1, \dots, n.$$ The goal is to find parameters $\theta$ such that the distribution of the simulated data $\mathcal D_\theta = \{x_\theta^{(i)}\}_{i=1}^n$ matches that of the real data $\mathcal D = \{x_i\}_{i=1}^n$ as closely as possible.

How can we tell if two datasets follow the same distribution? This is a classical **two-sample testing** problem in statistics. The idea is to compare various empirical statistics (or "moments") of the two datasets.

For example, we can match the sample means: $$\mathbb{E}[X_\theta] \approx \mathbb{E}[X_{\text{data}}].$$ However, matching only the mean is clearly insufficient. We can extend this idea by also matching higher-order moments, such as the variance. In the one-dimensional case, this corresponds to matching the second order moment: $$\mathbb{E}[X_\theta^2] \approx \mathbb{E}[X_{\text{data}}^2].$$ This is, of course, still not sufficient unless the data distribution is Gaussian. To ensure that two general distributions match perfectly, we would ideally want to match all possible statistics, such as, all polynomial moments.

![ Examples of different distributions and their discriminator functions. Left: two distributions with the same mean but different variances. Right: two distributions with the same mean and variance, but different higher-order statistics. ](/assets/modules/04-generative-adversarial-networks/discriminator_visualization.png)

This idea can be generalized by requiring that the generated distribution $P_\theta$ matches the data distribution $P_{\text{data}}$ on a rich class of test functions $\mathcal{H}$: $$\begin{aligned}
\mathbb{E}\bigl[h(X_\theta)\bigr] \approx \mathbb{E}\bigl[h(X_{\text{data}})\bigr],
\qquad \text{for all } h \in \mathcal{H}.
\end{aligned}$$

Here, the function class $\mathcal{H}$ serves as a collection of test functions used to probe and compare the two distributions. If $\mathcal{H}$ is sufficiently rich, then equality of expectations over all $h \in \mathcal{H}$ is enough to guarantee equality of the underlying distributions.

> [!note] Definition
> A set of functions $\mathcal{H}$ is called discriminative if it is rich enough such that matching the expectations over all $h \in \mathcal{H}$ implies equality of the distributions: $$P_\theta = P_{\text{data}}
> \quad \Longleftrightarrow \quad
> \mathbb{E}\bigl[h(X_\theta)\bigr] = \mathbb{E}\bigl[h(X_{\text{data}})\bigr],
> \qquad \forall h \in \mathcal{H}.$$

> [!example]
> For distributions defined on $\mathbb{R}^d$, we have $$P_{\text{data}} = P_\theta
> \quad \Longleftrightarrow \quad
> \mathbb{E}\bigl[h(X_\theta)\bigr] = \mathbb{E}\bigl[h(X_{\text{data}})\bigr],
> \quad \forall\, h \in \mathcal{H},$$ where the choice of the function class $\mathcal{H}$ determines which moments are being matched. Several important examples include:
>
> **Bounded Continuous Functions**
>
> $$
> \mathbb{E}[f(X_\theta)] = \mathbb{E}[f(X_{\text{data}})],
> \quad \text{for all bounded and continuous } f.
> $$
>
> Matching expectations for all bounded continuous functions guarantees equality of the two distributions.
>
> **Polynomials**
>
> $$
> \mathbb{E}[\mathrm{poly}(X_\theta)] = \mathbb{E}[\mathrm{poly}(X_{\text{data}})],
> \quad \text{for all polynomial functions.}
> $$
>
> If all polynomial moments agree, the two distributions coincide (under suitable regularity conditions).
>
> **Moment Generating Functions**
>
> $$
> \mathbb{E}\!\left[\exp(X_\theta^\top \omega)\right]
> =
> \mathbb{E}\!\left[\exp(X_{\text{data}}^\top \omega)\right],
> \quad \text{for all vectors } \omega.
> $$
>
> Equality of moment generating functions implies that the distributions are identical.
>
> **Single-Neuron Neural Network**
>
> $$
> \mathbb{E}[\sigma(\omega^\top X_\theta + b)]
> =
> \mathbb{E}[\sigma(\omega^\top X_{\text{data}} + b)],
> \quad \text{for all weights } \omega \text{ and biases } b.
> $$
>
> Here, $\sigma$ is a nonlinear activation function, such as $\sigma(x) = \max(x, 0)$.

In practice, we may choose the test function class $\mathcal{H}$ to be a family of neural networks. The example above shows that even a single neuron defines a valid test function for comparing distributions. More generally, multilayer neural networks provide a flexible and expressive class of test functions that can efficiently capture complex, high-dimensional relationships. They are also easy to implement and optimize using standard tools in modern machine learning.

### Integral Probability Metrics (IPM)

Based on the moment matching idea introduced above, we can define a notion of discrepancy between the model distribution $P_\theta$ and the data distribution $P_{\text{data}}$ by maximizing the difference of test function expectations under a size constraint: $$\begin{aligned}
D(P_\theta, P_{\text{data}})
\;=\;
\sup_{h \in \mathcal{H}}
\left\{
\left|\mathbb{E}\!\left[h(X_{\text{data}})\right]- \mathbb{E}\!\left[h(X_\theta)\right]\right|
\;
\right\}.
\end{aligned}$$ This family of divergences is known as the integral probability metric (IPM).

When $\mathcal{H}$ is symmetric (i.e., $h \in \mathcal{H} \Rightarrow -h \in \mathcal{H}$), we can remove the absolute value in the definition of the metric and write the IPM equivalently as

$$
D(P_\theta, P_{\text{data}})

=
\sup_{h \in \mathcal{H}}
\left\{
\mathbb{E}\!\left[h(X_{\text{data}})\right]

- \mathbb{E}\!\left[h(X_\theta)\right]
  \right\}.
$$

This simplification holds because, for any function $h \in \mathcal{H}$, its negation $-h$ also belongs to $\mathcal{H}$. Hence, the supremum of the absolute difference $\bigl|\mathbb{E}[h(X_{\text{data}})] - \mathbb{E}[h(X_\theta)]\bigr|$ is equivalent to the supremum of the signed difference over all $h \in \mathcal{H}$:

$$
\sup_{h \in \mathcal{H}}
  \left|
  \mathbb{E}\!\left[h(X_{\text{data}})\right]
- \mathbb{E}\!\left[h(X_\theta)\right]
  \right|
  =
  \sup_{h \in \mathcal{H}}
  \left\{
  \mathbb{E}\!\left[h(X_{\text{data}})\right]
- \mathbb{E}\!\left[h(X_\theta)\right]
  \right\}.
$$

In other words, the symmetry of $\mathcal{H}$ allows us to drop the absolute value without changing the value of the supremum.

> [!note] Remark
> The supremum in the IPM definition can diverge if the test function $h$ is allowed to grow arbitrarily large or oscillate too rapidly. For instance, if $h(x) = a\,x$ with an unrestricted constant $a>0$, then $$\mathbb{E}[a\,h(X_{\text{data}})] - \mathbb{E}[a\,h(X_\theta)]
> = a\big(\mathbb{E}[h(X_{\text{data}})] - \mathbb{E}[h(X_\theta)]\big),$$ which can be made arbitrarily large by increasing $a$. Constraining $a$ (e.g., $|a|\le1$) prevents this divergence.
>
> Hence, $\mathcal{H}$ must impose a constraint on the norm, magnitude, or smoothness of $h$, such as equiring $\|h\|_{\infty} \le 1$ or $\|h\|_{\text{Lip}} \le 1$; see Table `tab:ipm`. Such constraints limit the "strength'' of $h$, ensuring that the IPM measures genuine distributional differences rather than the effect of scaling.
> **Name** **$\mathcal{H}$** **Equivalent Formulation / Comment**
> Total variation (TV) $\{ h \colon \|h\|_\infty \le 1\}$ $D_{\text{TV}}(P,Q) = \int |p(x)-q(x)|\mathrm{d} x$
> Maximum mean discrepancy (MMD) $\{h \colon \|h\|_{\mathcal{H}_k} \le 1\}$ RKHS norm induced by a kernel $k$
> Wasserstein-1 distance $\{h \colon \|h\|_{\text{Lip}} \le 1\}$ Kantorovich--Rubinstein dual form

: Examples of integral probability metrics (IPMs). Examples of integral probability metrics (IPMs). By choosing different function classes $\mathcal{H}$, the IPM reduces to various classical measures of distributional discrepancy.

### Regularized IPM

In practice, it is common to add

### Generative Adversarial Networks (GANs)

In practice, one often considers a parametric critic $h_\beta$ and introduces a regularized empirical approximation:

$$
\begin{aligned}
D_{\mathrm{reg}}(P_\theta, P_{\text{data}})
\;=\;
\max_{\beta}
\left\{
\mathbb{E}\!\left[h_\beta(X_{\text{data}})\right]

- \mathbb{E}\!\left[h_\beta(X_\theta)\right]
- \lambda\,\Phi(h_\beta)
  \right\},
  \end{aligned}
$$

where we introduce a regularization term in which $\Phi(\cdot)$ measures the magnitude or smoothness of the test function $h$, and $\lambda > 0$ controls the strength of this regularization. The regularization term prevents the critic from growing without bound and helps ensure stability during optimization.

Training the generator then becomes a minimax problem:

$$
\min_{\theta}\; D_{\mathrm{reg}}(P_\theta, P_{\text{data}})
\;=\;
\min_{\theta}\max_{\beta}
\left\{
\mathbb{E}\!\left[h_\beta(X_{\text{data}})\right]

- \mathbb{E}\!\left[h_\beta(X_\theta)\right]
- \lambda\,\Phi(h_\beta)
  \right\}.
$$

This formulation captures the essence of adversarial training, in which two components (the generator and the critic) interact in a competitive optimization process:

- **Critic (or Discriminator) $h_\beta$:** Given the current generator $T_\theta$, the critic seeks to find a test function $h_\beta$ that maximizes the discrepancy between the real and generated data distributions. Intuitively, the critic tries to distinguish real samples from fake ones by assigning higher scores to $X_{\text{data}}$ and lower scores to $X_\theta$. The regularization term $\lambda\,\Phi(h_\beta)$ limits the critic's expressiveness to prevent overfitting and ensure a well-defined optimization.

- **Generator $T_\theta$:** The generator produces samples $X_\theta = T_\theta(\xi)$, where $\xi \sim \pi_0$, and aims to minimize the discrepancy measured by the critic. In doing so, it adjusts its parameters $\theta$ so that the generated distribution $P_\theta$ becomes indistinguishable from the data distribution $P_{\text{data}}$. When the minimax game reaches equilibrium, the generator has successfully learned to replicate the data distribution.

This adversarial structure underlies many modern generative models, including the Generative Adversarial Network (GAN), where the critic is typically implemented as a neural network, and the optimization alternates between updating $\beta$ and $\theta$.

##### Classical GAN

The original GAN, proposed by Goodfellow et al. [@goodfellow2020generative], can be viewed as a special case of the IPM framework with an implicit convex regularizer defined by

$$
\Phi(h)

=
\mathbb{E}\!\left[\phi\!\left(h(X_{\text{data}})\right) + \phi\!\left(h(X_\theta)\right)\right],
$$

where the convex function $\phi\colon \mathbb{R} \to [0, \infty)$ is given by

$$\phi(h) = \log\!\left(\exp(h) + \exp(-h)\right).$$

With this choice, the GAN loss becomes

$$
\begin{aligned}
L(\theta, \beta)
&=
\mathbb{E}\!\left[h_\beta(X_{\text{data}}) - h_\beta(X_\theta)

- \phi\!\left(h_\beta(X_{\text{data}})\right)
- \phi\!\left(h_\beta(X_\theta)\right)\right]\\[0.5em]
&=
\mathbb{E}\!\left[\log p_\beta(X_{\text{data}})\right]

+ \mathbb{E}\!\left[\log\!\bigl(1 - p_\beta(X_\theta)\bigr)\right],
  \end{aligned}
$$

where

$$
p_\beta(x)
=
\frac{\exp\bigl(h_\beta(x)\bigr)}
{\exp\bigl(h_\beta(x)\bigr) + \exp\bigl(-h_\beta(x)\bigr)},
\qquad
1 - p_\beta(x)
=
\frac{\exp\bigl(-h_\beta(x)\bigr)}
{\exp\bigl(h_\beta(x)\bigr) + \exp\bigl(-h_\beta(x)\bigr)}.
$$

Here, $p_\beta(x)$ can be interpreted as the discriminator's probability that the sample $x$ comes from the real data distribution. The discriminator is trained to distinguish real samples $X_{\text{data}}$ from generated samples $X_\theta$, while the generator $T_\theta$ is trained to make them indistinguishable.

More generally, $\phi(\cdot)$ can be replaced by any other nonnegative convex function, leading to the broader $f$-GAN family of models [@nowozin2016f].

##### Wasserstein GAN

The Wasserstein GAN [@arjovsky2017wasserstein] reformulates GAN training as minimizing the Wasserstein distance between the data and model distributions. This distance arises naturally from the integral probability metric (IPM) framework when the critic $h$ is constrained to be 1-Lipschitz continuous.

$$
W(P_\theta, P_{\text{data}})
=
\sup_{h:\, \|h\|_{\text{Lip}} \le 1}
\left\{
\mathbb{E}[h(X_{\text{data}})] - \mathbb{E}[h(X_\theta)]
\right\},
$$

where the Lipschitz norm is defined as

$$
\|h_\beta\|_{\text{Lip}}
=
\sup_{x \neq y}
\frac{|h_\beta(x) - h_\beta(y)|}{\|x - y\|_2}
=
\sup_x \|\nabla_x h_\beta(x)\|_2.
$$

##### Practical Implementation

In practice, the Lipschitz constraint is enforced approximately by constraining the parameters of the critic $h_\beta$:

$$
\min_{\theta} \max_{\beta}
\left\{
\mathbb{E}[h_\beta(X_{\text{data}})] - \mathbb{E}[h_\beta(X_\theta)]
\quad \text{s.t.} \quad |\beta|_\infty \le c
\right\},
$$

where $c > 0$ controls the scale of the critic's weights.

##### WGAN-GP

Weight clipping can restrict the critic's capacity and lead to optimization issues. To address this, the improved WGAN by [@gulrajani2017improved] replaces the hard constraint with a gradient penalty (GP) that softly enforces the Lipschitz condition:

$$
\min_{\theta}\max_{\beta}
\left\{
\mathbb{E}[h_\beta(X_{\text{data}})] - \mathbb{E}[h_\beta(X_\theta)]

- \lambda\,\texttt{GP}(h_\beta)
  \right\},
$$

where

$$
\texttt{GP}(h_\beta)
  =
  \mathbb{E}\!\left[\left(\|\nabla_{\hat X} h_\beta(\hat X)\|_2 - 1\right)^2\right],
$$

and

$$
\hat X = U X_{\text{data}} + (1 - U) X_\theta,
\qquad
U \sim \text{Uniform}[0,1].
$$

Here, $X_{\text{data}} \sim P^*$ is a real data sample, and $X_\theta = T_\theta(\xi)$ is a generated sample drawn from the model distribution $P_\theta$. The random interpolation coefficient $U$ ensures that $\hat X$ lies uniformly along the straight line segment between a real and a generated point.

The gradient penalty encourages the critic to have gradients of unit norm along the straight line between real and generated samples.

> [!note] Remark
> It is reasonable to use the following variance of gradient penalty: $$\texttt{GP}(h_\beta) = \mathbb{E}[(\max(\left\lVert \nabla_{\hat X} h_\beta(\hat X)\right\rVert^2 - 1, ~~0))^2],$$ which places penalty only when the gradient magnitude is larger than one, i.e., $|\nabla_{\hat X} h_\beta(\hat X)|^2 > 1$.

### Solving Minimax with Alternating Gradient Descent

In general, GAN training can be formulated as solving a minimax optimization problem:

$$
\min_{\theta}\max_{\beta} L(T_\theta,\, h_\beta),
$$

where

$$
L(T_\theta,\, h_\beta)

=
\mathbb{E}_{X_{\text{data}}\sim P^*}[h_\beta(X_{\text{data}})]

- \mathbb{E}_{X_\theta\sim P_\theta}[h_\beta(X_\theta)]
- \lambda\,\Phi(h_\beta)
$$

is the adversarial loss functional that couples the generator $T_\theta$ and the critic $h_\beta$.

In practice, GANs are trained by alternating gradient descent, that is, by repeatedly updating the critic and the generator in turn using gradient-based optimization. A typical training iteration alternates between:

1.  **Critic update:** Maximize $L(T_\theta, h_\beta)$ with respect to $\beta$ (often for several steps) while holding $\theta$ fixed.

2.  **Generator update:** Minimize $L(T_\theta, h_\beta)$ with respect to $\theta$ using the most recent critic.

See the algorithm below for an illustration of this alternating update procedure.

Update critic parameters $\beta$ (with generator fixed): $$\beta_t \gets \beta_t + \epsilon\, \nabla_\beta L(T_{\theta_t}, h_{\beta_t}).$$

Update generator parameters: $$\theta_t \gets \theta_t - \epsilon\, \nabla_\theta L(T_{\theta_t}, h_{\beta_t}).$$

> [!note] Remark
> Although the regularization term $\Phi(h_\beta)$ may depend indirectly on the generator through the generated data (for example, in WGAN-GP where $\hat X = U X_{\text{data}} + (1-U) X_\theta$ depends on $X_\theta = T_\theta(\xi)$), it is common practice to stop the gradient through $\Phi(h_\beta)$ when updating the generator. That is, we treat $\Phi(h_\beta)$ as a fixed quantity that does not backpropagate into the generator parameters $\theta$.

A practical implementation of the improved WGAN algorithm is summarized below:

Given: dataset $X_{\text{data}}$, prior $\pi_0$, generator $T_\theta$, critic $h_\beta$, penalty weight $\lambda$, total iterations $T$, and critic update frequency $T_{\text{critic}}$. Sample real data $x_{\text{data}}^{(i)} \sim X_{\text{data}}$ Sample noise $\xi^{(i)} \sim \pi_0$, generate fake data $x_\theta^{(i)} = T_\theta(\xi^{(i)})$ Compute interpolation:

$$
\hat x^{(i)} = \alpha_i x_{\text{data}}^{(i)} + (1 - \alpha_i)x_\theta^{(i)},
    \quad \alpha_i \sim U[0,1].
$$

Update critic $h_\beta$ by minimizing

$$
\frac{1}{n}\sum_{i=1}^{n}
    \Bigl[h_\beta(x_\theta^{(i)}) - h_\beta(x_{\text{data}}^{(i)})\Bigr]
    +
    \lambda\left(\|\nabla_{\hat x^{(i)}} h_\beta(\hat x^{(i)})\|_2 - 1\right)^2.
$$

Sample noise $\xi^{(i)} \sim \pi_0$, generate fake data $x_\theta^{(i)} = T_\theta(\xi^{(i)})$ Update generator by minimizing

$$
\frac{1}{n}\sum_{i=1}^{n} -h_\beta(x_\theta^{(i)}).
$$

<!-- prettier-ignore-end -->
