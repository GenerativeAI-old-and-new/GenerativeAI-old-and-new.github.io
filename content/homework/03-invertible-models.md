---
title: "Homework 3: Invertible Models and Normalizing Flows"
description: "RealNVP implementation, change of variables, Jacobian log-determinants, and affine coupling layers."
publish: true
---

[Back to Module 3 notes](/modules/03-invertible-models)

## Coding

> [!problem|RealNVP on a 2D Toy Dataset]
> Complete the provided [Colab notebook](https://colab.research.google.com/drive/10hLmxvGQyW6-rteSLI5nzoVd4ieAxAyo?usp=sharing). You will implement a small invertible flow $T_\theta:\mathbb R^2\to\mathbb R^2$ that maps base noise $\xi\sim\mathcal N(0,I_2)$ to a mixture of eight Gaussians arranged on a ring.
>
> In the notes, $T_{k,\theta}$ denotes the base-to-data map. In the notebook API, `AffineCoupling.forward` is intentionally the inverse map $T_{k,\theta}^{-1}$ because it is used by likelihood evaluation, while `AffineCoupling.inverse` is the base-to-data map $T_{k,\theta}$ used by sampling.
>
> 1.  Implement the MLP subnetworks used inside the coupling layers.
> 2.  Implement `AffineCoupling.forward`, the data-to-base direction $T_{k,\theta}^{-1}$ used for likelihood evaluation.
> 3.  Implement `AffineCoupling.inverse`, the base-to-data direction $T_{k,\theta}$ used for generation.
> 4.  Implement `RealNVP2D.x_to_xi` by composing the data-to-base direction of all coupling layers in the correct order.
> 5.  Implement `RealNVP2D.xi_to_x` by composing the base-to-data direction of all coupling layers in the correct order.
> 6.  Implement `RealNVP2D.log_prob`, including both the base log-density and the accumulated inverse log-determinant.
>
> Submit your completed code, the final negative log-likelihood, a training curve, generated samples, and one short comment on whether the learned samples cover all eight modes.

<!--
> [!solution]- Solution
>
> [Solution notebook](https://drive.google.com/file/d/18U1A-8xs3opB6WspQgbddlZmQMeyftA7/view?usp=sharing)
>
> The coupling network can be a small two-hidden-layer MLP:
>
> ```python
> class MLP(nn.Module):
>     def __init__(self, in_dim, out_dim, hidden=64):
>         super().__init__()
>         self.net = nn.Sequential(
>             nn.Linear(in_dim, hidden), nn.ReLU(),
>             nn.Linear(hidden, hidden), nn.ReLU(),
>             nn.Linear(hidden, out_dim),
>         )
>
>     def forward(self, x):
>         return self.net(x)
> ```
>
> For one affine coupling layer, `forward` runs from data to base and therefore uses the negative scale. `inverse` runs from base to data:
>
> ```python
> def forward(self, x):
>     idx_a = self.mask.bool()
>     idx_b = (1 - self.mask).bool()
>     x_a = x[:, idx_a]
>     s = torch.tanh(self.s_net(x_a)) * self.s_clamp
>     t = self.t_net(x_a)
>
>     xi = x.clone()
>     xi[:, idx_b] = (x[:, idx_b] - t) * torch.exp(-s)
>     logdet_Tinv = (-s).sum(dim=1)
>     return xi, logdet_Tinv
>
> def inverse(self, xi):
>     idx_a = self.mask.bool()
>     idx_b = (1 - self.mask).bool()
>     xi_a = xi[:, idx_a]
>     s = torch.tanh(self.s_net(xi_a)) * self.s_clamp
>     t = self.t_net(xi_a)
>
>     x = xi.clone()
>     x[:, idx_b] = xi[:, idx_b] * torch.exp(s) + t
>     return x
> ```
>
> If $T_\theta=T_{K,\theta}\circ\cdots\circ T_{1,\theta}$, sampling applies $T_{1,\theta},\ldots,T_{K,\theta}$, while likelihood evaluation applies the inverse layers in reverse order:
>
> ```python
> def x_to_xi(self, x):
>     h = x
>     total_logdet = torch.zeros(x.size(0), device=x.device)
>     for layer in reversed(self.layers):
>         h, logdet = layer(h)
>         total_logdet += logdet
>     return h, total_logdet
>
> def xi_to_x(self, xi):
>     h = xi
>     for layer in self.layers:
>         h = layer.inverse(h)
>     return h
>
> def log_prob(self, x):
>     xi, logdet_Tinv = self.x_to_xi(x)
>     return self._pi0().log_prob(xi) + logdet_Tinv
> ```
>
> Two useful checks are
>
> $$
> T_\theta^{-1}(T_\theta(\xi))\approx \xi
> \qquad\text{and}\qquad
> -\frac1n\sum_{i=1}^n\log p_\theta(x_i)
> \text{ decreases during training.}
> $$
>
> In one reference run with six coupling layers and 5000 updates, the fresh-data NLL was about $2.68$, and the maximum round-trip error was about $2.7\times10^{-5}$. The generated samples covered all eight modes. Exact values will vary with initialization and training settings.
-->

## Theory

Unless stated otherwise, let $\xi\in\mathbb R^d$ be a base sample with density $\pi_0$, and let $x=T_\theta(\xi)$ where $T_\theta:\mathbb R^d\to\mathbb R^d$ is invertible. For density evaluation, write $\xi=T_\theta^{-1}(x)$. The change-of-variables formula is

$$
\log p_\theta(x)
=
\log \pi_0(\xi)
+
\log\left|\det \nabla_x T_\theta^{-1}(x)\right|
=
\log \pi_0(\xi)
-
\log\left|\det \nabla_\xi T_\theta(\xi)\right|.
$$

For coupling layers, split a vector as $x=(x_a,x_b)$ with dimensions $d_a+d_b=d$. In the theory problems below, $T_\theta$ and $T_{k,\theta}$ denote base-to-data maps; inverse maps are written explicitly as $T_\theta^{-1}$ or $T_{k,\theta}^{-1}$.

> [!problem|Why the Log-Determinant Matters]
> Consider a one-dimensional flow
>
> $$
> x=T_\theta(\xi)=\theta\xi,\qquad \xi\sim\mathcal N(0,1),\qquad \theta>0.
> $$
>
> Given data $x_1,\ldots,x_n$ with $\sum_i x_i^2>0$, derive $p_\theta(x)$ and $\log p_\theta(x)$ using the change-of-variables formula. Then write the log-likelihood $\mathcal L(\theta)=\sum_i\log p_\theta(x_i)$ and solve for the MLE $\hat\theta$. Finally, explain what changes if $\theta$ is allowed to be negative.

<!--
> [!solution]- Solution
>
> The inverse map is $T_\theta^{-1}(x)=x/\theta$, with derivative $1/\theta$. Therefore
>
> $$
> p_\theta(x)
> =
> \pi_0\!\left(\frac{x}{\theta}\right)\frac1\theta
> =
> \frac{1}{\theta\sqrt{2\pi}}
> \exp\!\left(-\frac{x^2}{2\theta^2}\right).
> $$
>
> Thus $X\sim\mathcal N(0,\theta^2)$ and
>
> $$
> \log p_\theta(x)
> =
> -\frac12\log(2\pi)-\log\theta-\frac{x^2}{2\theta^2}.
> $$
>
> Let $S=\sum_{i=1}^n x_i^2$. The log-likelihood and its derivative are
>
> $$
> \mathcal L(\theta)
> =
> -\frac n2\log(2\pi)-n\log\theta-\frac{S}{2\theta^2},
> \qquad
> \mathcal L'(\theta)=-\frac n\theta+\frac{S}{\theta^3}.
> $$
>
> Setting the derivative to zero gives
>
> $$
> \hat\theta=\sqrt{\frac1n\sum_{i=1}^n x_i^2}.
> $$
>
> If negative values are allowed, the density depends only on $|\theta|$ and $\theta^2$. The two MLEs are therefore $\hat\theta=\pm\sqrt{S/n}$: changing the sign flips the latent variable but leaves the distribution of $X$ unchanged.
-->

> [!problem|What Breaks If We Drop the Jacobian?]
> For the same one-dimensional flow, suppose someone uses the incorrect objective
>
> $$
> \widetilde{\mathcal L}(\theta)
> =
> \sum_{i=1}^n
> \log \pi_0\!\left(\frac{x_i}{\theta}\right),
> $$
>
> omitting the Jacobian term. For data with $\sum_i x_i^2>0$, show that this objective has no finite maximizer when $\theta>0$. Then compute $\int_{\mathbb R}\pi_0(x/\theta)\,\mathrm d x$ and explain, in plain language, why omitting the Jacobian breaks probability conservation.

<!--
> [!solution]- Solution
>
> Again let $S=\sum_i x_i^2>0$. The incorrect objective is
>
> $$
> \widetilde{\mathcal L}(\theta)
> =
> -\frac n2\log(2\pi)-\frac{S}{2\theta^2}.
> $$
>
> Its derivative is
>
> $$
> \widetilde{\mathcal L}'(\theta)=\frac{S}{\theta^3}>0.
> $$
>
> Hence the objective keeps increasing as $\theta$ grows. It approaches $-\frac n2\log(2\pi)$ as $\theta\to\infty$, but never reaches this value at a finite $\theta$.
>
> The missing normalization is visible from a change of variables $u=x/\theta$:
>
> $$
> \int_{\mathbb R}\pi_0(x/\theta)\,\mathrm d x
> =
> \theta\int_{\mathbb R}\pi_0(u)\,\mathrm d u
> =\theta.
> $$
>
> Stretching the coordinate by a factor of $\theta$ stretches each interval by the same factor. The Jacobian term $1/\theta$ lowers the density to compensate. Without it, the total mass becomes $\theta$ rather than $1$.
-->

> [!problem|Optional Numerical Check of the Jacobian Term]
> Simulate $n=1000$ samples from $X\sim\mathcal N(0,2^2)$. Compute the closed-form MLE from the correct likelihood. Then run gradient ascent on the incorrect objective from the previous problem with $\theta>0$ and describe what happens to $\theta$ during training.
>
> Plot the correct log-likelihood and the incorrect objective as functions of $\theta$ on the same axis. Use the plot to explain why the correct objective has a finite optimum but the incorrect objective keeps improving as $\theta$ grows.

<!--
> [!solution]- Solution
>
> For simulated data, the correct estimate is
>
> ```python
> torch.manual_seed(0)
> x = 2.0 * torch.randn(1000)
> theta_mle = x.square().mean().sqrt()
> print(theta_mle)
> ```
>
> It should be close to $2$. The two objectives can be evaluated with
>
> ```python
> grid = torch.linspace(0.2, 8.0, 400)
> S, n = x.square().sum(), x.numel()
> correct = -0.5*n*math.log(2*math.pi) - n*grid.log() - S/(2*grid**2)
> incorrect = -0.5*n*math.log(2*math.pi) - S/(2*grid**2)
> ```
>
> The correct curve peaks near the sample estimate of $2$. The incorrect curve rises toward a horizontal asymptote, so gradient ascent continues to increase $\theta$. The exact value reached in a finite run depends on the learning rate and number of updates.
-->

> [!problem|Additive Coupling Layer]
> Define an additive coupling layer
>
> $$
> y_a=x_a,\qquad y_b=x_b+t_\theta(x_a),
> $$
>
> where $t_\theta:\mathbb R^{d_a}\to\mathbb R^{d_b}$ is differentiable. Write the inverse mapping $x=T_\theta^{-1}(y)$. Then write the forward and inverse Jacobian matrices in block form and compute $\log|\det\nabla_y T_\theta^{-1}(y)|$.

<!--
> [!solution]- Solution
>
> Since $y_a=x_a$, we recover $x_a$ first and then subtract the translation:
>
> $$
> x_a=y_a,
> \qquad
> x_b=y_b-t_\theta(y_a).
> $$
>
> Let $J_t(u)=\nabla_u t_\theta(u)$. With coordinates ordered as $(a,b)$,
>
> $$
> \nabla_xT_\theta(x)
> =
> \begin{bmatrix}
> I&0\\
> J_t(x_a)&I
> \end{bmatrix},
> \qquad
> \nabla_yT_\theta^{-1}(y)
> =
> \begin{bmatrix}
> I&0\\
> -J_t(y_a)&I
> \end{bmatrix}.
> $$
>
> Both matrices are block triangular with ones on the diagonal. Therefore
>
> $$
> \det\nabla_yT_\theta^{-1}(y)=1,
> \qquad
> \log|\det\nabla_yT_\theta^{-1}(y)|=0.
> $$
-->

> [!problem|Affine Coupling Layer]
> Define an affine coupling layer
>
> $$
> y_a=x_a,\qquad
> y_b=x_b\odot\exp(s_\theta(x_a))+t_\theta(x_a),
> $$
>
> where $s_\theta,t_\theta:\mathbb R^{d_a}\to\mathbb R^{d_b}$ and $\odot$ denotes elementwise multiplication. Write the inverse mapping $x=T_\theta^{-1}(y)$ and explain why the layer is invertible for any differentiable $s_\theta,t_\theta$. Then compute the forward log-determinant $\log|\det\nabla_x T_\theta(x)|$ and the inverse log-determinant $\log|\det\nabla_y T_\theta^{-1}(y)|$.

<!--
> [!solution]- Solution
>
> The pass-through block gives $x_a=y_a$. The transformed block is inverted coordinate by coordinate:
>
> $$
> x_b
> =
> \big(y_b-t_\theta(y_a)\big)
> \odot\exp\!\big(-s_\theta(y_a)\big).
> $$
>
> Each scale factor is an exponential and is therefore nonzero, so the transformed block can always be inverted. The networks $s_\theta$ and $t_\theta$ do not need to be invertible.
>
> The forward Jacobian is block triangular. Its diagonal blocks are $I$ and $\operatorname{Diag}(\exp(s_\theta(x_a)))$, so
>
> $$
> \log|\det\nabla_xT_\theta(x)|
> =
> \sum_{j=1}^{d_b}s_{\theta,j}(x_a).
> $$
>
> The inverse has reciprocal scale factors, giving
>
> $$
> \log|\det\nabla_yT_\theta^{-1}(y)|
> =
> -\sum_{j=1}^{d_b}s_{\theta,j}(y_a)
> $$
-->

> [!problem|Stacking Coupling Layers]
> Let $h_0=\xi$ and $h_k=T_{k,\theta}(h_{k-1})$ for $k=1,\ldots,K$, with $y=h_K$. For each layer, suppose the affine coupling split is determined by index sets $a^{(k)}$ and $b^{(k)}$, and the scale network is $s_{k,\theta}:\mathbb R^{|a^{(k)}|}\to\mathbb R^{|b^{(k)}|}$.
>
> Show that the log-density of $y$ can be written as
>
> $$
> \log p_\theta(y)
> =
> \log \pi_0(\xi)
> -
> \sum_{k=1}^{K}
> \mathbf 1^\top
> s_{k,\theta}\!\left((h_{k-1})_{a^{(k)}}\right),
> \qquad
> \xi=T_\theta^{-1}(y).
> $$
>
> Here $\mathbf 1$ has the same dimension as the transformed block $b^{(k)}$ for layer $k$.
>
> Explain why this formula is cheap to compute for triangular-Jacobian coupling layers, while a dense unconstrained Jacobian determinant would typically require cubic-time linear algebra.

<!--
> [!solution]- Solution
>
> The Jacobian determinant of a composition is the product of the layer determinants. Taking logs turns this product into a sum:
>
> $$
> \log|\det\nabla_\xi T_\theta(\xi)|
> =
> \sum_{k=1}^K
> \log|\det\nabla T_{k,\theta}(h_{k-1})|.
> $$
>
> For affine coupling layer $k$, the unchanged coordinates contribute ones to the diagonal, while each transformed coordinate contributes $\exp(s_{k,\theta,j})$. Hence
>
> $$
> \log|\det\nabla T_{k,\theta}(h_{k-1})|
> =
> \mathbf 1^\top
> s_{k,\theta}\!\left((h_{k-1})_{a^{(k)}}\right).
> $$
>
> Substituting this result into change of variables gives
>
> $$
> \log p_\theta(y)
> =
> \log\pi_0(\xi)
> -
> \sum_{k=1}^K
> \mathbf 1^\top
> s_{k,\theta}\!\left((h_{k-1})_{a^{(k)}}\right),
> \qquad
> \xi=T_\theta^{-1}(y).
> $$
>
> A triangular determinant is the product of its diagonal entries, so reading and summing the log-scales costs $\mathcal O(d)$ per layer. A general dense $d\times d$ determinant is usually computed by an LU or QR factorization, which costs $\mathcal O(d^3)$.
-->
