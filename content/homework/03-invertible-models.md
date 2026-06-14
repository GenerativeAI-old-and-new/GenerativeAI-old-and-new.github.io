---
title: "Homework 3: Invertible Models and Normalizing Flows"
description: "Homework problems for Module 3 covering change of variables, log-determinants, coupling layers, and RealNVP-style flows."
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
> omitting the Jacobian term. For data with $\sum_i x_i^2>0$, show that this objective has no finite maximizer when $\theta>0$. Then compute $\int_{\mathbb R}\pi_0(x/\theta)\,\mathrm dx$ and explain, in plain language, why omitting the Jacobian breaks probability conservation.

> [!problem|Optional Numerical Check of the Jacobian Term]
> Simulate $n=1000$ samples from $X\sim\mathcal N(0,2^2)$. Compute the closed-form MLE from the correct likelihood. Then run gradient ascent on the incorrect objective from the previous problem with $\theta>0$ and describe what happens to $\theta$ during training.
>
> Plot the correct log-likelihood and the incorrect objective as functions of $\theta$ on the same axis. Use the plot to explain why the correct objective has a finite optimum but the incorrect objective keeps improving as $\theta$ grows.

> [!problem|Additive Coupling Layer]
> Define an additive coupling layer
>
> $$
> y_a=x_a,\qquad y_b=x_b+t_\theta(x_a),
> $$
>
> where $t_\theta:\mathbb R^{d_a}\to\mathbb R^{d_b}$ is differentiable. Write the inverse mapping $x=T_\theta^{-1}(y)$. Then write the forward and inverse Jacobian matrices in block form and compute $\log|\det\nabla_y T_\theta^{-1}(y)|$.

> [!problem|Affine Coupling Layer]
> Define an affine coupling layer
>
> $$
> y_a=x_a,\qquad
> y_b=x_b\odot\exp(s_\theta(x_a))+t_\theta(x_a),
> $$
>
> where $s_\theta,t_\theta:\mathbb R^{d_a}\to\mathbb R^{d_b}$ and $\odot$ denotes elementwise multiplication. Write the inverse mapping $x=T_\theta^{-1}(y)$ and explain why the layer is invertible for any differentiable $s_\theta,t_\theta$. Then compute the forward log-determinant $\log|\det\nabla_x T_\theta(x)|$ and the inverse log-determinant $\log|\det\nabla_y T_\theta^{-1}(y)|$.

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
