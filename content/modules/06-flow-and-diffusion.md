---
title: "Module 6: Flow and Diffusion"
description: "Rectified flow, ODE/SDE generative models, diffusion, and score-based dynamics."
publish: true
---

<!-- prettier-ignore-start -->

**Optional reading.** For a more detailed treatment of rectified flow, see the [ICML 2025 tutorial](https://rectifiedflow.github.io/), [Qiang Liu's notes](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html), and the [original paper](https://arxiv.org/abs/2209.03003).

# Rectified Flow

Many learning problems can be viewed as transport: given two distributions $\pi_0$ and $\pi_1$ on $\mathbb R^d$, find a map $T:\mathbb R^d\to\mathbb R^d$ such that

$$
Z_1:=T(Z_0)\sim \pi_1,
\qquad
Z_0\sim \pi_0.
$$

For generative modeling, $\pi_0$ is an elementary distribution such as a standard Gaussian, and $\pi_1$ is the data distribution. In many of the models from the previous modules, the transport map $T$ was represented directly by a neural network. Here, $T$ is defined implicitly as the time-one map of an ordinary differential equation (ODE):

$$
\mathrm dZ_t=v^\theta(Z_t,t)\,\mathrm dt,
\qquad
t\in[0,1],
\qquad
Z_0\sim \pi_0.
$$

The velocity field $v^\theta:\mathbb R^d\times[0,1]\to\mathbb R^d$ is represented by a neural network. Starting from $Z_0\sim\pi_0$ and integrating to $t=1$ produces $Z_1=T^\theta(Z_0)$; we want $Z_1\sim\pi_1$. Thus, the network specifies the local velocity at each state and time rather than the complete map from noise to data.

The ODE keeps the state in $\mathbb R^d$, so the base and data distributions must have the same dimension. Under the usual conditions that guarantee a unique solution, the flow map is invertible: a sample at $t=1$ can be mapped back to $t=0$ by integrating the same ODE backward in time.

Sampling requires numerical integration. With forward Euler and step size $\epsilon=1/N$,

$$
Z_{t+\epsilon}
=
Z_t+\epsilon\,v^\theta(Z_t,t),
\qquad
t\in\{0,\epsilon,2\epsilon,\ldots,1\}.
$$

The number of Euler steps needed for accurate sampling depends on the geometry of the trajectories. Curved trajectories require small steps to track their changing direction. A perfectly straight trajectory has constant velocity and is recovered exactly by one Euler step from $t=0$ to $t=1$.

## Learning the ODE Drift

As discussed for continuous normalizing flows in Module 3, an ODE can be trained by maximum likelihood. More generally, if $\rho_1^v$ denotes the distribution of the endpoint under velocity field $v$, one could minimize

$$
\min_v D(\rho_1^v,\pi_1).
$$

Every evaluation of this objective requires solving the ODE, because $\rho_1^v$ depends on the full trajectory from $t=0$ to $t=1$. Repeating that solve inside each training update is expensive.

Rectified flow begins instead with a reference interpolation whose intermediate states and velocities are available in closed form. Only its endpoint distributions are fixed, so we are free to choose this interpolation. The simplest choice connects paired samples from $\pi_0$ and $\pi_1$ by straight lines.

Training then reduces to regression: sample a time $t$, evaluate the reference path at that time, and fit the velocity of the path.

## Straight Interpolation

Take a coupling $(X_0,X_1)$ of $\pi_0$ and $\pi_1$, meaning a joint distribution with marginals $X_0\sim\pi_0$ and $X_1\sim\pi_1$. In minibatch training we usually use the independent coupling and draw the two endpoints independently. Their linear interpolation is

$$
X_t=tX_1+(1-t)X_0,
\qquad
t\in[0,1].
$$

Along each sampled pair, the path has constant velocity:

$$
\mathrm dX_t=(X_1-X_0)\,\mathrm dt.
$$

To see the resulting velocity field explicitly, first consider a single fixed endpoint $X_1=x^{\mathrm{data}}$:

$$
X_t=t\,x^{\mathrm{data}}+(1-t)X_0.
$$

Differentiating gives

$$
\frac{\mathrm d}{\mathrm dt}X_t=x^{\mathrm{data}}-X_0.
$$

To write the velocity as a function of the current state $X_t$, solve

$$
X_0=\frac{X_t-tx^{\mathrm{data}}}{1-t}.
$$

Substituting this expression into the derivative gives, for $t<1$,

$$
v^*(x,t)=\frac{x^{\mathrm{data}}-x}{1-t}.
$$

The numerator is the remaining displacement to the data point, and $1-t$ is the remaining time. Although this formula contains $1/(1-t)$, its value along the interpolation is the finite, constant vector $x^{\mathrm{data}}-X_0$.

<figure class="flow-figure flow-figure-narrow">
  <img src="/assets/modules/06-flow-and-diffusion/lines_one_point.png" alt="Straight paths from noise samples to one data point." />
  <figcaption>Straight interpolations from several noise samples to one fixed data point.</figcaption>
</figure>

## Causalizing the Interpolation

The interpolation $X_t$ has the desired endpoint distributions, but it cannot be simulated from $X_0$ alone: its velocity $X_1-X_0$ uses the unknown endpoint $X_1$. The problem is visible when interpolation paths cross. At the same state $x$ and time $t$, different pairs may prescribe different velocities, whereas an ODE must assign one velocity to each $(x,t)$.

Rectified flow resolves the ambiguity by projecting the interpolation velocity onto functions of the available state and time. With an $L^2$ projection, the velocity field solves

$$
\min_v
\int_0^1
\mathbb E
\left[
  \left\|
    (X_1-X_0)-v(X_t,t)
  \right\|^2
\right]\,\mathrm dt.
$$

<figure class="flow-figure-grid flow-figure-grid-two" aria-label="Linear interpolation paths and rectified flow paths.">
  <img src="/assets/modules/06-flow-and-diffusion/line_two_points.png" alt="Straight interpolation paths that cross." />
  <img src="/assets/modules/06-flow-and-diffusion/rf_two_points.png" alt="Rectified flow paths after rewiring crossings." />
  <figcaption>Left: straight interpolations may cross. Right: the rectified ODE assigns a unique average direction at each crossing.</figcaption>
</figure>

The minimizer is the conditional expectation

$$
v^*(x,t)
=
\mathbb E[X_1-X_0\mid X_t=x].
$$

This is the standard conditional-mean solution to least-squares regression. At $(x,t)$, it averages the velocities of all interpolation paths that pass through $x$ at time $t$.

The resulting rectified ODE is

$$
\mathrm dZ_t=v^*(Z_t,t)\,\mathrm dt,
\qquad
Z_0\sim \pi_0.
$$

The ODE trajectories are therefore a rewiring of the reference paths: they use the averaged direction at crossings and can be simulated without access to $X_1$. As shown below, this rewiring changes individual paths but preserves the distribution at every time.

## Training Objective

Parameterize the velocity by a neural network $v^\theta(x,t)$ and estimate the projection objective with samples of $(X_0,X_1,t)$:

$$
L(\theta)
=
\int_0^1
\mathbb E_{X_0,X_1}
\left[
  \left\|
    X_1-X_0-v^\theta(X_t,t)
  \right\|^2
\right]\,\mathrm dt,
\qquad
X_t=tX_1+(1-t)X_0.
$$

One minibatch is constructed by sampling

$$
X_1\sim \pi_1,
\qquad
X_0\sim \pi_0,
\qquad
t\sim\mathrm{Uniform}([0,1]),
$$

For each batch, form $X_t=tX_1+(1-t)X_0$ and regress $v^\theta(X_t,t)$ onto the target velocity $X_1-X_0$.

No ODE solve is needed to construct this loss. ODE integration is used only after training, when a new sample is generated from fresh noise:

$$
\mathrm dZ_t=v^\theta(Z_t,t)\,\mathrm dt,
\qquad
Z_0\sim \pi_0.
$$

## Marginal Preservation

The interpolation process $X_t$ and the rectified ODE process $Z_t$ generally follow different sample paths. Nevertheless, at every fixed time they have the same distribution:

$$
Z_t\overset{d}{=}X_t,
\qquad
\forall t\in[0,1].
$$

This is the **marginal-preserving property**. Since $v^*(x,t)$ is the conditional average of the interpolation velocities at $(x,t)$, it produces the same probability flux as the reference process. In particular, $Z_0\sim\pi_0$ and $Z_1\sim\pi_1$, so $(Z_0,Z_1)$ defines a new coupling of the endpoint distributions.

<figure class="flow-figure flow-figure-wide">
  <img src="/assets/modules/06-flow-and-diffusion/rf_x.png" alt="Rectified flow marginal distributions matching the interpolation marginals." />
  <figcaption>The reference interpolation and rectified ODE have different trajectories but the same distribution at each time.</figcaption>
</figure>

> [!theorem|Marginal Preservation]
> Let $p_t$ be the density of $X_t$, and define
>
> $$
> v^*(x,t)=\mathbb E[\dot X_t\mid X_t=x].
> $$
>
> Then $p_t$ satisfies the continuity equation
>
> $$
> \partial_t p_t(x)
> =
> -\nabla\cdot\big(v^*(x,t)p_t(x)\big).
> $$
>
> If this continuity equation has a unique solution from the initial density $p_0$, then the ODE process $\dot Z_t=v^*(Z_t,t)$ has the same one-time marginals as $X_t$.

> [!proof|Continuity Equation]-
> Let $h:\mathbb R^d\to\mathbb R$ be smooth and compactly supported. From the density side,
>
> $$
> \frac{\mathrm d}{\mathrm dt}\mathbb E[h(X_t)]
> =
> \frac{\mathrm d}{\mathrm dt}\int h(x)p_t(x)\,\mathrm dx
> =
> \int h(x)\partial_t p_t(x)\,\mathrm dx.
> $$
>
> From the path side, by the chain rule,
>
> $$
> \frac{\mathrm d}{\mathrm dt}\mathbb E[h(X_t)]
> =
> \mathbb E[\nabla h(X_t)^\top \dot X_t].
> $$
>
> Condition on $X_t$:
>
> $$
> \mathbb E[\nabla h(X_t)^\top \dot X_t]
> =
> \mathbb E[\nabla h(X_t)^\top \mathbb E[\dot X_t\mid X_t]]
> =
> \int \nabla h(x)^\top v^*(x,t)p_t(x)\,\mathrm dx.
> $$
>
> Using integration by parts,
>
> $$
> \int \nabla h(x)^\top v^*(x,t)p_t(x)\,\mathrm dx
> =
> -
> \int h(x)\nabla\cdot\big(v^*(x,t)p_t(x)\big)\,\mathrm dx.
> $$
>
> Comparing the two expressions for $\frac{\mathrm d}{\mathrm dt}\mathbb E[h(X_t)]$ gives
>
> $$
> \int h(x)
> \left[
> \partial_t p_t(x)
> +
> \nabla\cdot\big(v^*(x,t)p_t(x)\big)
> \right]\,\mathrm dx
> =0
> $$
>
> for all such $h$, hence
>
> $$
> \partial_t p_t(x)
> =
> -\nabla\cdot\big(v^*(x,t)p_t(x)\big).
> $$

## Straightness and Reflow

The reference paths are straight, but the first rectified flow need not be. Near a crossing, averaging several line directions can bend the ODE trajectory. **Reflow** trains another rectified flow using endpoint pairs produced by the current model.

Denote the rectified flow induced by $(X_0,X_1)$ as $\boldsymbol Z=\mathsf{Rectflow}((X_0,X_1))$. Repeating the construction gives

$$
\boldsymbol Z^{k+1}
=
\mathsf{Rectflow}((Z_0^k,Z_1^k)),
\qquad
(Z_0^0,Z_1^0)=(X_0,X_1).
$$

To train iteration $k+1$, sample $Z_0^k\sim\pi_0$, integrate the current ODE to obtain the paired endpoint $Z_1^k$, and train on straight interpolations between these paired samples. This endpoint coupling records which output the current ODE assigns to each initial sample; its linear interpolations have fewer conflicting directions than those of the original independent coupling. Repeating the procedure makes the learned trajectories progressively straighter.

Straighter trajectories incur less Euler discretization error. In the perfectly straight case,

$$
Z_t=Z_0+t\,v(Z_0,0),
$$

so the ODE can be solved exactly with a single Euler step.

## Diffusion Models

Rectified flow generates samples with the deterministic ODE

$$
\mathrm d Z_t=v_{\mathrm{RF}}(Z_t,t)\,\mathrm dt.
$$

A diffusion model instead uses a stochastic differential equation (SDE):

$$
\mathrm d Z_t
=
v(Z_t,t)\,\mathrm dt
+
\sigma(Z_t,t)\,\mathrm d W_t,
$$

where $W_t$ is Brownian motion and $\sigma$ controls the amount of injected noise. Setting $\sigma=0$ recovers an ODE. In the other direction, stochastic dynamics can be added to a rectified-flow ODE while preserving its marginal distributions.

Let $\rho_t$ be the density of $X_t$ (and hence of the exact rectified flow $Z_t$). For a time-dependent noise level $\sigma_t$, add a Langevin term based on the score $\nabla\log\rho_t$:

$$
\mathrm d Z_t
=
v_{\mathrm{RF}}(Z_t,t)\,\mathrm dt
+
\sigma_t^2\nabla\log\rho_t(Z_t)\,\mathrm dt
+
\sqrt{2}\,\sigma_t\,\mathrm d W_t.
$$

<figure class="flow-figure-grid flow-figure-grid-three" aria-label="Euler sampling error and Langevin score correction.">
  <img src="/assets/modules/06-flow-and-diffusion/euler_sample_result.png" alt="Euler sampling result with visible trajectory error." />
  <img src="/assets/modules/06-flow-and-diffusion/sde_velocity.png" alt="Rectified flow velocity field." />
  <img src="/assets/modules/06-flow-and-diffusion/sde_score.png" alt="Score field used by Langevin correction." />
  <figcaption>Euler discretization may move samples away from the intended density; the score field supplies the Langevin correction.</figcaption>
</figure>

The two parts have different roles. The rectified-flow velocity drives the evolution of the time-dependent marginals $\{\rho_t\}$. At each fixed $t$, the score drift and Brownian noise form Langevin dynamics that preserve $\rho_t$. Their contributions to the Fokker--Planck equation cancel because

$$
-\nabla\cdot\left(\sigma_t^2\rho_t\nabla\log\rho_t\right)
+\sigma_t^2\Delta\rho_t
=0.
$$

Thus the exact ODE and this family of SDEs have the same time marginals. With an approximate velocity field and a finite-step solver, the stochastic correction can also move samples back toward regions favored by the current density.

> [!theorem|Tweedie's Formula]
> Assume $X_0\sim\mathcal N(0,I)$ and $X_1\sim\pi_1$, with
>
> $$
> X_t=tX_1+(1-t)X_0.
> $$
>
> Let $\rho_t$ be the density of $X_t$. Conditioning on $X_1=x_1$ gives
>
> $$
> X_t\mid X_1=x_1
> \sim
> \mathcal N\!\left(tx_1,(1-t)^2I\right),
> $$
>
> so
>
> $$
> \rho_t(x)
> \propto
> \int
> \rho_1(x_1)
> \exp\!\left(
>   -\frac{\|x-tx_1\|^2}{2(1-t)^2}
> \right)\,\mathrm dx_1.
> $$
>
> Differentiate $\log\rho_t(x)$ under the integral:
>
> $$
> \nabla\log\rho_t(x)
> =
> \mathbb E\!\left[
>   \frac{tX_1-x}{(1-t)^2}
>   \,\middle|\,
>   X_t=x
> \right].
> $$
>
> This is Tweedie's formula for the Gaussian straight interpolation. The RF velocity also gives
>
> $$
> v_{\mathrm{RF}}(x,t)
> =
> \mathbb E\!\left[
>   \frac{X_1-x}{1-t}
>   \,\middle|\,
>   X_t=x
> \right],
> $$
>
> or equivalently
>
> $$
> \mathbb E[X_1\mid X_t=x]
> =
> x+(1-t)v_{\mathrm{RF}}(x,t).
> $$
>
> Substituting into Tweedie's formula relates the score to the RF velocity:
>
> $$
> \nabla\log\rho_t(x)
> =
> \frac{t\,v_{\mathrm{RF}}(x,t)-x}{1-t}.
> $$
>
> Thus, for this interpolation, the RF velocity determines the score required by the stochastic sampler; a separate score network is not needed.

## Homework

[Homework 6: Flow and Diffusion](/homework/06-flow-and-diffusion)

<!-- prettier-ignore-end -->
