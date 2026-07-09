---
title: "Module 6: Flow and Diffusion"
description: "Rectified flow, ODE/SDE generative models, diffusion, and score-based dynamics."
publish: true
---

<!-- prettier-ignore-start -->

Optional: For a deeper introduction to rectified flow, see the [ICML 25 Tutorial](https://rectifiedflow.github.io/), [Qiang Liu's blog](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html), and the [original paper](https://arxiv.org/abs/2209.03003).

# Rectified Flow

Many learning problems can be viewed as transport: given two distributions $\pi_0$ and $\pi_1$ on $\mathbb R^d$, find a map $T:\mathbb R^d\to\mathbb R^d$ such that

$$
Z_1:=T(Z_0)\sim \pi_1,
\qquad
Z_0\sim \pi_0.
$$

For generative modeling, $\pi_0$ is an elementary distribution such as a standard Gaussian, and $\pi_1$ is the data distribution. Rectified flow learns this transport map implicitly by learning an ODE whose paths travel as straight as possible.

An ODE flow turns generation into local motion by learning a velocity field:

$$
\mathrm dZ_t=v^\theta(Z_t,t)\,\mathrm dt,
\qquad
t\in[0,1],
\qquad
Z_0\sim \pi_0.
$$

The drift $v^\theta:\mathbb R^d\times[0,1]\to\mathbb R^d$ is represented by a neural network. Following the ODE from $Z_0\sim\pi_0$ gives an endpoint $Z_1$; the goal is to construct $v^\theta$ so that $Z_1\sim\pi_1$. Instead of predicting the whole transport map in one shot, the model predicts local directions along a path.

Sampling requires numerical integration. With forward Euler and step size $\epsilon=1/N$,

$$
Z_{t+\epsilon}
=
Z_t+\epsilon\,v^\theta(Z_t,t),
\qquad
t\in\{0,\epsilon,2\epsilon,\ldots,1\}.
$$

Straight flows have small time-discretization error. In the perfectly straight case, one Euler step from $t=0$ to $t=1$ recovers the endpoint exactly.

## Learning the ODE Drift

One natural way to learn $v$ is to minimize a discrepancy between the terminal distribution of the ODE and $\pi_1$. If $\rho_1^v$ denotes the law of $Z_1$ after solving the ODE with drift $v$, this would take the form

$$
\min_v D(\rho_1^v,\pi_1).
$$

Evaluating $\rho_1^v$, either by sampling or by likelihood computation, requires repeated ODE simulation. The unknown intermediate trajectory is the expensive part.

Rectified flow avoids this inference loop by choosing the intermediate trajectories first. Since only the starting and terminal distributions are prescribed, the intermediate path can be selected as a prior. The simplest choice is a straight path between paired samples from $\pi_0$ and $\pi_1$.

The training problem then becomes local: at a sampled time $t$, show the model a point on the chosen path and ask it to predict the velocity of that path.

## Straight Interpolation

Take a coupling $(X_0,X_1)$ of $\pi_0$ and $\pi_1$. In minibatch training this is often the independent coupling, obtained by drawing $X_0\sim\pi_0$ and $X_1\sim\pi_1$ independently. The linear interpolation is

$$
X_t=tX_1+(1-t)X_0,
\qquad
t\in[0,1].
$$

It follows the simple dynamics

$$
\mathrm dX_t=(X_1-X_0)\,\mathrm dt,
$$

so the point moves in the line direction $X_1-X_0$ with constant speed. For a single fixed endpoint $X_1=x^{\mathrm{data}}$,

$$
X_t=t\,x^{\mathrm{data}}+(1-t)X_0.
$$

The interpolation is a training device. It is allowed to use both endpoints, while the final sampler must move forward from $Z_0$ without knowing the endpoint in advance.

Differentiating gives

$$
\frac{\mathrm d}{\mathrm dt}X_t=x^{\mathrm{data}}-X_0.
$$

To write the velocity as a function of the current state $X_t$, solve

$$
X_0=\frac{X_t-tx^{\mathrm{data}}}{1-t}.
$$

Substituting this into the derivative gives

$$
v^*(x,t)=\frac{x^{\mathrm{data}}-x}{1-t}.
$$

The factor $1/(1-t)$ is the remaining displacement divided by the remaining time.

<figure class="flow-figure flow-figure-narrow">
  <img src="/assets/modules/06-flow-and-diffusion/lines_one_point.png" alt="Straight paths from noise samples to one data point." />
</figure>

## Causalizing the Interpolation

The interpolation $X_t$ already transfers $\pi_0$ to $\pi_1$, but it is not a causal ODE. At time $t<1$, its update direction still depends on the final state $X_1$. In the path picture, this non-causality appears at crossings: when several interpolation paths meet at the same $(x,t)$, their line directions need not agree.

A causal ODE must assign a single drift vector to each state-time pair. Rectified flow obtains such a drift by projecting the interpolation velocity onto functions of $(X_t,t)$:

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
</figure>

The minimizer is the conditional expectation

$$
v^*(x,t)
=
\mathbb E[X_1-X_0\mid X_t=x].
$$

Equivalently, this is ordinary least-squares regression: among all drift fields that can only see $(x,t)$, choose the one closest to the straight-line velocities on average.

Thus $v^*(x,t)$ is the average of the line directions of all interpolation paths passing through $x$ at time $t$. The rectified ODE is

$$
\mathrm dZ_t=v^*(Z_t,t)\,\mathrm dt,
\qquad
Z_0\sim \pi_0.
$$

Its trajectories trace the same density map as the interpolation trajectories, but are rewired at the crossing points so that the dynamics can be simulated causally.

## Training Objective

Parameterize $v$ by a neural network $v^\theta(x,t)$ and estimate the least-squares objective with empirical draws of $(X_0,X_1,t)$:

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

A minibatch implementation is direct:

$$
X_1\sim \pi_1,
\qquad
X_0\sim \pi_0,
\qquad
t\sim\mathrm{Uniform}([0,1]),
$$

then form $X_t=tX_1+(1-t)X_0$ and regress $v^\theta(X_t,t)$ onto $X_1-X_0$.

Training is supervised regression on synthetic intermediate points and their target velocities. No ODE solve is needed during training. Sampling later solves the ODE from fresh noise.

After training, generation uses the learned ODE:

$$
\mathrm dZ_t=v^\theta(Z_t,t)\,\mathrm dt,
\qquad
Z_0\sim \pi_0.
$$

## Marginal Preservation

The interpolation process $X_t$ and the rectified ODE process $Z_t$ usually follow different sample paths, but they have the same marginal distributions:

$$
Z_t\overset{d}{=}X_t,
\qquad
\forall t\in[0,1].
$$

Hence $(Z_0,Z_1)$ is a coupling of $\pi_0$ and $\pi_1$.

This is the marginal preserving property. The rectified velocity creates the same local probability flux as the interpolation velocity, because it averages the line directions conditional on the current location.

The ODE may change individual trajectories, but this flux identity preserves the distribution at each time.

<figure class="flow-figure flow-figure-wide">
  <img src="/assets/modules/06-flow-and-diffusion/rf_x.png" alt="Rectified flow marginal distributions matching the interpolation marginals." />
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

Denote the rectified flow induced from $(X_0,X_1)$ by $\boldsymbol Z=\mathsf{Rectflow}((X_0,X_1))$. Reflow applies the same operator recursively:

$$
\boldsymbol Z^{k+1}
=
\mathsf{Rectflow}((Z_0^k,Z_1^k)),
\qquad
(Z_0^0,Z_1^0)=(X_0,X_1).
$$

In practice, one samples pairs $(Z_0^k,Z_1^k)$ from the $k$-th rectified flow and trains a new flow on those pairs. This procedure straightens the paths of rectified flows as $k$ increases. Flows with nearly straight paths have small time-discretization error; in the perfectly straight case,

$$
Z_t=Z_0+t\,v(Z_0,0),
$$

so the ODE can be solved exactly with a single Euler step.

The learned flow also provides a better coupling between noise and data than the initial independent coupling. Re-training on this coupling removes many unnecessary crossings.

## Diffusion Models

The rectified-flow sampler is an ODE:

$$
\mathrm d Z_t=v_{\mathrm{RF}}(Z_t,t)\,\mathrm dt.
$$

Diffusion models use an SDE of the form

$$
\mathrm d Z_t
=
v(Z_t,t)\,\mathrm dt
+
\sigma(Z_t,t)\,\mathrm d W_t,
$$

where $\sigma$ is a diffusion coefficient and $W_t$ is Brownian motion. ODEs correspond to the special case $\sigma=0$. Conversely, an ODE learned by rectified flow can be converted into a stochastic sampler without changing the time marginals.

Let $\rho_t$ be the density of the interpolation $X_t$. A Langevin correction at time $t$ uses the score $\nabla\log\rho_t$ and gives the combined SDE

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
</figure>

The first drift term is the rectified-flow velocity. The remaining two terms are Langevin dynamics for the time-$t$ density $\rho_t$. In practice this correction is used to reduce drift caused by model approximation or numerical discretization.

The RF drift moves samples forward in time; the score term corrects their position relative to the current density $\rho_t$.

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
> This is Tweedie's formula for the straight interpolation. The RF velocity also gives
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
> The RF velocity therefore provides the score term needed for diffusion-like stochastic samplers, without training a separate score network.

## Homework

[Homework 6: Flow and Diffusion](/homework/06-flow-and-diffusion)

<!-- prettier-ignore-end -->
