---
title: "Homework 6: Flow and Diffusion"
description: "Homework problems covering rectified flow, ODE/SDE sampling, Langevin correction, and diffusion-style score identities."
publish: true
---

[Back to Module 6 notes](/modules/06-flow-and-diffusion)

## Coding

> [!problem|Rectified Flow on 2D Toy Data]
> Open the [2D rectified-flow notebook](https://github.com/lqiang67/rectified-flow/blob/main/examples/train_2d_toys.ipynb).
>
> 1.  Run the default 2D Gaussian-mixture experiment. Show the generated samples and sampling trajectories from the trained flow.
> 2.  Replace the target distribution with one of the 2D toy datasets used in the [Homework 4 GAN notebook](https://drive.google.com/file/d/1CGxibklD0-PjlJrNQsevGGIF5JMnra2c/view?usp=sharing), such as `rings` or `2spirals`. Train the rectified flow again and show the generated results.
> 3.  Implement your own version of `get_loss` for the rectified-flow objective:
>
>     $$
>     \mathcal L(\theta)
>     =
>     \mathbb E_{t\sim \mathrm{Unif}[0,1],\,x_0\sim P_0,\,x_1\sim P_1}
>     \left[
>       \left\|x_1-x_0-v^\theta(x_t,t)\right\|^2
>     \right],
>     \qquad
>     x_t=t x_1+(1-t)x_0.
>     $$
>
>     Replace the original `get_loss` call with your implementation, train the model, and compare the loss curve and generated samples with the original version.
> 4.  Try changing training choices such as learning rate, batch size, number of epochs, or model width. Describe the visible effect on trajectories, mode coverage, and sample quality.

> [!problem|Rectified Flow on MNIST]
> Use the [MNIST rectified-flow Colab notebook](https://colab.research.google.com/drive/1ygE5l4uKXSISISwhmJfUgR2-EDNn3LyT?usp=sharing) to train an image generator from noise to handwritten digits.
>
> 1.  Train the model and plot the training loss. With an A100 GPU, you can train longer, for example 100--200 epochs, to get cleaner samples. Without a fast GPU, it is fine to train for a shorter run, for example 20 epochs.
> 2.  Try comparing generation results after different numbers of training epochs. For example, show samples from an earlier checkpoint and a later checkpoint, and describe how sample quality changes.
> 3.  Compare different numbers of Euler sampling steps `num_steps`. Try a small value such as 2--10 and a larger value such as 50--100, then describe the difference in sample quality and sampling time.
> 4.  The notebook also contains conditional generation with classifier-free guidance (CFG). For a label $y$, the guided velocity is
>
>     $$
>     v_t^{\mathrm{CFG}}(x)
>     =
>     v_t(x,\varnothing)
>     +
>     s\big(v_t(x,y)-v_t(x,\varnothing)\big),
>     $$
>
>     where $s$ is the CFG scale. Try a few values of $s$ and compare how the generated digits change in label accuracy, visual quality, and diversity.

> [!problem|Advanced: FLUX Inference and Editing]
> This optional exploration uses a large pretrained rectified-flow text-to-image model. It requires an A100 GPU with high VRAM in Colab.
>
> - [Image generation notebook](https://colab.research.google.com/github/lqiang67/rectified-flow/blob/main/examples/inference_flux_dev.ipynb)
> - [Image editing notebook](https://colab.research.google.com/github/lqiang67/rectified-flow/blob/main/examples/editing_flux_dev.ipynb)
>
> For text-to-image generation, try a few prompts and explore how the sampler choice and number of sampling steps affect the generated images. You may compare sample quality, prompt following, runtime, and visible artifacts.
>
> For image editing, start from the provided example, then try your own input image or editing prompt. You may also vary parameters such as noise level, `start_t`, `end_t`, `eta_base`, or schedule type. Show representative editing results and briefly describe what changed.

## Theory

Unless stated otherwise, let $X_0\sim P_0$ be a noise sample, let $X_1\sim P_{\mathrm{data}}$ be a data sample, and assume $X_0$ and $X_1$ are independent. The straight interpolation is

$$
X_t=tX_1+(1-t)X_0,\qquad t\in[0,1].
$$

The ideal rectified-flow velocity is

$$
v_t^*(x)
=
\mathbb E[X_1-X_0\mid X_t=x].
$$

A learned rectified-flow model uses a neural velocity field $v^\theta(x,t)$ to approximate $v_t^*(x)$.

> [!problem|Straight Paths to One Data Point]
> Suppose the data distribution is a single point mass at $x^\star\in\mathbb R^d$, so $X_1=x^\star$ always. The interpolation is
>
> $$
> X_t=t x^\star+(1-t)X_0.
> $$
>
> 1.  Derive the time-dependent velocity field $v_t^*(x)$ whose ODE $\dot X_t=v_t^*(X_t)$ follows this straight path.
> 2.  Consider any partition $0=t_0<t_1<\cdots<t_N=1$ and the Euler update
>
>     $$
>     X_{t_{k+1}}
>     =
>     X_{t_k}
>     +
>     (t_{k+1}-t_k)v_{t_k}^*(X_{t_k}).
>     $$
>
>     Show that the update lands exactly on the straight interpolation at every grid point, $X_{t_k}=t_kx^\star+(1-t_k)X_0$. Explain why this special case has no Euler discretization error.

> [!problem|Intersections and Averaged Velocity]
> With multiple data points, different straight interpolations can pass through the same location at the same time. At that state-time pair, the ODE still needs one velocity value.
>
> 1.  Give an example in $\mathbb R$ or $\mathbb R^2$ of two pairs $(X_0,X_1)$ and $(X_0',X_1')$ whose straight interpolations meet at the same location $x$ at the same time $t$.
> 2.  Explain why a deterministic ODE $\dot Z_t=v_t(Z_t)$ cannot assign two different velocities to the same state-time pair $(x,t)$.
> 3.  Explain how
>
>     $$
>     v_t^*(x)=\mathbb E[X_1-X_0\mid X_t=x]
>     $$
>
>     assigns a single velocity by averaging the directions of the paths that pass through $x$ at time $t$.

> [!problem|Bias-Variance Decomposition for Regression]
> A basic regression fact underlies rectified-flow training. Let $(U,V)$ be any pair of random variables and consider
>
> $$
> \min_f\; \mathbb E\big[\|V-f(U)\|^2\big],
> $$
>
> where $f$ ranges over measurable functions.
>
> 1.  Show the decomposition
>
>     $$
>     \mathbb E[\|V-f(U)\|^2]
>     =
>     \mathbb E[\|V-\mathbb E[V\mid U]\|^2]
>     +
>     \mathbb E[\|\mathbb E[V\mid U]-f(U)\|^2].
>     $$
>
>     State what the two terms represent.
> 2.  Conclude that the best predictor is $f^*(u)=\mathbb E[V\mid U=u]$.
> 3.  Apply this result with $U=(X_t,t)$ and $V=X_1-X_0$ to justify the rectified-flow training target.

> [!problem|Data Prediction, Noise Prediction, and Velocity]
> For $X_t=tX_1+(1-t)X_0$ and $t\in(0,1)$, define the two prediction targets
>
> $$
> \hat x_{\mathrm{data}}(x,t)=\mathbb E[X_1\mid X_t=x],
> \qquad
> \hat x_{\mathrm{noise}}(x,t)=\mathbb E[X_0\mid X_t=x].
> $$
>
> These two quantities are often called data prediction and noise prediction. The key fact is the linearity of conditional expectation.
>
> 1.  Condition on the event $X_t=x$ and take conditional expectation on both sides of
>
>     $$
>     X_t=tX_1+(1-t)X_0.
>     $$
>
>     Use linearity of expectation to show that
>
>     $$
>     x=t\,\hat x_{\mathrm{data}}(x,t)+(1-t)\,\hat x_{\mathrm{noise}}(x,t).
>     $$
>
> 2.  Starting from the rectified-flow target
>
>     $$
>     v_t^*(x)=\mathbb E[X_1-X_0\mid X_t=x],
>     $$
>
>     use the same linearity to derive
>
>     $$
>     v_t^*(x)
>     =
>     \hat x_{\mathrm{data}}(x,t)-\hat x_{\mathrm{noise}}(x,t)
>     =
>     \frac{\hat x_{\mathrm{data}}(x,t)-x}{1-t}
>     =
>     \frac{x-\hat x_{\mathrm{noise}}(x,t)}{t}.
>     $$
>
> 3.  Suppose a neural network predicts either $\hat x_{\mathrm{data}}(x,t)$ or $\hat x_{\mathrm{noise}}(x,t)$. Explain how to convert its output into a velocity prediction.

> [!problem|Time Weighting in the Training Loss]
> Consider the weighted objective
>
> $$
> \min_v
> \int_0^1
> w_t\,
> \mathbb E\!\left[
>   \|X_1-X_0-v_t(X_t)\|^2
> \right]\,\mathrm d t,
> \qquad
> w_t>0.
> $$
>
> 1.  If each time $t$ has its own independent function $v_t(\cdot)$, show that the minimizer is still
>
>     $$
>     v_t^*(x)=\mathbb E[X_1-X_0\mid X_t=x].
>     $$
>
> 2.  Now suppose one neural network $v^\theta(x,t)$ is shared across all times. Explain why the choice of $w_t$ can affect the learned model in this finite-capacity setting.

> [!problem|Closed-Form Gaussian-to-Gaussian Case]
> Let $X_0\sim\mathcal N(0,1)$ and $X_1\sim\mathcal N(\mu,1)$ be independent, and let $X_t=tX_1+(1-t)X_0$.
>
> 1.  Compute $\mathbb E[X_1\mid X_t=x]$ and $\mathbb E[X_0\mid X_t=x]$.
> 2.  Derive
>
>     $$
>     v^*(x,t)=\mathbb E[X_1-X_0\mid X_t=x],
>     $$
>
>     and write it in affine form $v^*(x,t)=a(t)x+b(t)$.
> 3.  Check that this velocity gives the correct distributions at the two ends of the path: $\mathcal N(0,1)$ at $t=0$ and $\mathcal N(\mu,1)$ at $t=1$.

> [!problem|Two-Point Data Mixture]
> Let $X_0\sim\mathcal N(0,I_d)$, and suppose $X_1$ is either $x^\star$ or $y^\star$ with probability $1/2$ each. For $t\in[0,1)$, let $X_t=tX_1+(1-t)X_0$.
>
> 1.  Use Bayes' rule to compute
>
>     $$
>     \mathbb P(X_1=x^\star\mid X_t=x)
>     =
>     \frac{
>       \exp\!\left(-\frac{\|x-tx^\star\|^2}{2(1-t)^2}\right)
>     }{
>       \exp\!\left(-\frac{\|x-tx^\star\|^2}{2(1-t)^2}\right)
>       +
>       \exp\!\left(-\frac{\|x-ty^\star\|^2}{2(1-t)^2}\right)
>     }.
>     $$
>
> 2.  Let $\omega_u(x,t)=\mathbb P(X_1=u\mid X_t=x)$ for $u\in\{x^\star,y^\star\}$. Show that
>
>     $$
>     v^*(x,t)
>     =
>     \sum_{u\in\{x^\star,y^\star\}}
>     \omega_u(x,t)\frac{u-x}{1-t}.
>     $$
>
> 3.  Interpret the formula as a weighted average of two velocities, one pointing toward each possible data value.

> [!problem|Tweedie's Identity from Rectified Flow]
> Assume $X_0\sim\mathcal N(0,I)$ and $X_1\sim P_{\mathrm{data}}$. Let $\rho_t$ be the density of $X_t=tX_1+(1-t)X_0$.
>
> 1.  Show that, up to a normalizing constant,
>
>     $$
>     \rho_t(x)
>     \propto
>     \int
>     \rho_1(x_1)
>     \exp\!\left(
>       -\frac{\|x-tx_1\|^2}{2(1-t)^2}
>     \right)\,\mathrm d x_1.
>     $$
>
> 2.  Differentiate $\log\rho_t(x)$ to obtain Tweedie's identity:
>
>     $$
>     \nabla\log\rho_t(x)
>     =
>     \mathbb E\!\left[
>       \frac{tX_1-x}{(1-t)^2}
>       \,\middle|\,
>       X_t=x
>     \right].
>     $$
>
> 3.  Use $\mathbb E[X_1\mid X_t=x]=x+(1-t)v^*(x,t)$ to show
>
>     $$
>     \nabla\log\rho_t(x)
>     =
>     \frac{t\,v^*(x,t)-x}{1-t}.
>     $$
>
>     Use this relation to express the score needed for Langevin correction in terms of the rectified-flow velocity.

> [!problem|Noise Schedule Near t=1]
> In RF plus Langevin dynamics, the Langevin drift uses the score $\nabla\log\rho_t(x)$. From the previous problem,
>
> $$
> \nabla\log\rho_t(x)=\frac{t\,v^*(x,t)-x}{1-t},
> $$
>
> so the Langevin drift magnitude can scale like $\sigma_t^2/(1-t)$ near $t=1$.
>
> 1.  Suppose $\sigma_t=c(1-t)^\alpha$ with $c>0$. Find the condition on $\alpha$ such that $\sigma_t^2/(1-t)$ remains bounded as $t\to 1$.
> 2.  Explain what happens when $\alpha<1/2$, $\alpha=1/2$, and $\alpha>1/2$.
> 3.  Explain why choosing $\sigma_t\to 0$ as $t\to 1$ helps avoid noisy final samples.

> [!problem|Euler-Maruyama Step for RF plus Langevin]
> Consider one time step of size $\Delta t$ at state $x$ and time $t$. The hybrid RF plus Langevin update is
>
> $$
> Z_{t+\Delta t}
> =
> x
> +
> \left[
>   v^*(x,t)+\sigma_t^2\nabla\log\rho_t(x)
> \right]\Delta t
> +
> \sqrt{2}\,\sigma_t\sqrt{\Delta t}\,\xi,
> \qquad
> \xi\sim\mathcal N(0,I).
> $$
>
> Substitute
>
> $$
> \nabla\log\rho_t(x)=\frac{t\,v^*(x,t)-x}{1-t}
> $$
>
> to write the same update using only $v^*(x,t)$, $x$, $t$, $\sigma_t$, and $\Delta t$. Then write the conditional mean and covariance of $Z_{t+\Delta t}$ given $Z_t=x$.

> [!problem|Single-Point Data with Langevin Correction]
> Let $P_{\mathrm{data}}=\delta_{x^\star}$ and $X_0\sim\mathcal N(0,I)$. Then $X_t=t x^\star+(1-t)X_0$.
>
> 1.  Show that
>
>     $$
>     v^*(x,t)=\frac{x^\star-x}{1-t},
>     \qquad
>     \nabla\log\rho_t(x)=\frac{t x^\star-x}{(1-t)^2}.
>     $$
>
> 2.  The hybrid drift is
>
>     $$
>     b(x,t)
>     =
>     \frac{x^\star-x}{1-t}
>     +
>     \sigma_t^2\frac{t x^\star-x}{(1-t)^2}.
>     $$
>
>     Explain how the second term changes the deterministic RF motion, and how its strength depends on $t$ and $\sigma_t$.
> 3.  Verify that when $\sigma_t\equiv0$, the dynamics reduce to the exact straight-line ODE from the first theory problem.

> [!problem|Discrete Langevin Stability]
> Consider the unadjusted Langevin update
>
> $$
> x_{k+1}
> =
> x_k+\epsilon\nabla\log p(x_k)+\sqrt{2\epsilon}\,\xi_k,
> \qquad
> \xi_k\sim\mathcal N(0,I),
> $$
>
> where
>
> $$
> p(x)\propto \exp\!\left(-\frac12\|x-\mu\|^2\right),
> \qquad
> \nabla\log p(x)=\mu-x.
> $$
>
> 1.  Show that the update can be written as
>
>     $$
>     x_{k+1}=(1-\epsilon)x_k+\epsilon\mu+\sqrt{2\epsilon}\,\xi_k.
>     $$
>
> 2.  In one dimension, show that mean-square stability holds exactly when $0<\epsilon<2$. You may use $y_k=x_k-\mu$.
> 3.  Compute the stationary variance in one dimension:
>
>     $$
>     \mathrm{Var}_\infty
>     =
>     \frac{2\epsilon}{1-(1-\epsilon)^2}
>     =
>     \frac{1}{1-\epsilon/2}.
>     $$
>
>     Explain why this approaches the correct variance as $\epsilon\to0$, but is larger than $1$ for any fixed stable step size $\epsilon>0$. State the coordinate-wise extension to $\mathbb R^d$.

> [!problem|Continuity Equation]
> Let $p_t$ be the density of a smooth process $X_t$, and define
>
> $$
> v_t^*(x)=\mathbb E[\dot X_t\mid X_t=x].
> $$
>
> Let $h:\mathbb R^d\to\mathbb R$ be smooth and compactly supported.
>
> 1.  Show that
>
>     $$
>     \frac{d}{dt}\mathbb E[h(X_t)]
>     =
>     \int h(x)\,\partial_t p_t(x)\,\mathrm d x.
>     $$
>
> 2.  Use the chain rule and conditional expectation to show that
>
>     $$
>     \frac{d}{dt}\mathbb E[h(X_t)]
>     =
>     \int \nabla h(x)^\top v_t^*(x)p_t(x)\,\mathrm d x.
>     $$
>
> 3.  Apply integration by parts to derive the continuity equation
>
>     $$
>     \partial_t p_t(x)
>     =
>     -\nabla\cdot\big(v_t^*(x)p_t(x)\big).
>     $$

> [!problem|Neural ODE Likelihood]
> Let $Z_t$ follow the ODE
>
> $$
> \dot Z_t=v^\theta(Z_t,t),
> \qquad
> Z_0\sim p_0,
> $$
>
> and let $p_t^\theta$ be the density of $Z_t$.
>
> 1.  Using change of variables along the ODE flow, derive
>
>     $$
>     \log p_t^\theta(x)
>     =
>     \log p_0(z_0)
>     -
>     \int_0^t
>     \operatorname{tr}\!\left(\nabla_x v^\theta(z_\tau,\tau)\right)
>     \,\mathrm d \tau,
>     $$
>
>     where $z_\tau$ is the trajectory ending at $z_t=x$.
> 2.  Specialize to $d=1$ and $v^\theta(z,t)=a(t)z$. Write $\log p_t^\theta(x)$ in terms of $\log p_0(z_0)$ and $\int_0^t a(\tau)\,\mathrm d \tau$.
> 3.  Explain why likelihood training for a Neural ODE is usually more expensive than rectified-flow regression training.
