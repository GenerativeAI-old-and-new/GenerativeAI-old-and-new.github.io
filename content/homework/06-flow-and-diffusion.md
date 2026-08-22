---
title: "Homework 6: Flow and Diffusion"
description: "Rectified-flow regression, ODE/SDE sampling, Langevin correction, and score identities."
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
>
> 4.  Try changing training choices such as learning rate, batch size, number of epochs, or model width. Describe the visible effect on trajectories, mode coverage, and sample quality.

<!--
> [!solution]- Solution
>
> For the default straight interpolation, `get_loss` can be written as follows:
>
> ```python
> def get_loss_manual(rectified_flow, x_0, x_1):
>     batch_size = x_1.shape[0]
>     t = rectified_flow.sample_train_time(batch_size)
>     x_t, target_velocity = rectified_flow.get_interpolation(x_0, x_1, t)
>     predicted_velocity = rectified_flow.get_velocity(x_t, t)
>
>     per_sample_loss = (predicted_velocity - target_velocity).square()
>     per_sample_loss = per_sample_loss.flatten(1).mean(dim=1)
>     return per_sample_loss.mean()
> ```
>
> Here `target_velocity` is $x_1-x_0$. With uniform time sampling and uniform time weights, this is the loss used by the notebook. The code averages over coordinates, while the displayed objective uses a squared norm; the two differ only by a constant factor.
>
> To compare the two implementations, use the same seed, data, model, and number of updates. Their loss curves and generated samples should be similar, apart from randomness in the sampled batches and times.
>
> In the first experiment, the final samples should cover both components of the target Gaussian mixture. After replacing the target with `rings` or `2spirals`, check whether the generated samples cover the full shape rather than only part of it. For the hyperparameter study, change one setting at a time. A learning rate that is too large often gives an unstable loss; a very small one learns slowly. More training or a wider MLP can improve the fit when the model is undertrained or too small.
-->

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

<!--
> [!solution]- Solution
>
> The loss curve and image grids come from the training run. Compare an early and a later checkpoint using the same sampling settings. A lower training loss is useful only when the generated digits also improve.
>
> With $K$ uniform Euler steps and $\Delta t=1/K$, sampling uses
>
> $$
> x_{k+1}=x_k+\Delta t\,v^\theta(x_k,t_k).
> $$
>
> Increasing $K$ reduces Euler discretization error and costs $K$ network evaluations. Very small values can reduce sample quality. Once the discretization error is small relative to the model error, adding more steps has little effect.
>
> Classifier-free guidance can be written as
>
> $$
> v_t^{\mathrm{CFG}}(x)
> =(1-s)v_t(x,\varnothing)+s\,v_t(x,y).
> $$
>
> Thus $s=0$ gives unconditional generation, $s=1$ gives the ordinary conditional velocity, and $s>1$ extrapolates in the class-conditioned direction. Larger values can improve label agreement, but excessive guidance can reduce diversity or introduce artifacts. Use the same labels and initial noise when comparing scales.
-->

> [!problem|Advanced: FLUX Inference and Editing]
> This optional exploration uses a large pretrained rectified-flow text-to-image model. It requires an A100 GPU with high VRAM in Colab.
>
> - [Image generation notebook](https://colab.research.google.com/github/lqiang67/rectified-flow/blob/main/examples/inference_flux_dev.ipynb)
> - [Image editing notebook](https://colab.research.google.com/github/lqiang67/rectified-flow/blob/main/examples/editing_flux_dev.ipynb)
>
> For text-to-image generation, try a few prompts and explore how the sampler choice and number of sampling steps affect the generated images. You may compare sample quality, prompt following, runtime, and visible artifacts.
>
> For image editing, start from the provided example, then try your own input image or editing prompt. You may also vary parameters such as noise level, `start_t`, `end_t`, `eta_base`, or schedule type. Show representative editing results and briefly describe what changed.

<!--
> [!solution]- Solution
>
> There is no single image for this problem. For generation, keep the prompt and seed fixed while changing either the sampler or the number of steps. Report the runtime with each image. More steps require more model evaluations; their benefit should be read from the resulting images.
>
> For editing, keep the input image fixed and change one setting at a time. Compare how much of the original composition remains, whether the requested edit appears, and whether unrelated regions change. The noise level and the interval set by `start_t` and `end_t` control how strongly the editing field is applied. A complete answer shows the original image beside the edited images and states which parameter changed.
-->

## Theory

Unless stated otherwise, let $X_0\sim P_0$ and $X_1\sim P_{\mathrm{data}}$ be independent random vectors in $\mathbb R^d$. The straight interpolation is

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

<!--
> [!solution]- Solution
>
> Since $X_1=x^\star$,
>
> $$
> \dot X_t=x^\star-X_0.
> $$
>
> For $t<1$, the interpolation gives $X_0=(x-tx^\star)/(1-t)$ when $X_t=x$. Hence
>
> $$
> v_t^*(x)
> =x^\star-\frac{x-tx^\star}{1-t}
> =\frac{x^\star-x}{1-t}.
> $$
>
> Along the prescribed path this velocity is constant:
>
> $$
> v_t^*(X_t)
> =\frac{x^\star-[t x^\star+(1-t)X_0]}{1-t}
> =x^\star-X_0.
> $$
>
> Suppose $X_{t_k}=t_kx^\star+(1-t_k)X_0$. One Euler step gives
>
> $$
> \begin{aligned}
> X_{t_{k+1}}
> &=X_{t_k}+(t_{k+1}-t_k)(x^\star-X_0)\\
> &=t_{k+1}x^\star+(1-t_{k+1})X_0.
> \end{aligned}
> $$
>
> The claim follows by induction from $X_{t_0}=X_0$. Euler is exact here because the velocity along each path is constant.
-->

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

<!--
> [!solution]- Solution
>
> In $\mathbb R^2$, consider the two pairs
>
> $$
> (X_0,X_1)=((0,0),(2,0)),
> \qquad
> (X_0',X_1')=((1,1),(1,-1)).
> $$
>
> Their paths are $(2t,0)$ and $(1,1-2t)$. Both pass through $(1,0)$ at $t=1/2$, but their velocities there are $(2,0)$ and $(0,-2)$.
>
> A deterministic vector field is a function of $(x,t)$, so it can return only one vector at $((1,0),1/2)$. Rectified flow uses
>
> $$
> v_t^*(x)
> =\sum_i
> \mathbb P(\text{path }i\mid X_t=x)(X_1^{(i)}-X_0^{(i)}).
> $$
>
> It therefore averages the possible velocities using their conditional probabilities. If the two paths above have equal conditional probability, the assigned velocity is $(1,-1)$.
-->

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
>
> 2.  Conclude that the best predictor is $f^*(u)=\mathbb E[V\mid U=u]$.
> 3.  For a fixed time $t$, apply this result with $U=X_t$ and $V=X_1-X_0$ to justify the rectified-flow training target.

<!--
> [!solution]- Solution
>
> Let $m(U)=\mathbb E[V\mid U]$. Write
>
> $$
> V-f(U)=[V-m(U)]+[m(U)-f(U)].
> $$
>
> After expanding the squared norm, the cross term vanishes because
>
> $$
> \begin{aligned}
> &\mathbb E\!\left[(V-m(U))^\top(m(U)-f(U))\right]\\
> &\quad=
> \mathbb E\!\left[
> \mathbb E[V-m(U)\mid U]^\top(m(U)-f(U))
> \right]
> =0.
> \end{aligned}
> $$
>
> This gives
>
> $$
> \mathbb E\|V-f(U)\|^2
> =
> \mathbb E\|V-m(U)\|^2
> +
> \mathbb E\|m(U)-f(U)\|^2.
> $$
>
> The first term is uncertainty in $V$ that cannot be removed after observing $U$. The second is the prediction error of $f$. Only the second depends on $f$, so the minimizer is $f^*(U)=m(U)$ almost surely.
>
> For each fixed $t$, taking $U=X_t$ and $V=X_1-X_0$ gives
>
> $$
> \mathbb E[X_1-X_0\mid X_t=x]=v_t^*(x),
> $$
>
> which is the target learned by rectified-flow regression.
-->

> [!problem|Data Prediction, Noise Prediction, and Velocity]
> For $X_t=tX_1+(1-t)X_0$ and $t\in(0,1)$, define the two prediction targets
>
> $$
> \hat x_{\mathrm{data}}(x,t)=\mathbb E[X_1\mid X_t=x],
> \qquad
> \hat x_{\mathrm{noise}}(x,t)=\mathbb E[X_0\mid X_t=x].
> $$
>
> These two quantities are often called data prediction and noise prediction. Use the linearity of conditional expectation in the following derivation.
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

<!--
> [!solution]- Solution
>
> Conditioning the interpolation on $X_t=x$ gives
>
> $$
> x
> =t\,\mathbb E[X_1\mid X_t=x]
> +(1-t)\,\mathbb E[X_0\mid X_t=x]
> =t\,\hat x_{\mathrm{data}}+(1-t)\,\hat x_{\mathrm{noise}}.
> $$
>
> The velocity is
>
> $$
> v_t^*(x)
> =\hat x_{\mathrm{data}}-\hat x_{\mathrm{noise}}.
> $$
>
> Solving the interpolation identity for either prediction gives
>
> $$
> v_t^*(x)
> =\frac{\hat x_{\mathrm{data}}(x,t)-x}{1-t}
> =\frac{x-\hat x_{\mathrm{noise}}(x,t)}{t}.
> $$
>
> Therefore a data-prediction network is converted by subtracting $x$ and dividing by $1-t$. A noise-prediction network is converted by subtracting its output from $x$ and dividing by $t$. These formulas apply for $0<t<1$.
-->

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

<!--
> [!solution]- Solution
>
> Fix a time $t$. Since $w_t>0$ is a constant with respect to $v_t$, multiplying that time's regression loss by $w_t$ does not change its minimizer. The bias-variance decomposition therefore gives
>
> $$
> v_t^*(x)=\mathbb E[X_1-X_0\mid X_t=x].
> $$
>
> With a shared network, the same parameters must fit all times. The weight $w_t$ changes how strongly errors at time $t$ contribute to the objective and to each gradient update. A finite network may not fit every time equally well, so changing $w_t$ changes the compromise selected during training.
-->

> [!problem|Closed-Form Gaussian-to-Gaussian Case]
> Let $X_0\sim\mathcal N(0,1)$ and $X_1\sim\mathcal N(\mu,1)$ be independent, and let $X_t=tX_1+(1-t)X_0$.
>
> 1.  Compute $\mathbb E[X_1\mid X_t=x]$ and $\mathbb E[X_0\mid X_t=x]$.
> 2.  Derive
>
>     $$
>     v_t^*(x)=\mathbb E[X_1-X_0\mid X_t=x],
>     $$
>
>     and write it in affine form $v_t^*(x)=a(t)x+b(t)$.
>
> 3.  Check that this velocity gives the correct distributions at the two ends of the path: $\mathcal N(0,1)$ at $t=0$ and $\mathcal N(\mu,1)$ at $t=1$.

<!--
> [!solution]- Solution
>
> Define
>
> $$
> D(t)=t^2+(1-t)^2.
> $$
>
> Since $X_t$ is Gaussian,
>
> $$
> \mathbb E[X_t]=t\mu,
> \qquad
> \operatorname{Var}(X_t)=D(t).
> $$
>
> Also,
>
> $$
> \operatorname{Cov}(X_1,X_t)=t,
> \qquad
> \operatorname{Cov}(X_0,X_t)=1-t.
> $$
>
> The Gaussian conditioning formula gives
>
> $$
> \mathbb E[X_1\mid X_t=x]
> =\mu+\frac{t}{D(t)}(x-t\mu),
> $$
>
> $$
> \mathbb E[X_0\mid X_t=x]
> =\frac{1-t}{D(t)}(x-t\mu).
> $$
>
> Subtracting the two expressions,
>
> $$
> v_t^*(x)
> =\frac{2t-1}{D(t)}x+\frac{1-t}{D(t)}\mu.
> $$
>
> Thus
>
> $$
> a(t)=\frac{2t-1}{D(t)},
> \qquad
> b(t)=\frac{(1-t)\mu}{D(t)}.
> $$
>
> Let $m_t$ and $q_t$ be the mean and variance under the affine ODE $\dot Z_t=a(t)Z_t+b(t)$. They satisfy
>
> $$
> \dot m_t=a(t)m_t+b(t),
> \qquad
> \dot q_t=2a(t)q_t.
> $$
>
> Substitution shows that $m_t=t\mu$ and $q_t=D(t)$ solve these equations with $m_0=0$ and $q_0=1$. Hence
>
> $$
> Z_t\sim\mathcal N\!\left(t\mu,\,t^2+(1-t)^2\right).
> $$
>
> At $t=0$ this is $\mathcal N(0,1)$, and at $t=1$ it is $\mathcal N(\mu,1)$.
-->

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
>     v_t^*(x)
>     =
>     \sum_{u\in\{x^\star,y^\star\}}
>     \omega_u(x,t)\frac{u-x}{1-t}.
>     $$
>
> 3.  Interpret the formula as a weighted average of two velocities, one pointing toward each possible data value.

<!--
> [!solution]- Solution
>
> Conditional on $X_1=u$,
>
> $$
> X_t\mid X_1=u
> \sim
> \mathcal N\!\left(tu,(1-t)^2I_d\right).
> $$
>
> The two values of $X_1$ have equal prior probability, so Bayes' rule gives
>
> $$
> \mathbb P(X_1=x^\star\mid X_t=x)
> =
> \frac{
> \exp\!\left(-\frac{\|x-tx^\star\|^2}{2(1-t)^2}\right)
> }{
> \exp\!\left(-\frac{\|x-tx^\star\|^2}{2(1-t)^2}\right)
> +
> \exp\!\left(-\frac{\|x-ty^\star\|^2}{2(1-t)^2}\right)
> }.
> $$
>
> For a fixed value $X_1=u$, the interpolation implies
>
> $$
> X_1-X_0=\frac{u-X_t}{1-t}.
> $$
>
> Taking the conditional expectation therefore yields
>
> $$
> v_t^*(x)
> =
> \sum_{u\in\{x^\star,y^\star\}}
> \omega_u(x,t)\frac{u-x}{1-t}.
> $$
>
> Each candidate data value contributes a velocity pointing from $x$ toward that value. Its posterior probability $\omega_u(x,t)$ determines the weight.
-->

> [!problem|Tweedie's Identity from Rectified Flow]
> Assume $X_0\sim\mathcal N(0,I_d)$ and $X_1\sim P_{\mathrm{data}}$. Let $\rho_t$ be the density of $X_t=tX_1+(1-t)X_0$.
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
>     \nabla \log\rho_t(x)
>     =
>     \mathbb E\!\left[
>       \frac{tX_1-x}{(1-t)^2}
>       \,\middle|\,
>       X_t=x
>     \right].
>     $$
>
> 3.  Use $\mathbb E[X_1\mid X_t=x]=x+(1-t)v_t^*(x)$ to show
>
>     $$
>     \nabla \log\rho_t(x)
>     =
>     \frac{t\,v_t^*(x)-x}{1-t}.
>     $$
>
>     Use this relation to express the score needed for Langevin correction in terms of the rectified-flow velocity.

<!--
> [!solution]- Solution
>
> Given $X_1=x_1$, the variable $X_t$ is Gaussian with mean $tx_1$ and covariance $(1-t)^2I_d$. Marginalizing over $X_1$ gives
>
> $$
> \rho_t(x)
> =C_t\int
> \rho_1(x_1)
> \exp\!\left(-\frac{\|x-tx_1\|^2}{2(1-t)^2}\right)
> \,\mathrm d x_1,
> $$
>
> where $C_t$ does not depend on $x$. Differentiating under the integral and dividing by $\rho_t(x)$ gives
>
> $$
> \nabla \log\rho_t(x)
> =
> \mathbb E\!\left[
> \frac{tX_1-x}{(1-t)^2}
> \mathrel{\Big|}X_t=x
> \right].
> $$
>
> From data prediction,
>
> $$
> \mathbb E[X_1\mid X_t=x]
> =x+(1-t)v_t^*(x).
> $$
>
> Substituting this identity gives
>
> $$
> \begin{aligned}
> \nabla \log\rho_t(x)
> &=\frac{t[x+(1-t)v_t^*(x)]-x}{(1-t)^2}\\
> &=\frac{t\,v_t^*(x)-x}{1-t}.
> \end{aligned}
> $$
>
> Thus the Langevin score can be computed directly from the rectified-flow velocity; a separate score network is not required.
-->

> [!problem|Noise Schedule Near t=1]
> In RF plus Langevin dynamics, the Langevin drift uses the score $\nabla \log\rho_t(x)$. From the previous problem,
>
> $$
> \nabla \log\rho_t(x)=\frac{t\,v_t^*(x)-x}{1-t},
> $$
>
> so the Langevin drift magnitude can scale like $\sigma_t^2/(1-t)$ near $t=1$.
>
> 1.  Suppose $\sigma_t=c(1-t)^\alpha$ with $c>0$. Find the condition on $\alpha$ such that $\sigma_t^2/(1-t)$ remains bounded as $t\to 1$.
> 2.  Explain what happens when $\alpha<1/2$, $\alpha=1/2$, and $\alpha>1/2$.
> 3.  Explain why choosing $\sigma_t\to 0$ as $t\to 1$ helps avoid noisy final samples.

<!--
> [!solution]- Solution
>
> With $\sigma_t=c(1-t)^\alpha$,
>
> $$
> \frac{\sigma_t^2}{1-t}
> =c^2(1-t)^{2\alpha-1}.
> $$
>
> This remains bounded as $t\to1$ exactly when
>
> $$
> \alpha\geq\frac12.
> $$
>
> If $\alpha<1/2$, the factor diverges. If $\alpha=1/2$, it approaches $c^2$. If $\alpha>1/2$, it approaches zero.
>
> The random increment has size $\sqrt{2}\,\sigma_t\sqrt{\Delta t}$. Choosing a schedule with $\sigma_t\to0$ suppresses this added noise near the final sample. The condition $\alpha\geq1/2$ also keeps the score correction from becoming unbounded under the scaling above.
-->

> [!problem|Euler-Maruyama Step for RF plus Langevin]
> Consider one time step of size $\Delta t$ at state $x$ and time $t$. The hybrid RF plus Langevin update is
>
> $$
> Z_{t+\Delta t}
> =
> x
> +
> \left[
>   v_t^*(x)+\sigma_t^2\nabla \log\rho_t(x)
> \right]\Delta t
> +
> \sqrt{2}\,\sigma_t\sqrt{\Delta t}\,\xi,
> \qquad
> \xi\sim\mathcal N(0,I_d).
> $$
>
> Substitute
>
> $$
> \nabla \log\rho_t(x)=\frac{t\,v_t^*(x)-x}{1-t}
> $$
>
> to write the same update using only $v_t^*(x)$, $x$, $t$, $\sigma_t$, and $\Delta t$. Then write the conditional mean and covariance of $Z_{t+\Delta t}$ given $Z_t=x$.

<!--
> [!solution]- Solution
>
> Substituting the score identity gives
>
> $$
> Z_{t+\Delta t}
> =x+
> \left[
> v_t^*(x)
> +\sigma_t^2\frac{t\,v_t^*(x)-x}{1-t}
> \right]\Delta t
> +\sqrt{2}\,\sigma_t\sqrt{\Delta t}\,\xi.
> $$
>
> Conditioned on $Z_t=x$, all terms except $\xi$ are fixed. Therefore
>
> $$
> \mathbb E[Z_{t+\Delta t}\mid Z_t=x]
> =x+
> \left[
> v_t^*(x)
> +\sigma_t^2\frac{t\,v_t^*(x)-x}{1-t}
> \right]\Delta t,
> $$
>
> and
>
> $$
> \operatorname{Cov}(Z_{t+\Delta t}\mid Z_t=x)
> =2\sigma_t^2\Delta t\,I_d.
> $$
-->

> [!problem|Single-Point Data with Langevin Correction]
> Let $P_{\mathrm{data}}=\delta_{x^\star}$ and $X_0\sim\mathcal N(0,I_d)$. Then $X_t=t x^\star+(1-t)X_0$.
>
> 1.  Show that
>
>     $$
>     v_t^*(x)=\frac{x^\star-x}{1-t},
>     \qquad
>     \nabla \log\rho_t(x)=\frac{t x^\star-x}{(1-t)^2}.
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
>
> 3.  Verify that when $\sigma_t\equiv0$, the dynamics reduce to the exact straight-line ODE from the first theory problem.

<!--
> [!solution]- Solution
>
> Since $X_1=x^\star$,
>
> $$
> v_t^*(x)=\frac{x^\star-x}{1-t}.
> $$
>
> Also,
>
> $$
> X_t\sim\mathcal N\!\left(tx^\star,(1-t)^2I_d\right),
> $$
>
> so its score is
>
> $$
> \nabla \log\rho_t(x)
> =-\frac{x-tx^\star}{(1-t)^2}
> =\frac{tx^\star-x}{(1-t)^2}.
> $$
>
> The RF term points toward $x^\star$. The Langevin term points toward $tx^\star$, the mean of the distribution at the current time. Its coefficient is $\sigma_t^2/(1-t)^2$, so its effect near $t=1$ depends strongly on how quickly $\sigma_t$ decreases.
>
> When $\sigma_t=0$, the stochastic term and the score correction both disappear. The ODE becomes
>
> $$
> \dot Z_t=\frac{x^\star-Z_t}{1-t},
> $$
>
> whose solution from $Z_0$ is $Z_t=tx^\star+(1-t)Z_0$.
-->

> [!problem|Discrete Langevin Stability]
> Consider the unadjusted Langevin update
>
> $$
> x_{k+1}
> =
> x_k+\epsilon\nabla \log p(x_k)+\sqrt{2\epsilon}\,\xi_k,
> \qquad
> \xi_k\sim\mathcal N(0,I_d),
> $$
>
> where
>
> $$
> p(x)\propto \exp\!\left(-\frac12\|x-\mu\|^2\right),
> \qquad
> \nabla \log p(x)=\mu-x.
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
>     \operatorname{Var}_\infty
>     =
>     \frac{2\epsilon}{1-(1-\epsilon)^2}
>     =
>     \frac{1}{1-\epsilon/2}.
>     $$
>
>     Explain why this approaches the correct variance as $\epsilon\to0$, but is larger than $1$ for any fixed stable step size $\epsilon>0$. State the coordinate-wise extension to $\mathbb R^d$.

<!--
> [!solution]- Solution
>
> Substituting $\nabla \log p(x_k)=\mu-x_k$ gives
>
> $$
> x_{k+1}
> =(1-\epsilon)x_k+\epsilon\mu+\sqrt{2\epsilon}\,\xi_k.
> $$
>
> In one dimension, let $y_k=x_k-\mu$. Then
>
> $$
> y_{k+1}=(1-\epsilon)y_k+\sqrt{2\epsilon}\,\xi_k.
> $$
>
> The second moment follows the recursion
>
> $$
> \mathbb E[y_{k+1}^2]
> =(1-\epsilon)^2\mathbb E[y_k^2]+2\epsilon.
> $$
>
> A finite stationary second moment exists exactly when
>
> $$
> |1-\epsilon|<1,
> \qquad\text{or equivalently}\qquad
> 0<\epsilon<2.
> $$
>
> At stationarity, the variance $r_\infty$ satisfies
>
> $$
> r_\infty=(1-\epsilon)^2r_\infty+2\epsilon,
> $$
>
> hence
>
> $$
> r_\infty
> =\frac{2\epsilon}{1-(1-\epsilon)^2}
> =\frac{1}{1-\epsilon/2}.
> $$
>
> This tends to the target variance $1$ as $\epsilon\to0$, but is larger than $1$ for every fixed $\epsilon\in(0,2)$. In $\mathbb R^d$, each coordinate obeys the same recursion, so the stationary covariance is $(1-\epsilon/2)^{-1}I_d$.
-->

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
>     \frac{\mathrm d}{\mathrm d t}\mathbb E[h(X_t)]
>     =
>     \int h(x)\,\partial_t p_t(x)\,\mathrm d x.
>     $$
>
> 2.  Use the chain rule and conditional expectation to show that
>
>     $$
>     \frac{\mathrm d}{\mathrm d t}\mathbb E[h(X_t)]
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

<!--
> [!solution]- Solution
>
> Writing the expectation as an integral and differentiating in time,
>
> $$
> \frac{\mathrm d}{\mathrm d t}\mathbb E[h(X_t)]
> =\frac{\mathrm d}{\mathrm d t}\int h(x)p_t(x)\,\mathrm d x
> =\int h(x)\,\partial_t p_t(x)\,\mathrm d x.
> $$
>
> The chain rule and conditional expectation give a second expression:
>
> $$
> \begin{aligned}
> \frac{\mathrm d}{\mathrm d t}\mathbb E[h(X_t)]
> &=\mathbb E[\nabla h(X_t)^\top\dot X_t]\\
> &=\mathbb E[\nabla h(X_t)^\top v_t^*(X_t)]\\
> &=\int \nabla h(x)^\top v_t^*(x)p_t(x)\,\mathrm d x.
> \end{aligned}
> $$
>
> Because $h$ is compactly supported, integration by parts has no boundary term:
>
> $$
> \int \nabla h(x)^\top v_t^*(x)p_t(x)\,\mathrm d x
> =-
> \int h(x)\nabla\cdot\big(v_t^*(x)p_t(x)\big)\,\mathrm d x.
> $$
>
> Comparing the two expressions for every smooth compactly supported $h$ yields
>
> $$
> \partial_t p_t(x)
> =-\nabla\cdot\big(v_t^*(x)p_t(x)\big).
> $$
-->

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
>
> 2.  Specialize to $d=1$ and $v^\theta(z,t)=a(t)z$. Write $\log p_t^\theta(x)$ in terms of $\log p_0(z_0)$ and $\int_0^t a(\tau)\,\mathrm d \tau$.
> 3.  Explain why likelihood training for a Neural ODE is usually more expensive than rectified-flow regression training.

<!--
> [!solution]- Solution
>
> Let $\phi_t$ be the ODE flow map and let
>
> $$
> J_t=\frac{\partial z_t}{\partial z_0}
> $$
>
> be its Jacobian along a path. Change of variables gives
>
> $$
> p_t^\theta(z_t)|\det J_t|=p_0(z_0).
> $$
>
> Differentiating the ODE with respect to $z_0$ gives
>
> $$
> \dot J_t=\nabla_x v^\theta(z_t,t)J_t.
> $$
>
> Jacobi's formula then gives
>
> $$
> \frac{\mathrm d}{\mathrm d t}\log|\det J_t|
> =\operatorname{tr}\!\left(J_t^{-1}\dot J_t\right)
> =\operatorname{tr}\!\left(\nabla_x v^\theta(z_t,t)\right).
> $$
>
> Integrating along the path ending at $z_t=x$ gives
>
> $$
> \log p_t^\theta(x)
> =\log p_0(z_0)
> -\int_0^t
> \operatorname{tr}\!\left(\nabla_x v^\theta(z_\tau,\tau)\right)
> \,\mathrm d \tau.
> $$
>
> In one dimension, let $A(t)=\int_0^t a(\tau)\,\mathrm d \tau$. The ODE $\dot z_t=a(t)z_t$ has solution
>
> $$
> z_t=e^{A(t)}z_0,
> \qquad
> z_0=e^{-A(t)}x.
> $$
>
> Therefore
>
> $$
> \log p_t^\theta(x)
> =\log p_0\!\left(e^{-A(t)}x\right)-A(t).
> $$
>
> Likelihood evaluation requires solving an ODE and accumulating a divergence term for each sample. Rectified-flow training instead uses a direct regression loss at sampled states and times, without integrating a full path during each update.
-->
