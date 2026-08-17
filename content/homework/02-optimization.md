---
title: "Homework 2: Optimization and Neural Networks"
description: "Gradient descent regimes, adaptive optimizers, nonconvex landscapes, and MLP training."
publish: true
---

[Back to Module 2 notes](/modules/02-deep-learning-basics)

## Problem 1

> [!problem|Gradient Descent on Toy Quadratic Function]
> Consider applying gradient descent with constant step size on the following two-variable quadratic loss function:
>
> $$L(x,y)=\frac{1}{2}\big(x^2+\gamma y^2\big), \qquad \gamma>0.$$
>
> The gradient descent update is
>
> $$
> \begin{aligned}
> x_{t+1} &= x_t-\epsilon\,\nabla_x L(x_t,y_t),\\
> y_{t+1} &= y_t-\epsilon\,\nabla_y L(x_t,y_t),
> \end{aligned}
> $$
>
> where $\epsilon>0$ is the step size, and $\nabla_x L(x,y)$ and $\nabla_y L(x,y)$ are the gradients of $L(x,y)$ with respect to $x$ and $y$. Here $(x,y)$ are optimization variables, not data points.
>
> By varying $\epsilon$, the algorithm exhibits very different behaviors. For simplicity, assume $\gamma>1$ throughout this problem; the case $0<\gamma<1$ is similar in spirit. Answer the following questions with a mix of mathematical derivation and coding.
>
> | $\epsilon<\epsilon_1^*$                                                                                    | $\epsilon=\epsilon_1^*$                                                                                                      | $\epsilon\in(\epsilon_1^*,\epsilon_2^*)$                                                                        | $\epsilon=\epsilon_2^*$                                                                                             | $\epsilon>\epsilon_2^*$                                                                                             |
> | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
> | <img src="/assets/homework/02-optimization/gd-monotone-path.png" alt="Monotone GD trajectory" width="130"> | <img src="/assets/homework/02-optimization/gd-critical-eps1-path.png" alt="Critical first step size trajectory" width="130"> | <img src="/assets/homework/02-optimization/gd-damped-path.png" alt="Damped oscillation trajectory" width="130"> | <img src="/assets/homework/02-optimization/gd-undamped-path.png" alt="Undamped oscillation trajectory" width="130"> | <img src="/assets/homework/02-optimization/gd-diverge-path.png" alt="Divergent oscillation trajectory" width="130"> |
> | <img src="/assets/homework/02-optimization/gd-monotone-loss.png" alt="Monotone GD loss" width="130">       | <img src="/assets/homework/02-optimization/gd-critical-eps1-loss.png" alt="Critical first step size loss" width="130">       | <img src="/assets/homework/02-optimization/gd-damped-loss.png" alt="Damped oscillation loss" width="130">       | <img src="/assets/homework/02-optimization/gd-undamped-loss.png" alt="Undamped oscillation loss" width="130">       | <img src="/assets/homework/02-optimization/gd-diverge-loss.png" alt="Divergent oscillation loss" width="130">       |
>
> 1.  Let $(x_t,y_t)$ be the parameters after $t$ iterations of gradient descent with constant step size $\epsilon$ and initialization $(x_0,y_0)$. Since $L(x,y)$ is quadratic, write down a closed-form formula for $(x_t,y_t)$ in terms of $(x_0,y_0)$, $\epsilon$, and $\gamma$. Derive this formula explicitly.
> 2.  There exists a critical step size $\epsilon_1^*$ with the following properties:
>     1.  If $\epsilon\leq\epsilon_1^*$, the loss decreases monotonically to zero.
>     2.  At $\epsilon=\epsilon_1^*$, the $y$-coordinate converges to zero in one iteration.
>
>     Derive $\epsilon_1^*$ and show your reasoning. Also explain why the trajectories in the monotone cases exhibit an L-shape rather than moving straight to the origin.
> 3.  There exists another critical step size $\epsilon_2^*$ beyond which gradient descent fails to converge. Derive $\epsilon_2^*$ and explain the cases below:
>     1.  If $\epsilon\in(\epsilon_1^*,\epsilon_2^*)$, the trajectory converges with damped oscillations.
>     2.  If $\epsilon=\epsilon_2^*$, the trajectory oscillates indefinitely with constant amplitude.
>     3.  If $\epsilon>\epsilon_2^*$, the trajectory diverges with growing oscillations.
> 4.  Implement gradient descent for this example and reproduce contour plots similar to the phase reference above. Use $\gamma=10$ and $(x_0,y_0)=(-5,5)$. The plots do not need to be exactly the same, but they should qualitatively match the different phases. You may use this [Colab contour plotting example](https://colab.research.google.com/drive/1zblv6U8SH-4YFPxfFrHpkkmgVlw5n8t_?usp=sharing).
> 5.  Write down the Adam optimizer update rule for this toy problem.
> 6.  Implement Adam on the same quadratic function with $\gamma=10$ and $(x_0,y_0)=(-5,5)$. Vary Adam's learning rate (denote it by $\alpha$ to distinguish it from the gradient descent step size $\epsilon$) from very small to very large, and observe how the behavior changes. Comment on what you observe and how it differs from, or resembles, gradient descent. You may choose other Adam hyperparameters, such as $\beta_1$ and $\beta_2$, freely.
> 7.  **Optional.** Assume we will terminate the algorithm at iteration $t=t_*$. Decide the best choice of step size $\epsilon$ so that the loss function is as small as possible at iteration $t_*$. Precisely, choose an optimal step size $\epsilon_{t_*}^*$ such that $L(x_{t_*},y_{t_*})$ is minimized.
>
>     Note that:
>     1.  The optimal choice of $\epsilon_{t_*}^*$ should depend on $(x_0,y_0)$, $\gamma$, and the terminal iteration $t_*$.
>     2.  The optimal $\epsilon_{t_*}^*$ never lies in the monotonic decreasing region $(0,\epsilon_1^*)$ and may exhibit damped or even undamped oscillation. By sacrificing monotonic decrease, we can take larger steps and make faster progress within $t_*$ iterations.
>
>     In addition, derive the limit of $\epsilon_{t_*}^*$ as $t_*\to+\infty$.
> 8.  **Optional.** The studies above suggest that the efficiency of gradient descent decreases when $\gamma$ is very large. Consider the following approaches to speed up gradient descent:
>     1.  **Per-coordinate step size.** Use different step sizes for different coordinates:
>
>         $$
>         \begin{aligned}
>         x_{t+1} &\gets x_t-\epsilon_x\nabla_x L(x_t,y_t),\\
>         y_{t+1} &\gets y_t-\epsilon_y\nabla_y L(x_t,y_t),
>         \end{aligned}
>         $$
>
>         where $\epsilon_x$ is the step size for $x$ and $\epsilon_y$ is the step size for $y$. Describe a choice of $\epsilon_x$ and $\epsilon_y$ such that the algorithm reaches the minimizer $(0,0)$ within one iteration.
>
>     2.  **Per-iteration step size.** Vary the step size across iterations:
>
>         $$
>         \begin{aligned}
>         x_{t+1} &\gets x_t-\epsilon_t\nabla_x L(x_t,y_t),\\
>         y_{t+1} &\gets y_t-\epsilon_t\nabla_y L(x_t,y_t),
>         \end{aligned}
>         $$
>
>         where the step size $\epsilon_t$ depends on iteration $t$. Show that, with a well-chosen step size scheme $\{\epsilon_t\}$, gradient descent reaches the minimizer $(0,0)$ within at most two steps. Show your derivation.

> [!solution]- Solution
>
> **1. Closed-form GD iterates.**
>
> We first compute the gradient:
>
> $$
> \nabla L(x,y)=(x,\gamma y).
> $$
>
> Therefore the gradient descent update separates into the two coordinates:
>
> $$
> x_{t+1}=(1-\epsilon)x_t,\qquad
> y_{t+1}=(1-\epsilon\gamma)y_t.
> $$
>
> Iterating the recursion gives
>
> $$
> x_t=(1-\epsilon)^t x_0,\qquad
> y_t=(1-\epsilon\gamma)^t y_0.
> $$
>
> **2. First critical step size.**
>
> For the first threshold, we want $y_1=0$ after one step:
>
> $$
> y_1=(1-\epsilon\gamma)y_0=0,
> $$
>
> so
>
> $$
> \epsilon_1^*=\frac1\gamma.
> $$
>
> If $0<\epsilon\le 1/\gamma$ and $\gamma>1$, then
>
> $$
> 0\le 1-\epsilon\gamma\le 1,\qquad 0\le 1-\epsilon\le 1.
> $$
>
> Hence both coordinates move toward zero without changing sign, and the loss decreases monotonically. At $\epsilon=1/\gamma$, the $y$ coordinate becomes zero in one step. Since the $y$ direction contracts much faster than the $x$ direction, the path first moves quickly toward the $x$-axis and then moves slowly along it. This gives the L-shaped trajectory.
>
> **3. Convergence threshold.**
>
> The convergence threshold is determined by requiring both coordinate multipliers to have magnitude less than one:
>
> $$
> |1-\epsilon|<1,\qquad |1-\epsilon\gamma|<1.
> $$
>
> Since $\gamma>1$, the stricter condition is
>
> $$
> 0<\epsilon<\frac{2}{\gamma}.
> $$
>
> Thus
>
> $$
> \epsilon_2^*=\frac{2}{\gamma}.
> $$
>
> For $\epsilon\in(1/\gamma,2/\gamma)$, the $y$ multiplier is negative but has magnitude below one. The sign of $y_t$ alternates, while the amplitude decays, giving damped oscillations. At $\epsilon=2/\gamma$, the $y$ multiplier is $-1$, so $y_t=(-1)^t y_0$ oscillates without decay. For $\epsilon>2/\gamma$, $|1-\epsilon\gamma|>1$, so the $y$ coordinate grows in magnitude and the method diverges.
>
> **4. Plotting check.**
>
> For the plotting part, use the update
>
> ```python
> def gd_step(x, y, eps, gamma):
>     return (1 - eps) * x, (1 - eps * gamma) * y
> ```
>
> With $\gamma=10$, the reference thresholds are
>
> $$\epsilon_1^*=0.1,\qquad \epsilon_2^*=0.2.$$
>
> For the five regimes in the figure, one can use $\epsilon\in\{0.05,0.1,0.15,0.2,0.22\}$.
>
> Solution notebook for the plotting part: [Problem 1 GD plotting solution Colab](https://colab.research.google.com/drive/1lp0VqSz1J0nzsZurPosRiTgRCNjL-tNK?usp=sharing).
>
> **5. Adam update.**
>
> For Adam, let $g_t=(x_t,\gamma y_t)$. A standard Adam update is
>
> $$
> \begin{aligned}
> m_t &= \beta_1 m_{t-1}+(1-\beta_1)g_t,\\
> v_t &= \beta_2 v_{t-1}+(1-\beta_2)g_t\odot g_t,\\
> \hat m_t &= \frac{m_t}{1-\beta_1^t},\qquad
> \hat v_t = \frac{v_t}{1-\beta_2^t},\\
> (x_{t+1},y_{t+1})&=(x_t,y_t)-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\delta},
> \end{aligned}
> $$
>
> where division and square root are coordinatewise, and $\delta$ is a small numerical constant such as $10^{-8}$.
>
> **6. Adam experiment.**
>
> The normalization by $\sqrt{\hat v_t}$ rescales the two coordinates, so Adam is less directly controlled by the curvature ratio $\gamma$ than plain GD. If $\alpha$ is too large, the iterates can still oscillate or diverge.
>
> **7. Optional fixed-horizon step size.**
>
> For the optional fixed-horizon question, the exact objective after $t_*$ steps is
>
> $$
> L(x_{t_*},y_{t_*})
> =\frac12 x_0^2(1-\epsilon)^{2t_*}
> +\frac12\gamma y_0^2(1-\gamma\epsilon)^{2t_*}.
> $$
>
> A stationary point in the oscillatory interval $(1/\gamma,2/\gamma)$ satisfies
>
> $$
> x_0^2(1-\epsilon)^{2t_*-1}
> =
> \gamma^2y_0^2(\gamma\epsilon-1)^{2t_*-1}.
> $$
>
> Equivalently, with
>
> $$r_{t_*}=\left(\frac{\gamma^2y_0^2}{x_0^2}\right)^{1/(2t_*-1)},$$
>
> one obtains
>
> $$\epsilon_{t_*}^*=\frac{1+r_{t_*}}{1+\gamma r_{t_*}}$$
>
> when this point is the minimizer over the allowed range. As $t_*\to\infty$, $r_{t_*}\to1$, so
>
> $$
> \lim_{t_*\to\infty}\epsilon_{t_*}^*=\frac{2}{1+\gamma}.
> $$
>
> This is the step size that balances the two asymptotic contraction factors:
>
> $$|1-\epsilon|=|1-\gamma\epsilon|.$$
>
> **8. Optional acceleration.**
>
> For the optional acceleration question, per-coordinate steps can solve the problem in one iteration:
>
> $$
> \epsilon_x=1,\qquad \epsilon_y=\frac1\gamma.
> $$
>
> Then $x_1=0$ and $y_1=0$. With a scalar step size that changes by iteration, two steps are enough: choose $\epsilon_0=1/\gamma$ to set $y_1=0$, then choose $\epsilon_1=1$ to set $x_2=0$ while keeping $y_2=0$.

## Problem 2

> [!problem|Optimizers on a Nonconvex 2D Landscape]
> Implement and compare several optimization algorithms on a simple two-dimensional nonconvex test function formed by a quadratic bowl plus two Gaussian wells:
>
> $$
> f(x,y)=
> -2\exp\left(-\frac{(x-1)^2+y^2}{0.2}\right)
> -3\exp\left(-\frac{(x+1)^2+y^2}{0.2}\right)
> +x^2+y^2.
> $$
>
> Complete the provided [Colab](https://colab.research.google.com/drive/132_vJGNN5FdHeY0faanYUMWBy7DmuZP7?usp=sharing) by adding implementations of the following optimizers in code: GD, Momentum, SignGD, SoftSignGD, RMSProp, and Adam.
>
> 1.  **Implementation.** Finish the optimizer implementations in the Colab.
> 2.  **Trajectory visualization.** For each optimizer, draw a contour map of $f$ over $[-2,2]^2$ and overlay the optimization trajectory for $T$ steps. Use at least three different initializations, for example $(x_0,y_0)\in\{(-1.2,1.2),(1.2,1.2),(0.3,1.5)\}$, and budgets $T\in\{10,50,200\}$. Mark the start with $\circ$ and the end with $\times$. Briefly describe qualitative patterns you observe.
> 3.  **Hyperparameter exploration and comparison.** For each optimizer, explore the effect of hyperparameters such as learning rate $\eta$, momentum or decay parameters, and the number of steps $T$. Try several starting points and different budgets, such as $T=50$ and $T=200$. Record:
>     1.  the final objective $f(x_T,y_T)$ under different hyperparameter choices,
>     2.  whether the run converged to the left or right well,
>     3.  your best settings for each optimizer and the corresponding final performance.
>
>     Then compare the optimizers: under a fixed step budget, which optimizer performs best on average across the specified initializations? For this comparison, use one chosen hyperparameter setting per optimizer, report how you selected it, and average the final objective over initializations. Name the winners and intuitively explain why they did well in this setting.

> [!solution]- Solution
>
> Solution notebook: [Problem 2 optimizer solution Colab](https://colab.research.google.com/drive/1cU08CK3iKwtBGpuim_mRAW3I4HrO4wkC?usp=sharing).
>
> **1. Optimizer updates.**
>
> Let $\theta_t=(x_t,y_t)$ and $g_t=\nabla f(\theta_t)$. The optimizer updates are:
>
> $$
> \begin{aligned}
> \text{GD:}\qquad
> &\theta_{t+1}=\theta_t-\eta g_t.\\
> \text{Momentum:}\qquad
> &m_t=\beta m_{t-1}+(1-\beta)g_t,\quad
> \theta_{t+1}=\theta_t-\eta m_t.\\
> \text{SignGD:}\qquad
> &\theta_{t+1}=\theta_t-\eta\,\operatorname{sign}(g_t).\\
> \text{SoftSignGD:}\qquad
> &\theta_{t+1}=\theta_t-\eta\,\frac{g_t}{|g_t|+\tau}.\\
> \text{RMSProp:}\qquad
> &s_t=\beta s_{t-1}+(1-\beta)(g_t\odot g_t),\quad
> \theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\delta}.\\
> \text{Adam:}\qquad
> &m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
> v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t),\\
> &\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
> \hat v_t=\frac{v_t}{1-\beta_2^t},\quad
> \theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\delta}.
> \end{aligned}
> $$
>
> All vector operations above are coordinatewise. Use defaults such as $\beta=0.9$, $(\beta_1,\beta_2)=(0.9,0.999)$, and $\delta=10^{-8}$.
>
> **2. Interpreting the trajectories.**
>
> The left well near $(-1,0)$ is deeper than the right well near $(1,0)$, because its Gaussian coefficient is $-3$ instead of $-2$. Runs initialized near the right side can still end in the right well, especially with a small step budget or a conservative learning rate. Larger learning rates and momentum may cross between basins, but they may also overshoot the narrow wells.
>
> **3. Comparison.**
>
> With the same initializations and the same step budget for every optimizer, the comparison entries are the average final objective and the number of runs ending in each well:
>
> | optimizer | hyperparameters | average final $f$ | left/right well counts |
> | --- | --- | --- | --- |
> | GD | $\eta=\cdots$ | $\cdots$ | $\cdots$ |
> | Momentum | $\eta=\cdots,\beta=\cdots$ | $\cdots$ | $\cdots$ |
> | SignGD | $\eta=\cdots$ | $\cdots$ | $\cdots$ |
> | SoftSignGD | $\eta=\cdots,\tau=\cdots$ | $\cdots$ | $\cdots$ |
> | RMSProp | $\eta=\cdots,\beta=\cdots$ | $\cdots$ | $\cdots$ |
> | Adam | $\eta=\cdots,\beta_1=\cdots,\beta_2=\cdots$ | $\cdots$ | $\cdots$ |
>
> Adaptive methods rescale coordinates when gradient magnitudes change. Momentum can move faster through shallow regions, but it can also pass through a narrow minimum.

## Problem 3

> [!problem|Implementing Gradient Descent on MLP]
> Implement a simple multi-layer neural network, also called a multi-layer perceptron (MLP), with gradient descent. Starter code is provided in this [Colab notebook](https://colab.research.google.com/drive/1FVu27mlStzCJQnymkMjfOo-FceU_pn_M?usp=sharing). You are also welcome to use other frameworks if you prefer.
>
> 1.  The provided code only works for two-layer networks. Extend the function `mlp` so that it can handle an arbitrary number of layers.
>
>     Using your extended code, try networks of different sizes, both in depth and width. Report what architectures you tested and describe how network size affected training loss and test performance.
>
> 2.  In the starter code, the weights are initialized by sampling from $\mathcal N(0,\sigma^2)$ with scale parameter $\sigma$. Try different values of $\sigma$, for example $10^{-5}$, $10^{-1}$, and $10$. Comment on what you observe. What happens if $\sigma$ is too small or too large? What initialization strategy can help alleviate the gradient vanishing problem?
> 3.  Replace the `ReLU` activation with `Sigmoid` and repeat the training. You will likely find it harder to train the network with `Sigmoid`. Give an intuitive explanation of why this happens. Then try to improve the result by tuning hyperparameters such as initialization scale, learning rate, and number of iterations. Report your best results and any insights.
> 4.  The mean square error is not always the best loss function. In the presence of outliers, the **Huber loss** is more robust, since it behaves like squared loss for small errors but like absolute loss for large errors. See the [Wikipedia page on Huber loss](https://en.wikipedia.org/wiki/Huber_loss).
>     1.  Understand the logic of Huber loss and explain why it is less sensitive to outliers than the mean square loss. You may consult references.
>     2.  Modify the training data by introducing a few outliers, for example by setting the labels of the first five training samples to $10$. Keep the test data unchanged. Compare the performance of MSE and Huber loss on this dataset, and report your findings.

> [!solution]- Solution
>
> Solution notebook for the coding parts: [Full solution Colab](https://colab.research.google.com/drive/1D1dhC_Qk5GU75QAts_1iw12l3JfVrkwD?usp=sharing).
>
> **1. General MLP implementation.**
>
> Implement the MLP as a list of layers rather than a hard-coded two-layer expression. For example, if `params` stores pairs `(W_l, b_l)`, the forward pass can be:
>
> ```python
> def mlp(params, x, activation=relu):
>     h = x
>     for W, b in params[:-1]:
>         h = activation(h @ W + b)
>     W_out, b_out = params[-1]
>     return h @ W_out + b_out
> ```
>
> **2. Network size.**
>
> For network size:
>
> - Very small networks can underfit: both training and test errors remain high.
> - Wider or deeper networks reduce training error in most runs.
> - Test error may improve at first, then stop improving or become worse if the model overfits.
> - Deeper networks are more sensitive to initialization, learning rate, and activation choice.
>
> **3. Initialization scale.**
>
> For initialization, if $\sigma$ is too small, the network starts close to a nearly constant function. Activations and gradients can be tiny, so learning is slow. If $\sigma$ is too large, pre-activations can become very large; this can produce unstable outputs for ReLU networks and saturation for Sigmoid networks. Variance-scaled choices are Xavier/Glorot initialization for Sigmoid/Tanh-style activations and He/Kaiming initialization for ReLU-style activations.
>
> **4. Sigmoid and Huber loss.**
>
> Sigmoid is harder to train because
>
> $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$
>
> is small when $z$ is very positive or very negative. Once many units saturate near $0$ or $1$, gradients shrink as they backpropagate through layers. To improve Sigmoid training, use Xavier initialization, normalized inputs, a smaller learning rate, and more iterations. ReLU trains faster on this task because active ReLU units keep derivative $1$.
>
> The Huber loss with threshold $\delta>0$ is
>
> $$
> \ell_\delta(r)=
> \begin{cases}
> \frac12 r^2, & |r|\le \delta,\\
> \delta(|r|-\frac12\delta), & |r|>\delta,
> \end{cases}
> $$
>
> where $r=\hat y-y$ is the residual. For small errors, it behaves like MSE, so it still encourages accurate fitting. For large errors, it grows only linearly, so a few outliers do not dominate the total loss. In the outlier experiment, MSE assigns quadratic penalty to the corrupted labels and pulls the fitted function toward them. Huber loss assigns only linear penalty to large residuals, so the fitted function changes less on the clean region.
>
> Record the architectures, initialization scales, learning rates, number of iterations, and train/test metrics used in the experiments. Loss curves or fitted-function plots show underfitting, instability, and outlier sensitivity.
