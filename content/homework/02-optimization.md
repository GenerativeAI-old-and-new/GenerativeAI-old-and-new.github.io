---
title: "Homework 2: Optimization and Neural Networks"
description: "Homework problems for Module 2 covering gradient descent, optimizer behavior, nonconvex landscapes, and MLP training."
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
>
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
>
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
