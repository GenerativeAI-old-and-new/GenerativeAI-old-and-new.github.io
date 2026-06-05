---
title: "Module 2: Deep Learning Basics"
description: "Optimization and neural network basics for generative modeling."
publish: true
---

<!-- prettier-ignore-start -->

## Optimization Basis

Training AI and machine learning models reduces to solving an optimization problem:

$$\min_{\theta} L(\theta),$$ where $L(\theta)$ is the loss function over model parameters $\theta$. It typically takes the form

$$
\begin{aligned}
    L(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(\theta; x_i^{\mathrm{data}}),
\end{aligned}
$$

where $\ell(\theta; x_i^{\mathrm{data}})$ is the loss contributed by a single data point $x_i^{\mathrm{data}}$.

The main challenge arises from the scale: modern deep learning involves billions of parameters and data points, making this a very expensive large scale optimization problem. The choice of optimizer directly affects the model's final accuracy, training speed, and cost.

##### Gradient Descent

Gradient descent updates the parameters by moving in the direction opposite to the gradient of the loss: $$\theta_{t+1} = \theta_t - \epsilon \cdot \nabla_\theta L(\theta_t),$$ where we start from an initialization $\theta_0$ and iteratively update $\theta_t$ by following the negative gradient direction.

The learning rate $\epsilon$ controls the step size. If it is too small, convergence is slow; if it is too large, the updates may overshoot or diverge.

<figure
  class="interactive-figure figure-wide"
  data-interactive-figure="optimizer-trajectory"
  data-case="gd-vs-sgd"
></figure>

##### Mini-batch Gradient Descent

For large datasets, the loss function in Eq. `equ:lossform` requires summing over many data points. Correspondingly, in gradient descent, computing the gradient involves evaluating the full sum:

$$
\theta_{t+1} = \theta_t - \epsilon \cdot
\frac{1}{n}\sum_{i=1}^n \nabla_{\theta} \ell(\theta, x_i^{\mathrm{data}}).
$$

Mini-batch gradient descent approximates this full-batch gradient by averaging over a smaller subset of data:

$$
\theta_{t+1} = \theta_t - \epsilon \cdot
\frac{1}{ \left|\mathcal B_t\right| }\sum_{i\in \mathcal B_t}\nabla \ell(\theta_t; x^{(i)}),
$$

where $\mathcal B_t \subset \{1,2,\ldots, n\}$ is a subset of data indices, known as a mini-batch. The mini-batch $\mathcal B_t$ may be drawn randomly at each iteration or selected sequentially by sweeping through the dataset. The batch size is $m = \left|\mathcal B_t\right|$.

An extreme case is when a single data point is used: $$\theta_{t+1} = \theta_t - \epsilon \cdot \nabla \ell(\theta_t; x^{(i_t)}),$$ where $i_t$ is a random data point selected the $t$-th iteration. This is known as stochastic gradient descent (SGD).

Using mini-batches introduces variance and noise in the gradient estimates. However, it significantly reduces computational cost per update and enables more frequent updates. In practice, it is often more effective to take many noisy steps than to compute a single precise but expensive update.

##### Momentum

In gradient descent, gradients can vary significantly across iterations; that is, the gradients $g_t$ and $g_{t-1}$ may differ substantially. This variation can result from noise, poor conditioning of the loss landscape, or large step sizes, and may cause the update direction to fluctuate sharply.

<figure
  class="interactive-figure figure-wide"
  data-interactive-figure="optimizer-trajectory"
  data-case="gd-vs-momentum"
></figure>

Momentum is a technique used to smooth the gradient across iterations. Instead of directly using the current gradient to update the parameters, momentum computes an exponentially weighted moving average of past gradients:

$$
\begin{aligned}
m_t &= \beta m_{t-1} + (1 - \beta) g_t \\
\theta_{t+1} &= \theta_t - \epsilon \cdot m_t.
\end{aligned}
$$

Unrolling the recurrence shows that $m_t$ is an exponentially weighted average of past gradients:

$$
m_t = (1 - \beta) \left( g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots + \beta^{t-1} g_1 \right) + \beta^t m_0.
$$

Typically, $\beta$ is set close to $1$, such as $\beta = 0.9$. This means the current gradient contributes $10\%$ to the update, with the rest influenced by past gradients.

The main benefit of momentum is that it smooths out noise, encourages consistent update directions, and reduces oscillations. This often leads to faster and more stable convergence.

In fact, in the limit of a small step size, momentum corresponds to the physical dynamics of a ball moving in a potential field with friction. The momentum term captures the effect of inertia in this analogy.

> [!remark] Remark
> Recall that $$1-\beta^t = (1-\beta) (1+\beta + \cdots + \beta^{t-1}).$$ Therefore, $m_t$ is a convex combination of $\{m_0, g_1, \ldots, g_t\}$ as shown in Eq. `equ:mtgt`.

##### Nesterov Momentum

If $\beta$ is too close to one, $m_t$ can deviate significantly from $g_t$. One approach is to add the gradient back and use a linear combination of the gradient and momentum for the update:

$$
\begin{aligned}
m_t &= \beta m_{t-1} + (1 - \beta) g_t \\
\theta_{t+1} &= \theta_t - \epsilon \cdot (\eta m_t + (1 - \eta) g_t).
\end{aligned}
$$

With a proper choice of $\eta$, it is possible to smooth the momentum without de-emphasizing the contribution of the current gradient.

## Adaptive Methods

##### Coordinate Imbalance

In neural network training, different parameters can have vastly different gradient scales; that is, the magnitudes of the elements in the gradient vector $g_t$ may differ significantly. As a result, some parameters may update too aggressively while others change too slowly, leading to instability. Note that this issue is distinct from the variation between $g_t$ and $g_{t+1}$, which momentum seeks to smooth.

##### Signed Gradient

Signed methods address coordinate imbalance by applying the sign function to the gradient or momentum vector, equalizing the update magnitude across coordinates:

$$
\begin{aligned}
\text{Signed GD:} & \quad \theta_{t+1} = \theta_t - \eta \cdot \text{sign}(g_t) \\
\text{Signed Momentum:} & \quad \theta_{t+1} = \theta_t - \eta \cdot \text{sign}(m_t).
\end{aligned}
$$

This is one of the most aggressive normalization approaches, as it forces all coordinates to have equal update magnitude. Consequently, the update depends only on the sign of the gradient or momentum, not its magnitude.

<figure
  class="interactive-figure figure-wide"
  data-interactive-figure="optimizer-trajectory"
  data-case="signed-vs-softsign"
></figure>

As a trade-off, smoother normalizations can be used, such as a soft variant of the sign function:

$$
\begin{aligned}
\text{SoftSign GD:} & \quad \theta_{t+1} = \theta_t - \eta \cdot \text{softsign}(g_t) \\
\text{SoftSign Momentum:} & \quad \theta_{t+1} = \theta_t - \eta \cdot \text{softsign}(m_t),
\end{aligned}
$$

where

$$\text{softsign}(x) = \frac{x}{|x| + \epsilon},$$

with $\epsilon \geq 0$. When $\epsilon = 0$, this reduces to the standard $\text{sign}(x)$ function.

##### Adam Optimizer

The [Adam optimizer](https://arxiv.org/abs/1412.6980) (short for adaptive moment estimation) combines momentum with adaptive scaling. A simplified version of Adam's update rule is:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\[0.5em]
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\[0.5em]
\theta_{t+1} &= \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon},
\end{aligned}
$$

where $m_t$ and $v_t$ are exponential moving averages of the gradient $g_t$ and its elementwise square $g_t^2$, respectively. The update uses the ratio $\frac{m_t}{\sqrt{v_t} + \epsilon}$, applied elementwise.

Note that when $\beta_1 = \beta_2 = 0$, we have $m_t = g_t$ and $v_t = g_t^2$, and the update reduces to the softsign gradient. Hence, Adam can be viewed as Signed Gradient Descent, smoothed across iterations through the recurrence of the two moment estimates $m_t$ and $v_t$.

Adam can also be interpreted as applying an adaptive scaling factor to softsign momentum: $$\theta_{t+1} = \theta_t - \eta_t \cdot \mathrm{softsign}(m_t),
\qquad
\eta_t = \eta \cdot \frac{|m_t| + \epsilon}{\sqrt{v_t} + \epsilon}.$$

The adaptive learning rate $\eta_t$ can be interpreted as a measure of gradient agreement across iterations. If the gradients $g_1, \dots, g_t$ are highly inconsistent (e.g., frequently changing sign) along a given coordinate, the magnitude of $m_t$ tends to be small due to cancellation, while $v_t$ remains relatively large. This results in a smaller $\eta_t$, which slows down the update along that coordinate. Therefore, Adam automatically adjusts the step size based on gradient coherence, a key distinction from softsign momentum.

<figure
  class="interactive-figure figure-wide"
  data-interactive-figure="optimizer-trajectory"
  data-case="adam-comparison"
></figure>

##### Bias Correction of Adam

Standard Adam implementation is slightly more complicate than what we wrote above:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\[0.5em]
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\[0.5em]
\hat{m}_{t}
&= \frac{m_t}{1-\beta_1^t} \\[0.5em]
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\[0.5em]
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\end{aligned}
$$

Here, we introduce $\hat m_t$ and $\hat v_t$ as scaled by a factor of $m_t$ and $v_t$.

To see why we may (or may not) need to introduce bias correction, consider the case where $m_0 = 0$. Then the update of $m_t$ becomes:

$$m_t = (1 - \beta_1) \left( g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \cdots + \beta_1^{t-1} g_1 \right).$$

Using the identity $1 - \beta_1^t = (1 - \beta_1)(1 + \beta_1 + \cdots + \beta_1^{t-1})$, the bias-corrected term $\hat{m}_t$ is given by:

$$
\begin{aligned}
\hat{m}_t &= \frac{(1 - \beta_1) \left( g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \cdots + \beta_1^{t-1} g_1\right)}{1 - \beta_1^t} \\
&= \frac{g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \cdots + \beta_1^{t-1} g_1}{1 + \beta_1 + \beta_1^2 + \cdots + \beta_1^{t-1}}.
\end{aligned}
$$

This shows that $\hat{m}_t$ is a weighted average of $\{g_1, \ldots, g_t\}$. In particular, if all gradients are equal, then $\hat{m}_t = g_t$.

However, if we initialize $m_0 = g_0$ (using an initial gradient estimate), then $m_t$ without correction becomes a convex combination of $\{m_0, g_1, \ldots, g_{t-1}\}$, and bias correction should not be applied.

> [!remark] Remark
> Hence, the need for this correction term is directly tied to initializing the momentum with $m_0 = 0$, $v_0 = 0$.
>
> In practice, the bias correction becomes less important as $t$ increases, since $1 - \beta_i^t \to 1$ as $t \to \infty$.

## Regularization and Weight Decay

##### The Overfitting Problem

Large neural networks can memorize training data instead of learning generalizable patterns. This leads to excellent training performance but poor validation performance. Regularization is a standard technique to mitigate overfitting by discouraging overly complex models.

##### L2 Regularization and Weight Decay

L2 regularization is one of the most commonly used techniques. It is also known as ridge regression in the context of least squares. It adds a penalty term to the loss function: $$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|^2.$$

Applying gradient descent to this objective yields: $$\theta_{t+1} = \theta_t - \eta \left(\nabla L(\theta_t) + \lambda \theta_t\right).$$ The term $\lambda\theta_t$ is known as a weight decay. Rewriting this highlights the explicit weight decay effect: $$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \nabla L(\theta_t).$$ It shrinks the parameters toward zero, encouraging smoother models that tend to generalize better.

##### AdamW

For adaptive optimizers like Adam, weight decay should be applied separately from the gradient update. This is implemented in the [AdamW](https://arxiv.org/abs/1711.05101) variant: $$\begin{aligned}
\theta_{t+1} &= \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right),
\end{aligned}$$ where $\hat{m}_t$ and $\hat{v}_t$ are the bias-corrected first and second moment estimates, as defined previously.

Note, however, that this is no longer equivalent to minimizing an L2-regularized objective.

This decoupled approach ensures consistent regularization across parameters and often leads to improved generalization performance.

##### Muon Optimizer

Adam normalizes updates coordinate by coordinate. [Muon](https://arxiv.org/abs/2502.16982) takes a different view: many important neural network parameters are matrices, such as the weight matrix of a hidden linear layer. Instead of only balancing individual coordinates, Muon tries to balance the update as a matrix.

The name Muon stands for **MomentUm Orthogonalized by Newton-Schulz**. A simplified version of the update is:

$$
\begin{aligned}
M_t &= \beta M_{t-1} + (1-\beta)G_t \\
U_t &\approx \mathrm{Ortho}(M_t) \\
W_{t+1} &= W_t - \eta U_t,
\end{aligned}
$$

where $G_t$ is the gradient of a matrix parameter $W_t$. The first line is momentum, as before. The second line approximately orthogonalizes the momentum matrix. Intuitively, this flattens the singular values of the update, so the update does not over-emphasize a few dominant matrix directions.

If $M_t = P \Sigma Q^\top$ is the singular value decomposition of the momentum matrix, exact orthogonalization would replace $M_t$ by $P Q^\top$. Computing this exactly with SVD is usually too expensive inside every optimizer step. Muon instead uses a few Newton-Schulz iterations, which are based on matrix multiplications and are much more suitable for GPUs.

The practical benefit is efficiency. In recent language model training experiments, Muon often reaches the same validation loss with fewer tokens or fewer training steps than AdamW. Its per-step update can be slightly more expensive than AdamW, but the improved sample efficiency can still reduce the overall training cost. This is why Muon has become interesting for modern LLM pretraining, where optimizer efficiency directly translates into saved GPU time.

<figure
  class="interactive-figure figure-wide"
  data-interactive-figure="optimizer-trajectory"
  data-case="muon-comparison"
></figure>

The figure is a toy two-dimensional illustration. It treats the two plotted directions as a proxy for singular directions: momentum smooths the update, Adam rescales coordinates, and Muon-like orthogonalization flattens the update scale before applying the step.

In practice, Muon is mainly used for hidden two-dimensional weight matrices. Other parameters, such as embeddings, output heads, biases, gains, and scalar or vector parameters, are usually optimized with AdamW.

---

## Neural Networks

A central goal in machine learning is function approximation: to learn an unknown function $f^*$ from observed data. In many real-world tasks, the true mapping between inputs and outputs is highly complex and cannot be described explicitly by hand-crafted formulas. Instead, we rely on data-driven approaches to approximate this mapping in a flexible manner. Neural networks provide a powerful and versatile class of function approximators that are particularly well-suited for high-dimensional and structured data.

![image](/assets/modules/02-deep-learning-basics/universal_approximation.png)

We typically adopt a **parametric approximation** framework. This means that rather than searching over all possible functions $f$, which would be infeasible, we restrict ourselves to a parameterized family of functions: $$\{ f_\theta : \theta \in \Theta \}.$$ Here, the parameter vector $\theta$ encodes the adjustable components of the model, such as the weights and biases in a neural network. By varying $\theta$, the hypothesis class $\{f_\theta\}$ can represent a wide range of candidate functions.

The learning problem is then formulated as an optimization task: find the parameter configuration that minimizes the expected loss between the predictions and the ground truth: $$\min_{\theta \in \Theta} \, \mathbb{E}_{x,y}\big[L(f_\theta(x), y)\big].$$ In this expression, the loss function $L$ quantifies the discrepancy between the model output $f_\theta(x)$ and the true label $y$. Typical choices include mean squared error for regression tasks or cross-entropy for classification tasks.

This formulation highlights the interplay between three key elements: the choice of function class (the architecture of the neural network), the loss function that reflects the task objective, and the optimization procedure used to adjust $\theta$. Together, they form the foundation of modern neural network training.

<figure class="compact-concept-figure" aria-label="A target function is approximated by the best candidate inside a hypothesis class.">
  <svg viewBox="0 0 520 150" role="img">
    <defs>
      <marker id="function-approx-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path class="compact-arrowhead" d="M 0 0 L 10 5 L 0 10 z"></path>
      </marker>
    </defs>
    <rect class="compact-node compact-target" x="18" y="54" width="108" height="48" rx="6"></rect>
    <text x="72" y="73" text-anchor="middle">target</text>
    <text x="72" y="92" text-anchor="middle">f*</text>
    <rect class="compact-region" x="275" y="18" width="228" height="114" rx="8"></rect>
    <text class="compact-muted" x="389" y="38" text-anchor="middle">hypothesis class { f_theta }</text>
    <rect class="compact-node compact-best" x="306" y="58" width="84" height="38" rx="6"></rect>
    <text x="348" y="78" text-anchor="middle">f_hat</text>
    <circle class="compact-dot" cx="430" cy="62" r="5"></circle>
    <circle class="compact-dot" cx="452" cy="100" r="5"></circle>
    <circle class="compact-dot" cx="322" cy="112" r="5"></circle>
    <line class="compact-dash" x1="126" y1="78" x2="306" y2="78" marker-end="url(#function-approx-arrow)"></line>
    <text class="compact-muted" x="216" y="65" text-anchor="middle">approximation error</text>
  </svg>
</figure>

## From Linear to Nonlinear: Building Neural Networks

The simplest parametric model is the **linear function**: $$f_\theta(x) = w^\top x + b, \quad \theta = (w,b).$$ This model is attractive for its simplicity and efficiency. Linear models can be trained efficiently, often with closed-form solutions in regression settings, and they provide clear interpretability: each coefficient $w_j$ reflects the contribution of the $j$-th input dimension to the prediction.

However, despite these advantages, linear models have limited expressivity. They can only capture relationships that are linear in the input space, meaning they are fundamentally incapable of representing curved decision boundaries or complex nonlinear structures. This limitation makes them insufficient for most modern machine learning tasks, such as image recognition or natural language processing, where data exhibits intricate nonlinear dependencies.

<figure class="compact-concept-figure compact-concept-figure-narrow" aria-label="A linear model maps input x to output y through an affine function.">
  <svg viewBox="0 0 420 105" role="img">
    <defs>
      <marker id="linear-model-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path class="compact-arrowhead" d="M 0 0 L 10 5 L 0 10 z"></path>
      </marker>
    </defs>
    <rect class="compact-node compact-target" x="20" y="34" width="72" height="36" rx="6"></rect>
    <text x="56" y="54" text-anchor="middle">input x</text>
    <rect class="compact-node compact-best" x="153" y="25" width="116" height="54" rx="6"></rect>
    <text x="211" y="45" text-anchor="middle">linear map</text>
    <text x="211" y="63" text-anchor="middle">w^T x + b</text>
    <rect class="compact-node compact-target" x="328" y="34" width="72" height="36" rx="6"></rect>
    <text x="364" y="54" text-anchor="middle">output y</text>
    <line class="compact-flow" x1="92" y1="52" x2="153" y2="52" marker-end="url(#linear-model-arrow)"></line>
    <line class="compact-flow" x1="269" y1="52" x2="328" y2="52" marker-end="url(#linear-model-arrow)"></line>
  </svg>
</figure>

To overcome this limitation, we introduce **neural networks**, which generalize linear models by combining linear transformations with nonlinear activation functions. A single-layer neural network, also known as a shallow network, can be expressed as $$f_\theta(x) = \sum_{i=1}^N a_i \, \sigma(w_i^\top x + b_i),$$ where $\sigma$ denotes a nonlinear activation function such as the rectified linear unit (ReLU), the sigmoid, or the hyperbolic tangent ($\tanh$). Each term in the summation corresponds to a neuron: a linear projection of the input followed by a nonlinear transformation, scaled by an output weight $a_i$.

The introduction of nonlinearity is the key to increasing expressivity. Without $\sigma$, the composition of linear maps would remain linear, regardless of depth. With nonlinear activations, however, neural networks can approximate highly complex functions, enabling them to model intricate structures in data.

![image](/assets/modules/02-deep-learning-basics/1_ZafDv3VUm60Eh10OeJu1vw.png)

In this way, neural networks can be seen as building blocks: each layer transforms the input into a new representation, with nonlinearities ensuring that successive layers capture progressively richer patterns. This simple extension beyond linear models forms the foundation of deep learning.

<figure
  class="interactive-figure figure-wide"
  data-interactive-figure="universal-approximation"
  data-target="wave"
  data-width="8"
  data-activation="relu"
></figure>

The figure shows a one-dimensional target function $f^*$, the best linear fit, and a shallow neural-network fit $f_\theta$. The lower strip shows the hidden units: each curve is one activated feature, and the output layer combines these features with learned weights to sculpt the final approximation. Here the output weights are fitted directly to highlight representational capacity, so the figure is not simulating gradient descent.

## Universal Approximation Theorem

> [!theorem] Theorem
> A neural network with a single hidden layer can approximate any bounded continuous function $f^*$ on a bounded domain $\Omega$ to arbitrary accuracy, provided it has sufficiently many neurons: $$f_\theta(x) = \sum_{i=1}^N a_i \, \sigma(w_i^\top x + b_i).$$ More precisely, if $\sigma$ is continuous and not a polynomial, then for any $\epsilon > 0$ there exists $N>0$ and parameters $\theta = \{(a_i,w_i,b_i)\}_{i=1}^N$ such that $$\max_{x \in \Omega} |f^*(x) - f_\theta(x)| \leq \epsilon.$$

![image](/assets/modules/02-deep-learning-basics/Figure_1.png)

##### Assumptions and scope.

The statement is uniform approximation on compact sets: boundedness of $\Omega$ and continuity of $f^*$ ensure $\|f^*\|_\infty<\infty$ and density is taken in the sup norm. The condition "$\sigma$ non-polynomial" excludes degenerate cases; popular choices such as ReLU, sigmoid, and $\tanh$ satisfy it. Biases $b_i$ are essential---without them, the attainable set collapses on certain symmetric subclasses. The theorem guarantees existence of a good parameterization but is silent on how large $N$ must be, how to find $\theta$ by optimization, or how the required width scales with dimension.

##### Proof sketch (ReLU case).

1.  **Piecewise-linear reduction.** Since $f^*$ is continuous on compact $\Omega$, for any $\epsilon>0$ there exists a piecewise-linear (PL) function $p$ such that $\|f^*-p\|_\infty\le\epsilon/2$ (e.g., by triangulating $\Omega$ and linearly interpolating nodal values).

2.  **ReLU realizes PL functions.** In one dimension, a ReLU atom $x\mapsto \mathrm{ReLU}(x-t)=\max\{x-t,0\}$ creates a kink at $t$. Finite differences of shifted ReLUs synthesize hat/bump functions, e.g., $$\phi_{t,h}(x) \;=\; \mathrm{ReLU}(x-t)-2\,\mathrm{ReLU}(x-t-h)+\mathrm{ReLU}(x-t-2h),$$ which is triangular on $[t,t+2h]$ and zero outside; summing such bumps reproduces any 1D PL function exactly. In higher dimensions, PL functions are piecewise affine on a polyhedral complex; each region can be carved out by maxima of affine forms and expressed as sums of ReLU of affine functions. Therefore there exists a one-hidden-layer ReLU network $f_\theta$ with $\|p-f_\theta\|_\infty\le\epsilon/2$, yielding $\|f^*-f_\theta\|_\infty\le\epsilon$ by the triangle inequality.

##### Intuition.

Each neuron implements an affine projection followed by a nonlinearity, introducing a kink (ReLU) or a smooth bump (sigmoid/$\tanh$). By superposing many such localized shape primitives, the network sculpts increasingly fine features of $f^*$ until the uniform error falls below any prescribed tolerance.

##### Why not polynomial?

If $\sigma$ were a polynomial, then any finite linear combination $\sum_i a_i\,\sigma(w_i^\top x+b_i)$ remains a polynomial in the input coordinates (after expansion). The hypothesis class would then coincide with a finite-degree polynomial family, which is not dense in $C(\Omega)$ under the sup norm unless degree tends to infinity. The non-polynomial condition prevents this collapse and ensures density.

##### What the theorem does not say.

It does not provide approximation rates in terms of smoothness of $f^*$, nor does it claim that the required width $N$ is moderate---$N$ can scale poorly with input dimension and target complexity. It also does not address statistical generalization or optimization; a representable $f_\theta$ may be difficult to learn in practice. Finally, for discontinuous targets one typically resorts to $L^p$ approximation on $\Omega$ instead of uniform approximation.

##### Depth vs. width.

While the theorem uses a single hidden layer (width-driven expressivity), increased depth can represent certain function families with exponentially fewer neurons. Thus, depth trades width for hierarchical composition, often yielding superior parameter efficiency in practice, even though universality already holds at depth $2$.

## Deep Neural Networks: Composition of Functions

Instead of wide single-layer networks, we can build **deep architectures** by stacking multiple layers together: $$f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x).$$ Here, each layer $f_\ell$ is itself a nonlinear transformation, typically a linear projection followed by an activation function. By composing such transformations, deep networks successively refine the representation of the input: the early layers extract simple patterns, while later layers encode higher-level features.

![image](/assets/modules/02-deep-learning-basics/Two-or-more-hidden-layers-comprise-a-Deep-Neural-Network.png)

The key distinction between shallow and deep models lies not in their theoretical ability to approximate functions (both can be universal approximators), but in the efficiency and structure of that approximation. Depth introduces a hierarchy that allows complex functions to be expressed more compactly.

### Pros of Depth

- **Hierarchical feature learning.** Deep networks naturally learn representations at multiple levels of abstraction: for example, in vision, from edges and textures to shapes and objects. This compositionality mirrors the structure of many real-world signals.

- **Efficient parameter usage.** Certain functions that would require exponentially many neurons in a shallow network can be represented with polynomially many parameters in a deep network. Depth thus provides a powerful mechanism for compact representation.

- **Better generalization.** Empirically, deeper architectures often generalize well despite their size. This is partly because hierarchical representations capture underlying structures in data, which improve transferability and robustness.

### Cons of Depth

- **Vanishing and exploding gradients.** During backpropagation, repeated multiplication through many layers can cause gradients to shrink towards zero or blow up, making optimization unstable.

- **Degradation problem.** Beyond a certain depth, simply adding more layers does not always improve performance. In fact, deeper models can exhibit higher training error if not carefully designed.

- **Optimization challenges.** Deep networks are highly non-convex, and training requires sophisticated initialization, normalization, and optimization strategies to converge reliably.

Colab [Python Notebook](https://colab.research.google.com/drive/1J_UYVNjcfttadYn6Re0KhRvrNbB9sJNM?usp=sharing)

### Residual Connections (ResNet)

One of the most influential techniques to address these optimization difficulties is the introduction of **residual connections**, popularized by ResNet architectures. A residual block takes the form $$f_i(x) = x + nn_i(x),$$ where $nn_i(x)$ is a small neural sub-network (e.g., a few convolutional layers). The identity shortcut $x \mapsto x$ provides a direct path for information and gradients to flow, reducing the risk of vanishing signals. When $nn_i(x)\approx 0$, the block approximates the identity mapping, ensuring that stacking many such blocks does not harm performance.

Residual connections thus stabilize training, enable networks with hundreds of layers to be optimized effectively, and have become a cornerstone of modern deep learning architectures across vision, language, and speech.

## Attention Mechanism

Modern architectures introduce the **attention mechanism**. Given a query vector $x_{\text{query}}$ and reference vectors $\{x_i\}_{i=1}^n$, the attention output aggregates the values $V(x_i)$ with data-dependent weights: $$\text{Attention}(x_{\text{query}}, \{x_i\})
    = \sum_{i=1}^n \text{sim}(x_{\text{query}}, x_i)\, V(x_i),$$ where the weights are obtained by a softmax over similarity scores $$\text{sim}(x_{\text{query}}, x_i)
    = \frac{\exp(Q(x_{\text{query}})^\top K(x_i))}
        {\sum_{j=1}^n \exp(Q(x_{\text{query}})^\top K(x_j))}.$$ Here $Q(x),K(x),V(x)$ are linear maps producing query, key, and value representations. The softmax normalization makes the weights nonnegative and row-stochastic (sum to $1$), so the output is a convex combination of the values. Intuitively, $Q$ tests which references are most relevant to the query via inner products with the keys $K(x_i)$, and $V$ determines the content to be aggregated. This decoupling between matching ($Q,K$) and content ($V$) is the core advantage over fixed, position-only averaging.

From a matrix viewpoint, if we stack queries into $Q\in\mathbb{R}^{m\times d}$ and keys/values into $K,V\in\mathbb{R}^{n\times d}$, attention forms an $m\times n$ weight matrix by row-wise softmax of $QK^\top$, then multiplies by $V$. The operation is differentiable end-to-end and adapts the receptive field based on content rather than fixed geometry.

## Self-Attention and Multi-Head Attention

### Multi-Head Attention

Attention can be extended with multiple heads to allow the model to focus on complementary relational patterns: $$\text{MultiHead}(x_{\text{query}}, \{x_i\})
    = \sum_{h=1}^H W_h \left( \sum_{i=1}^n \text{sim}_h(x_{\text{query}}, x_i)\, V_h(x_i) \right),$$ where each head has its own $(Q_h,K_h,V_h)$ and an output projection $W_h$. Using multiple heads encourages the network to learn distinct similarity notions (e.g., syntactic vs. semantic relations in language) by projecting inputs into different subspaces before computing attention. The outer projections $\{W_h\}$ then mix these head-specific summaries into a single representation.

### Self-Attention

In self-attention, every element serves simultaneously as a query and as a reference: $$\text{SelfAttention}(\{x_i\}) = \big[\text{MultiHead}(x_i,\{x_j\}_{j=1}^n)\big]_{i=1}^n .$$ Each position $i$ aggregates information from all positions $j$, enabling direct long-range interactions without stacking many local operations.

##### Properties.

- **Permutation structure.** As a set operation, self-attention is permutation equivariant to the order of inputs; it becomes order-aware once position information is injected (see below). Global pooling of self-attention outputs yields permutation invariance.

- **Long-range dependencies.** Every token can directly attend to any other token in one layer, providing a content-adaptive receptive field.

- **Rich information exchange.** Attention weights form data-driven routing of information across elements, allowing the model to emphasize salient interactions while suppressing irrelevant ones.

## Language Models

Language models apply self-attention to sequences of tokens. Consider the sentence $$\text{``The cat sat on the mat.''}$$ We index tokens as $(1,\text{``The''}), (2,\text{``cat''}), (3,\text{``sat''}), \ldots$ and map both words and positions into vectors in $\mathbb{R}^d$. Inputs are typically formed by a sum of embeddings: $$x_i \;=\; \text{embedding}(\text{position}_i) \;+\; \text{embedding}(\text{word}_i).$$ The resulting sequence $\{x_i\}_{i=1}^L$ is processed by alternating layers of self-attention and position-wise MLPs: $$y \;=\; \text{MLP}\!\big(\text{SelfAttention}(\cdots \text{MLP}(\text{SelfAttention}(\{x_i\})))\big).$$ Position embeddings supply order information that self-attention alone does not encode. In autoregressive models, a causal mask guarantees that token $i$ only attends to positions $\le i$, aligning the computation with left-to-right generation.

## More Details on Transformer-Based LMs

- **Token-wise MLPs.** The MLP layers act independently on each token's hidden state (shared parameters across positions), providing nonlinear mixing of channel dimensions complementary to the cross-token mixing of attention.

- **Residual connections.** Additive shortcuts are used throughout the stack to preserve gradient flow and allow layers to learn residual refinements of the representation.

- **Layer normalization.** Normalization stabilizes optimization by reducing covariate shift within layers: $$\text{LayerNorm}(x) = a \cdot \frac{x - \text{mean}(x)}{\text{std}(x)} + b,$$ with trainable scale $a$ and bias $b$. It is applied at fixed points of the block (before/after sublayers) to keep activations in a favorable range.

- **Positional encodings.** A common choice is sinusoidal position embeddings, $$\text{embedding}(pos)
          \;=\; \big[(\cos(\omega_k\, pos),\, \sin(\omega_k\, pos))\big]_{k=1}^{d_{\text{embd}}},$$ which provide a deterministic, smooth encoding of order and relative offsets. (Other encodings are possible; here we focus on the sinusoidal case for clarity.)

- **Input composition.** Inputs are typically the sum of word and position embeddings rather than their concatenation, keeping the model width fixed while allowing both content and order information to coexist in each token vector.

- **Causal masking.** In autoregressive LMs (e.g., GPT-style), a strictly triangular attention mask enforces that token $i$ cannot attend to positions $> i$, ensuring the factorization needed for left-to-right likelihood and generation.

Overall, the attention mechanism provides content-adaptive aggregation; multi-head attention diversifies this aggregation across subspaces; self-attention enables all-to-all interaction within a layer; and positional information plus masking specialize the same machinery to the sequential constraints of language modeling. Together with residual connections, layer normalization, and token-wise MLPs, these components form the core computational pattern of Transformer-based LMs. "'

## Homework

[Homework 2: Optimization and Neural Networks](/homework/02-optimization)

<!-- prettier-ignore-end -->
