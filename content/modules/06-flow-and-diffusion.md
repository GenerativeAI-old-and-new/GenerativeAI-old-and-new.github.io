---
title: "Module 6: Flow and Diffusion"
description: "Rectified flow, ODE/SDE generative models, diffusion, and score-based dynamics."
publish: true
---

<!-- prettier-ignore-start -->

## Rectified Flow

Our goal is to learn a transport mapping $X = T^\theta(Z)$ that pushes a simple noise distribution $Z \sim P_0$ to the data distribution $X \sim P^*$. Classical approaches such as GANs, VAEs, and normalizing flows specify $T^\theta$ explicitly as a neural network, but each faces well-known training challenges: the minimax instability of GANs, the tension between reconstruction and regularization in VAEs, and the computational burden of exact likelihoods in flows.

We now discuss an alternative approach, in which $T^\theta$ is defined implicitly as the result of solving an iterative or continuous-time process whose local update is parameterized by a neural network. Such models are typically more flexible and can be easier to train because they only require learning local update directions, despite bringing higher computational cost in sampling.

> [!remark] Remark
> We distinguish two broad types of generative models:
>
> - **One-step models.** The mapping $T^\theta$ is specified directly by a neural network, as in normalizing flows, GANs, and autoencoder-based models.
> - **Process models.** The mapping is obtained by simulating an iterative or continuous-time process whose local updates are parameterized by a neural network, as in diffusion (SDE), flow (ODE), and autoregressive models.

In particular, in ODE ("flow") generative models, we train a continuous-time process to gradually transform noise $Z_0 \sim P_0$ into a sample $Z_1$ distributed as the data $P_{\mathrm{data}}$, assuming $Z_0$ and $Z_1$ have the same dimension. The dynamics are defined by the ODE $$\frac{d}{dt} Z_t = v^\theta(Z_t, t), \qquad t \in [0,1], \qquad
Z_0 \sim P_0,$$ where the velocity field $v^\theta(\cdot, t)\colon \mathbb{R}^d \to \mathbb{R}^d$ is a neural network with parameters $\theta$. At each time $t$, the field $v^\theta(Z_t, t)$ specifies the local update direction of the current state $Z_t$, and the interval $t \in [0,1]$ serves as an artificial "time" along which noise is continuously transformed into data. Thus the model starts from a simple distribution at $t=0$ and evolves smoothly until it matches the data distribution at $t=1$.

Integrating the ODE defines a mapping $T^\theta$ implicitly through $Z_1 = T^\theta(Z_0)$. Because the flow transports points along continuous trajectories, the initial noise $Z_0$ and the final output $Z_1$ necessarily share the same dimension. Throughout, we assume the ODE admits a unique solution for every initial condition so that the forward trajectory $t \mapsto Z_t$ is well defined.

##### Numerical Simulation of ODEs

Once the velocity field $v^\theta$ is trained, sampling from the model reduces to numerically solving the ODE to obtain $Z_1$. A simple discretization is the forward Euler method: $$Z_{t+\epsilon}
    = Z_t + \epsilon\, v^\theta(Z_t, t),
    \qquad
    t \in \{0, \epsilon, 2\epsilon, \ldots, 1\},$$ with step size $\epsilon = 1/N$. As $N$ grows, the discrete trajectory better approximates the continuous flow defined by the ODE.

##### Backward Integration and Inversion

Since the ODE admits a unique solution for each initial condition, the mapping $T^\theta$ is (approximately) invertible. This allows one to map a data point back to its noise counterpart by integrating the ODE in reverse. A backward Euler update takes the form $$Z_{t-\epsilon}
    = Z_t - \epsilon\, v^\theta(Z_t, t),
    \qquad
    t \in \{1, 1-\epsilon, \ldots, 0\}.$$ Starting from a sample at $t=1$, we can step backward to $t=0$. This reversibility is a key advantage of flow-based generative models and underlies their use in both sampling and likelihood estimation.

##### Maximum Likelihood Training of Neural ODEs

To understand how Neural ODEs can be trained by maximum likelihood, we consider the evolution of probability densities along the continuous flow induced by the ODE. Let $p_t^\theta$ denote the density of the random variable $Z_t$ obtained by integrating $$\frac{d}{dt} Z_t = v^\theta(Z_t, t), \qquad Z_0 \sim p_0,$$ up to time $t$. Thus $p_t^\theta$ represents the pushforward of the base density $p_0$ under the flow map defined by the velocity field $v^\theta$.

A key property of continuous-time flows is that the log-likelihood $\log p_t^\theta(x)$ admits an explicit expression in terms of the divergence (trace of the Jacobian) of the velocity field. If $\{z_\tau\}_{\tau=0}^t$ is the ODE trajectory that ends at $z_t = x$, then $$\log p_t^\theta(x)
    = \log p_0(z_0)
      - \int_0^t \mathrm{trace}\!\left( \nabla v^\theta(z_\tau, \tau) \right)
        \, d\tau.$$ This relation shows how the likelihood of a data point evolves as it is transported backward along the flow from time $t$ to time $0$. The formula is the continuous-time analogue of the change-of-variables formula for normalizing flows, where the determinant of the Jacobian is replaced by the time integral of the divergence.

### Maximum Likelihood Estimation

Given the expression above, one can in principle train the Neural ODE by maximum likelihood. For data sampled from the empirical distribution $P^{\text{data}}$, the objective takes the form $$\max_{\theta} \;
\mathbb{E}_{X^{\text{data}} \sim P^{\text{data}}}
\!\left[ \log p_{1}^{\theta}(X^{\text{data}}) \right],$$ where $\log p_1^\theta(x)$ is computed by tracing the flow backward to time $0$ and accumulating the divergence of the velocity field along the way. This approach is described in detail in .

### Practical Challenges

Although conceptually elegant, maximum likelihood training of Neural ODEs is computationally demanding. Each evaluation of the log-likelihood requires solving an ODE trajectory, and backpropagation involves differentiating again through this ODE solution. As a result, both forward and backward passes are significantly more expensive than in discrete normalizing flows.

Another limitation is that the optimal velocity field is not unique: many different flows can transport the base distribution to the same target distribution with equal likelihood. This non-uniqueness complicates optimization and can lead to training instabilities in practice.

## Rectified Flow: A Simpler and Better Approach

Although Neural ODEs can be trained using maximum likelihood, the approach is computationally heavy and difficult to optimize. It took several years of research to realize that there exists a much simpler and often better approach---one that is essentially simulation-free and avoids solving ODEs during training.

This insight first emerged from the study of diffusion generative models. Denoising diffusion probabilistic models (DDPMs) revealed that stochastic differential equation models could be trained by directly matching denoising behavior, and later developments such as denoising diffusion implicit models (DDIMs) demonstrated that these stochastic processes could be associated with deterministic ODEs. Similarly, score-based generative models were shown to correspond to a deterministic probability flow ODE, offering a new route to generative modeling without explicit simulation of stochastic dynamics.

These ideas were soon simplified and generalized into a family of formulations, including rectified flow, flow matching, and stochastic interpolants. All of these frameworks share the same principle: instead of learning an ODE by maximizing likelihood, one directly learns a velocity field that transports data and noise along simple, analytically chosen reference paths.

## Rectified Flow

To build intuition, consider the simplest case of transporting a single data point $x^{\text{data}}$. Starting from a random noise sample $X_0 \sim P_0$, we ask a basic but instructive question:

> What is the most natural ODE that moves $X_0$ to $X_1 = x^{\text{data}}$?

A natural choice is to use straight-line paths connecting noise and data. This geometric intuition leads to the so-called straight interpolation, $$X_t = t \, x^{\text{data}} + (1-t)\, X_0,$$ which traces the shortest path between the starting point $X_0$ and the target point $x^{\text{data}}$.

![image](/assets/modules/06-flow-and-diffusion/lines_one_point.png)

To derive the corresponding ODE, we differentiate the interpolation: $$\frac{d}{dt} X_t = x^{\text{data}} - X_0.$$ Since the ODE must be expressed in terms of the current state $X_t$, we substitute $X_0 = (X_t - t x^{\text{data}})/(1-t)$, which yields $$\frac{d}{dt} X_t
    = \frac{x^{\text{data}} - X_t}{1-t}.$$ This identifies the ideal velocity field for straight-line transport: $$v^{*}(x,t)
    = \frac{x^{\text{data}} - x}{1-t}.$$

The scaling factor $1/(1-t)$ guarantees that all trajectories reach the target point exactly at time $t=1$. Thus, rectified flow provides an analytically simple and geometrically intuitive velocity field, circumventing the complexities of likelihood training and numerical ODE simulation.

## Rectified Flow: Straight-Path Dynamics

For a single data point $x^{\mathrm{data}}$, the rectified flow formulation identifies a simple and fully explicit ODE that transports a noise sample $X_0$ to the target $x^{\mathrm{data}}$. Using the straight-line interpolation $$X_t = t\,x^{\mathrm{data}} + (1-t)\,X_0,$$ we can compute its time derivative and obtain the ideal dynamics $$\frac{d}{dt} X_t
    = x^{\mathrm{data}} - X_0
    = \frac{x^{\mathrm{data}} - X_t}{1-t}.$$ This ODE exactly reproduces the straight path for every $t\in[0,1]$, and it guarantees that the trajectory arrives at the destination $X_1 = x^{\mathrm{data}}$ at time $t=1$.

An appealing property of straight-line dynamics is that their numerical discretization is extremely simple. A single forward Euler update, $$\hat X_1 = X_0 + v(X_0, 0),$$ recovers the exact endpoint of the continuous path. Thus, the trajectories are not only perfectly straight but also exactly realizable with a one-step discretization.

To learn such dynamics with a neural network, we introduce a parametric velocity field $v_t^\theta$ and choose a loss that encourages the model to match the ideal velocity along the straight interpolation between noise and data.

This leads to the objective $$L(\theta)
    = \int_0^1
      \mathbb{E}_{X_0 \sim P_0}
      \Big[
            \big\|
                \tfrac{d}{dt} X_t
                - v^\theta(X_t, t)
            \big\|^2
      \Big] dt,$$ which, after substituting $\tfrac{d}{dt} X_t = x^{\mathrm{data}} - X_0$, can be written as $$L(\theta)
    = \int_0^1
      \mathbb{E}_{X_0 \sim P_0}
      \Big[
            \big\|
                x^{\mathrm{data}} - X_0
                - v^\theta(X_t, t)
            \big\|^2
      \Big] dt,$$ with $X_t = t\,x^{\mathrm{data}} + (1-t)\,X_0$. This simple regression objective captures the essence of rectified flow: the model only needs to fit the analytically known straight-path velocity, avoiding all simulation or likelihood computation.

## Rectified Flow with Multiple Data Points

The single-point case suggests a clean formulation of straight-path dynamics, but real data consist of many points. To understand how rectified flow generalizes, consider several data--noise pairs simultaneously. Suppose we draw two independent pairs $(X_0, X^{\text{data}})$ and $(X_0', X^{\text{data'}})$. Each pair induces its own straight-line interpolation, $$X_t = t\,X^{\text{data}} + (1-t)\,X_0,
\qquad
X_t' = t\,X^{\text{data}} + (1-t)\,X_0'.$$
![image](/assets/modules/06-flow-and-diffusion/line_two_points.png)
![image](/assets/modules/06-flow-and-diffusion/rf_two_points.png)
A difficulty appears immediately: straight-line interpolations from different data--noise pairs may intersect. That is, for some time $t$ and some point $x$, $$t X^{\mathrm{data}} + (1-t) X_0
    = t X^{\mathrm{data}'} + (1-t) X_0'.$$ However, intersections of this form are impossible for the trajectories of an ODE. If a point $X_t = x$ lies on a trajectory of an ODE $\dot X_t = v(X_t,t)$, then its instantaneous slope is uniquely determined by the velocity field $v(x,t)$. Thus, two different trajectories cannot pass through the same point with different directions, and naive linear interpolation does not define a valid ODE.

This raises the key question:

> Can we convert linear interpolation into a valid ODE flow that respects the non-intersection property?

The resolution is surprisingly simple. Whenever trajectories intersect, we assign them a common direction by taking the conditional expectation of the ideal velocity at that point. Formally, we define the ideal velocity field as $$v^*(x,t)
    = \mathbb{E}\!\left[\,\tfrac{d}{dt} X_t
        \,\big|\, X_t = x \right]
    = \mathbb{E}\!\left[\,X^{\mathrm{data}} - X_0
        \,\big|\, X_t = x \right],$$ where the expectation is over all data--noise pairs that interpolate to the same location $x$ at time $t$. This averaging eliminates the inconsistency created by intersections and produces a well-defined ODE velocity.

To estimate this conditional expectation in practice, we again use a regression loss. Sampling independent pairs $X^\mathrm{data} \sim P^{\mathrm{data}}$ and $X_0 \sim P_0$, and forming $X_t = t\,X^\mathrm{data} + (1-t)\,X_0$, we minimize $$L(\theta)
    = \int_0^1
      \mathbb{E}_{X^\mathrm{data},X_0}
      \Big[
        \big\|
            X^\mathrm{data} - X_0
            - v^\theta(X_t, t)
        \big\|^2
      \Big] \, dt.$$ This loss has a natural statistical interpretation. For any pair of random variables $(X,Y)$, the function $v^*(X) = \mathbb{E}[Y \mid X]$ is known to be the minimizer of $\mathbb{E}[\|Y - v(X)\|^2]$ over all measurable functions $v$. Thus, the learned velocity field $v^\theta$ approximates exactly the conditional expectation defining the ideal rectified flow.

## Rectified Flow Loss and Training Procedure

The rectified flow objective follows directly from the conditional expectation characterization derived earlier. Given independent draws $X^{\mathrm{data}} \sim P^{\mathrm{data}}$ and $X_0 \sim P_0$, and the straight-line interpolation $$X_t = t\,X^{\mathrm{data}} + (1-t)\,X_0,$$ we define the rectified flow loss as $$L(\theta)
    = \int_{0}^{1}
      \mathbb{E}_{X^{\mathrm{data}},X_0}
      \Big[
        \big\|
            X^{\mathrm{data}} - X_0
            - v^\theta(X_t, t)
        \big\|^{2}
      \Big] \, dt.$$ This regression loss trains the neural velocity field $v^\theta$ to approximate the ideal velocity $\mathbb{E}[X^{\mathrm{data}} - X_0 \mid X_t = x]$, thus constructing a valid ODE flow that interpolates between noise and data.

In practice, the integral and expectations are optimized using stochastic gradient descent. Each iteration draws a minibatch of samples $$X^{\mathrm{data}} \sim P^{\mathrm{data}},
\qquad
X_0 \sim P_0,
\qquad
t \sim \mathrm{Uniform}([0,1]),$$ and forms the interpolated points $$X_t
    = t\,X^{\mathrm{data}}
    + (1-t)\,X_0.$$ The instantaneous training loss for each sample is then computed as $$\texttt{Loss}
    = \big\|
        X^{\mathrm{data}} - X_0
        - v^\theta(X_t, t)
      \big\|^{2},$$ and the parameters $\theta$ are updated using its gradient. This procedure provides an unbiased stochastic estimate of the full objective $L(\theta)$.

After training, the learned velocity field defines a generative model. Starting from a noise sample $Z_0 \sim P_0$, one solves the ODE $$\frac{d}{dt} Z_t = v^\theta(Z_t, t),$$ transporting the initial noise toward the data distribution. The solution at time $t=1$ produces a generated sample $Z_1$, completing the rectified flow generative process.

## Theoretical Properties of Rectified Flow

To analyze the behavior of rectified flow, consider independent noise--data pairs $(X_0, X_1)$ with $X_0 \sim P_0$ and $X_1 \sim P^{\mathrm{data}}$. The straight-line interpolation between them is given by $$X_t = t X_1 + (1-t) X_0, \qquad t \in [0,1].$$ This linear path defines a simple time-indexed random process $\{X_t\}_{t\in[0,1]}$.

Rectified flow, on the other hand, induces a second process $\{Z_t\}_{t\in[0,1]}$ by solving the ODE $$\frac{d}{dt} Z_t = v^{*}(Z_t, t),
\qquad
Z_0 \sim P_0,$$ where the ideal velocity field is $$v^*(Z_t,t)
    = \mathbb{E}[X_1 - X_0 \mid X_t = Z_t].$$ This velocity field takes the average direction of all straight-line interpolations passing through any given point $Z_t$, ensuring that the rectified flow dynamics produce valid ODE trajectories.

Although the interpolation process $\{X_t\}$ and the ODE process $\{Z_t\}$ are generally different random processes, a crucial property holds:

$$X_t \overset{d}{=} Z_t, \qquad \forall t \in [0,1],$$ meaning that they share the same marginal distribution at every time. This agreement of marginals is a fundamental feature that enables rectified flow to match the data distribution while maintaining ODE-consistent dynamics.

![image](/assets/modules/06-flow-and-diffusion/rf_x.png)

> [!theorem] Theorem
> Let $p_t$ denote the marginal density of either process $\{X_t\}$ or $\{Z_t\}$. Then $p_t$ satisfies the continuity equation
>
> $$
> \frac{d}{dt} p_t(x)
> = - \nabla \cdot \left( v_t^{*}(x)\, p_t(x) \right),
> \qquad \forall t \in [0,1].
> $$
>
> Here the divergence of a vector field $g(x) = (g_1(x),\ldots,g_d(x))$ is
>
> $$
> \nabla \cdot g(x) = \sum_{i=1}^{d} \partial_{x_i} g_i(x).
> $$
>
> If the continuity equation admits a unique solution for the given initial density $p_0$, then both processes must share the same marginal distribution at all times. Thus, the rectified flow ODE and the linear interpolation induce the same family of marginals $\{p_t\}_{t\in[0,1]}$.

## Proof of the Continuity Equation

To derive the continuity equation satisfied by the marginal densities $\{p_t\}$, we begin by examining how the expectation of a smooth test function $h(x)$ evolves over time. By definition, $$\frac{d}{dt} \mathbb{E}[h(X_t)]
    = \frac{d}{dt} \int h(x)\, p_t(x)\, dx
    = \int h(x)\, \frac{d}{dt} p_t(x)\, dx.$$ This expresses the time derivative of the expectation in terms of the time derivative of the density.

We now compute the same derivative in a second way, using the dynamics of the process $X_t$. Applying the chain rule gives $$\frac{d}{dt} \mathbb{E}[h(X_t)]
    = \mathbb{E}\big[ \nabla h(X_t)^{\top} \dot X_t \big].$$ Using the tower property of conditional expectation, $$\mathbb{E}[\nabla h(X_t)^{\top} \dot X_t]
    = \mathbb{E}\Big[
        \mathbb{E}\big[
            \nabla h(X_t)^{\top} \dot X_t
            \,\big|\,
            X_t
        \big]
      \Big].$$ Since $\nabla h(X_t)$ is measurable with respect to $X_t$, it can be factored outside the inner expectation: $$= \mathbb{E}\big[
        \nabla h(X_t)^{\top}
        \mathbb{E}[\dot X_t \mid X_t]
      \big].$$ By definition of the ideal velocity, $$v^*(x,t) = \mathbb{E}[\dot X_t \mid X_t = x],$$ we obtain $$\frac{d}{dt} \mathbb{E}[h(X_t)]
    = \mathbb{E}\big[
        \nabla h(X_t)^{\top}
        v^*(X_t, t)
      \big]
    = \int \nabla h(x)^{\top} v^*(x,t)\, p_t(x)\, dx.$$

Finally, we apply integration by parts: $$\int \nabla h(x)^{\top} v^*(x,t)\, p_t(x)\, dx
    = - \int h(x)\, \nabla \cdot \big( v^*(x,t)\, p_t(x) \big)\, dx,$$ assuming that $h$ vanishes outside a compact set so that boundary terms disappear. Equating this with the earlier expression for $\tfrac{d}{dt} \mathbb{E}[h(X_t)]$ yields $$\int h(x)\, \frac{d}{dt} p_t(x)\, dx
    = - \int h(x)\,
        \nabla \cdot \big( v^*(x,t)\, p_t(x) \big)\, dx,
\qquad \forall\,h.$$ Since this equality holds for all test functions $h$, we conclude that the density $p_t$ satisfies the continuity equation $$\frac{d}{dt} p_t(x)
    = - \nabla \cdot \big( v^*(x,t)\, p_t(x) \big).$$

We have used the standard integration-by-parts identity $$\int \nabla h(x)^{\top} g(x)\, dx
    = - \int h(x)\, \nabla \cdot g(x)\, dx,$$ valid for smooth $h$ that vanishes outside a finite region.

---

## Rectified Flow: ODE Generative Model

![image](/assets/modules/06-flow-and-diffusion/line_two_points.png)
![image](/assets/modules/06-flow-and-diffusion/rf_two_points.png)

- Rectified Flow: Learning ODE generative models from interpolations
- Draw batches of samples: $$X^\text{data}\sim P^\text{data},\quad X_0\sim P_0,\quad t\sim \text{Uniform}([0,1]).$$
- Construct interpolation between noise and data: $$X_t = t\,X^\text{data} + (1 - t)\,X_0.$$
- Minimize the loss function: $$\min_{\theta} \mathbb{E}\left[ \left\lVert X^\text{data} - X_0 - v^\theta(X_t, t )\right\rVert^2\right].$$
- After training, generate new data by solving the ODE: $$\,d Z_t = v^\theta(Z_t, t)\,\,d t,\quad Z_0\sim P_0.$$

## ODE vs. SDE Models

- **ODE (Flow)**: Generate data $Z_1$ by solving $$\,d Z_t = v(Z_t, t)\,\,d t.$$

- **SDE (Diffusion)**: Generate data $Z_1$ by solving $$\,d Z_t = v(Z_t, t)\,\,d t + \sigma_t\,\,d W_t,$$ where $W_t$ is standard Brownian motion.
  ![image](/assets/modules/06-flow-and-diffusion/rf_two_points.png)
  ![image](/assets/modules/06-flow-and-diffusion/diffusion1.png)
- **SDE vs. ODE: How and Why?**

## Rectified Flow Recap

- Assume we have trained a rectified flow ODE model: $$\,d Z_t = v_{RF}(Z_t, t)\,\,d t.$$
- **Marginal preserving property**: At each time $t$, the distribution of $Z_t$ matches the distribution $\rho_t$ of the interpolation $$X_t = t X_1 + (1 - t) X_0.$$ Thus, $\rho_1$ matches the data distribution $X_1$.
- However, in practice, we cannot perfectly simulate the ODE due to model and numerical error.

![image](/assets/modules/06-flow-and-diffusion/euler_sample_result.png)

## Diffusion = ODE + Langevin

- If we know $\rho_t$, we can correct errors using Langevin dynamics: $$\,d Z_\tau = \sigma_\tau^2 \nabla \log\rho_\tau(Z_\tau)\,\,d \tau + \sqrt{2}\,\sigma_\tau\,\,d W_\tau.$$ (Here, $\tau$ is an auxiliary time scale for Langevin correction at fixed $t$.)
- Combine ODE and Langevin dynamics directly:

$$
\,d Z_t =
        \underbrace{v_{RF}(Z_t, t)\,\,d t}_{\text{Rectified Flow}} +
        \underbrace{\sigma_t^2 \nabla \log\rho_t(Z_t)\,\,d t + \sqrt{2}\,\sigma_t\,\,d W_t}_{\text{Langevin Dynamics}}
$$

![image](/assets/modules/06-flow-and-diffusion/sde_velocity.png)
![image](/assets/modules/06-flow-and-diffusion/sde_score.png)

- **Key**: For Gaussian noise, the score function $\nabla \log \rho_t(x)$ is directly related to $v_{RF}$: $$\textbf{Tweedie's Formula:}\quad
              \nabla \log \rho_t(x) = \frac{t v_{RF}(x,t)- x}{1-t}.$$

## Tweedie's Formula

At time $t$, let $\rho_t$ be the density of the interpolation $$X_t = t X_1 + (1 - t) X_0,\quad X_1 \sim P^{\text{data}},\quad X_0\sim\mathcal{N}(0,I).$$

Then, Tweedie's formula gives: $$\nabla \log \rho_t(x) = \mathbb{E}\left[\frac{t X_1 - x}{(1 - t)^2}\,\bigg|\, X_t = x\right].$$

> [!proof] Proof
> _Proof._ Given $X_t = x$, we have $t X_1 + (1 - t) X_0 = x$, thus $$X_0 = \frac{x - t X_1}{1 - t}.$$
>
> Since $X_0 \sim \mathcal{N}(0, I)$, we have $$\rho_t(x) \propto \int \rho_1(x_1) \exp\left(-\frac{\|x - t X_1\|^2}{2(1 - t)^2}\right)\,\,d x_1.$$
>
> Taking the gradient: $$\nabla \log \rho_t(x)
>             = \frac{\int \rho_1(x_1) \exp\left(-\frac{\|x - t X_1\|^2}{2(1 - t)^2}\right)\frac{t X_1 - x}{(1 - t)^2}\,\,d x_1}{\rho_t(x)}.$$
>
> Recognizing the conditional expectation, we get: $$\nabla \log \rho_t(x) = \mathbb{E}\left[\frac{t X_1 - x}{(1 - t)^2}\,\bigg|\, X_t = x\right].$$ ◻

- On the other hand, the RF velocity is: $$v_{RF}(x,t) = \mathbb{E}\left[\frac{X_1 - x}{1 - t}\,\bigg|\, X_t = x\right].$$

  Thus, $\mathbb{E}[X_1\,|\,X_t=x] = x + (1 - t)v_{RF}(x,t)$, and substituting this gives: $$\nabla \log \rho_t(x) = \frac{t v_{RF}(x,t) - x}{1 - t}.$$

<!-- prettier-ignore-end -->
