---
title: "Homework 1: Probability and MLE"
description: "Gaussian density operations, KL divergence, categorical MLE, and Langevin sampling for energy-based models."
publish: true
---

[Back to Module 1 notes](/modules/01-probability-basics)

## Problem 1

> [!problem|Gaussian Density Operations]
> Let $p_i(x)$ be the density of $\mathcal N(\mu_i,\sigma_i^2)$ for $i=1,2$ (with $\sigma_i>0$), defined on $\mathbb R$. Answer the following:
>
> 1.  Let
>
>     $$p(x)=\frac{p_1(x)p_2(x)}{Z},\qquad Z=\int_{\mathbb R} p_1(x)p_2(x)\,\mathrm d x.$$
>
>     Is $p(x)$ a valid density? If so, identify the distribution explicitly, including its parameters.
>
> 2.  Let
>
>     $$p(x)=\frac12\big(p_1(x)+p_2(x)\big).$$
>
>     Is $p(x)$ a valid density? If so, describe its distributional form. Is it generally a single Gaussian?
>
> 3.  Let $X=X_1+X_2$, where $X_1\sim\mathcal N(\mu_1,\sigma_1^2)$ and $X_2\sim\mathcal N(\mu_2,\sigma_2^2)$ are independent. What is the distribution of $X$? Give its mean and variance.
> 4.  Let $X=Z^2$ where $Z\sim\mathcal N(0,1)$. Derive the density of $X$ and state its support.

<!--
> [!solution]- Solution
>
> 1.  Yes. Since $p_1(x)p_2(x)\ge0$ and $Z=\int p_1(x)p_2(x)\,\mathrm d x$, the normalized function integrates to one.
>
>     Expanding the exponent,
>
>     $$
>     p_1(x)p_2(x)\propto
>     \exp\left[
>     -\frac12\left(
>     \frac{(x-\mu_1)^2}{\sigma_1^2}
>     +\frac{(x-\mu_2)^2}{\sigma_2^2}
>     \right)\right].
>     $$
>
>     Completing the square gives another Gaussian:
>
>     $$
>     p(x)=\mathcal N(x;m,v),
>     $$
>
>     where
>
>     $$
>     v=\left(\frac1{\sigma_1^2}+\frac1{\sigma_2^2}\right)^{-1}
>     =\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2},
>     $$
>
>     and
>
>     $$
>     m=v\left(\frac{\mu_1}{\sigma_1^2}+\frac{\mu_2}{\sigma_2^2}\right)
>     =
>     \frac{\mu_1\sigma_2^2+\mu_2\sigma_1^2}{\sigma_1^2+\sigma_2^2}.
>     $$
>
>     The new mean is a precision-weighted average of the two means. The normalizing constant is
>
>     $$
>     Z=\mathcal N(\mu_1;\mu_2,\sigma_1^2+\sigma_2^2)
>     =
>     \frac{1}{\sqrt{2\pi(\sigma_1^2+\sigma_2^2)}}
>     \exp\left(-\frac{(\mu_1-\mu_2)^2}{2(\sigma_1^2+\sigma_2^2)}\right).
>     $$
>
> 2.  Yes. It is nonnegative and integrates to
>
>     $$
>     \frac12\int p_1(x)\,\mathrm d x+\frac12\int p_2(x)\,\mathrm d x=1.
>     $$
>
>     This is a two-component Gaussian mixture:
>
>     $$
>     p(x)=\frac12\mathcal N(x;\mu_1,\sigma_1^2)
>     +\frac12\mathcal N(x;\mu_2,\sigma_2^2).
>     $$
>
>     It is generally not a single Gaussian. It becomes a single Gaussian only in the degenerate case where the two components are the same distribution, i.e. $\mu_1=\mu_2$ and $\sigma_1^2=\sigma_2^2$.
>
> 3.  The sum of independent Gaussian random variables is Gaussian:
>
>     $$
>     X_1+X_2\sim
>     \mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2).
>     $$
>
>     The mean adds by linearity of expectation, and the variance adds because the variables are independent.
>
> 4.  For $X=Z^2$, the support is $x\ge0$. For $x>0$, the equation $x=z^2$ has two preimages, $z=\sqrt{x}$ and $z=-\sqrt{x}$. Therefore
>
>     $$
>     f_X(x)
>     =
>     \phi(\sqrt{x})\frac1{2\sqrt{x}}
>     +
>     \phi(-\sqrt{x})\frac1{2\sqrt{x}}
>     =
>     \frac1{\sqrt{2\pi x}}e^{-x/2},
>     \qquad x>0.
>     $$
>
>     Thus $X\sim\chi^2(1)$, with density $f_X(x)=0$ for $x<0$.
-->

## Problem 2

> [!problem|KL Divergence]
>
> 1.  Consider two discrete distributions over $\Omega=\{1,2\}$:
>
>     $$
>     P=(0.5,\,0.5) \quad\text{(uniform)}, \qquad
>     Q=(1.0,\,0.0) \quad\text{(deterministic at outcome 1)}.
>     $$
>
>     Hand-calculate $\operatorname{KL}(P \,\|\, Q)$ and $\operatorname{KL}(Q \,\|\, P)$.
>
> 2.  Now let the sample space be $\Omega=\{1,2,3\}$ and consider
>
>     $$
>     P=(0.5,\,0.5,\,0.0), \qquad
>     Q=(1.0,\,0.0,\,0.0).
>     $$
>
>     Compute $\operatorname{KL}(P \,\|\, Q)$ and $\operatorname{KL}(Q \,\|\, P)$. Use the convention that $0\log 0=0$, because $\lim_{\epsilon \to 0^+}\epsilon\log\epsilon=0$, and that $a\log(a/0)=+\infty$ for $a>0$.
>
> 3.  For a discrete distribution $R$ on $\Omega$, define $\operatorname{supp}(R):=\{x\in\Omega:\ R(x)>0\}$, which is the set of all elements with positive probability.
>     1.  Assume $\operatorname{KL}(Q \,\|\, P) < +\infty$, must it be true that $\operatorname{supp}(Q)\subseteq\operatorname{supp}(P)$? Briefly justify.
>     2.  Assume $\operatorname{KL}(Q \,\|\, P) < +\infty$, must it be true that $\operatorname{supp}(P)\subseteq\operatorname{supp}(Q)$? If yes, explain; if not, give a counterexample and explain why.
> 4.  Consider the following divergence:
>
>     $$
>     \mathrm{D}(P,Q)
>     \;=\;
>     \int \big(\alpha\,p(x)+(1-\alpha)\,q(x)\big)\,
>     \Big[
>     \frac{p(x)}{q(x)} - \log\!\Big(\frac{p(x)}{q(x)}\Big) - 1
>     \Big]\; \mathrm d x.
>     $$
>
>     Here $\alpha \in [0,1]$, and $p(x)$ and $q(x)$ are the densities of $P$ and $Q$ respectively. Answer the following questions:
>     1.  Is this divergence a valid notion of discrepancy? Explain your reasoning.
>     2.  Under what conditions does this divergence reduce to the KL divergence (either $\operatorname{KL}(P \,\|\, Q)$ or $\operatorname{KL}(Q \,\|\, P)$)?

<!--
> [!solution]- Solution
>
> 1.  Using
>
>     $$
>     \operatorname{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)},
>     $$
>
>     we get
>
>     $$
>     \operatorname{KL}(P\|Q)
>     =
>     0.5\log\frac{0.5}{1.0}
>     +
>     0.5\log\frac{0.5}{0}
>     =
>     +\infty.
>     $$
>
>     This is infinite because $P$ puts positive mass on outcome $2$, but $Q$ assigns outcome $2$ zero probability. In the reverse direction,
>
>     $$
>     \operatorname{KL}(Q\|P)
>     =
>     1\cdot\log\frac1{0.5}
>     +0\cdot\log\frac0{0.5}
>     =
>     \log2.
>     $$
>
> 2.  The third coordinate does not change the conclusion because both distributions assign it probability zero:
>
>     $$
>     \operatorname{KL}(P\|Q)
>     =
>     0.5\log\frac{0.5}{1}
>     +0.5\log\frac{0.5}{0}
>     +0
>     =
>     +\infty,
>     $$
>
>     and
>
>     $$
>     \operatorname{KL}(Q\|P)=\log2.
>     $$
>
> 3.  If $\operatorname{KL}(Q\|P)<+\infty$, then every point with $Q(x)>0$ must also have $P(x)>0$. Otherwise one term would be
>
>     $$
>     Q(x)\log\frac{Q(x)}{0}=+\infty.
>     $$
>
>     Therefore $\operatorname{supp}(Q)\subseteq\operatorname{supp}(P)$.
>
>     The reverse inclusion is not required. A counterexample is
>
>     $$
>     P=(0.5,0.5),\qquad Q=(1,0).
>     $$
>
>     Here $\operatorname{supp}(P)=\{1,2\}$ and $\operatorname{supp}(Q)=\{1\}$, but
>
>     $$
>     \operatorname{KL}(Q\|P)=\log2<+\infty.
>     $$
>
> 4.  Let
>
>     $$t(x)=\frac{p(x)}{q(x)},\qquad g(t)=t-\log t-1.$$
>
>     For $t>0$, $g(t)\ge0$ and $g(t)=0$ if and only if $t=1$. Also
>
>     $$
>     \alpha p(x)+(1-\alpha)q(x)
>     =
>     q(x)\big(\alpha t(x)+1-\alpha\big)\ge0.
>     $$
>
>     Hence the integrand is nonnegative wherever the ratio is well-defined, and the divergence is nonnegative, possibly $+\infty$. It equals zero only when $p=q$ almost everywhere, so it is a valid divergence/discrepancy in the usual sense: nonnegative and zero only when the two distributions match. It is not a metric because it is not symmetric and does not satisfy a triangle inequality in general.
>
>     Expanding the two pieces gives
>
>     $$
>     \begin{aligned}
>     \mathrm D(P,Q)
>     &=
>     (1-\alpha)\int q(t-\log t-1)\,\mathrm d x
>     +\alpha\int p(t-\log t-1)\,\mathrm d x\\
>     &=
>     (1-\alpha)\operatorname{KL}(Q\|P)
>     +\alpha\left(\chi^2(P\|Q)-\operatorname{KL}(P\|Q)\right),
>     \end{aligned}
>     $$
>
>     where $\chi^2(P\|Q)=\int p(x)^2/q(x)\,\mathrm d x-1$. Therefore:
>
>     - If $\alpha=0$, then $\mathrm D(P,Q)=\operatorname{KL}(Q\|P)$.
>     - For $\alpha>0$, it is generally not equal to either direction of KL, except in the trivial case $P=Q$ where all divergences are zero.
-->

## Problem 3

> [!problem|Categorical MLE with Softmax]
> Let $x_1,\dots,x_n$ be i.i.d. observations taking values in $\{1,\ldots,K\}$. We parameterize the (unconditional) categorical probabilities via a softmax:
>
> $$
> \label{equ:softmaxprob}
> p_\theta(x=k) \;=\; \frac{\exp(\theta_k)}{\sum_{j=1}^K \exp(\theta_j)},
> \qquad \theta=(\theta_1,\ldots,\theta_K)\in\mathbb{R}^K.
> $$
>
> Here $(\theta_1,\ldots,\theta_K)$ are $K$ unconstrained parameters. _Note:_ the softmax is invariant to adding a constant to all coordinates, i.e. $p_{\theta}=p_{\theta+c\mathbf{1}}$ for any $c\in\mathbb{R}$.
>
> **Exercise:**
>
> 1.  Write down the log-likelihood $\ell(\theta)$ for this model (you may express it using the empirical counts $n_k=\sum_{i=1}^n \mathbf{1}\{x_i=k\}$).
> 2.  Compute the gradient $\nabla_\theta \ell(\theta)$ and set it to zero to derive the maximum likelihood estimator $\hat\theta_{\text{MLE}}$. Discuss: Is the parameter $\hat\theta_{\text{MLE}}$ unique? Is the induced distribution $p_{\hat\theta_{\text{MLE}}}$ unique?
> 3.  Directly evaluating exponentials can overflow/underflow. For each expression below, state whether it is numerically stable (in standard 64-bit floating point) and explain briefly:
>     1.  $\dfrac{\exp(10000)}{\exp(20000)+\exp(10000)}$,
>     2.  $\dfrac{\exp(-20000)}{\exp(-10000)+\exp(-20000)}$,
>     3.  $\dfrac{\exp(-10000)}{\exp(-10000)+\exp(0)}$.
>
>     (Clarify whether it may produce `Inf`/`NaN` or loss of significance.)
>
> 4.  Describe a numerically stable way to compute the softmax for a general vector $(\theta_1,\ldots,\theta_K)$, and give a stable formula for the log-likelihood.

<!--
> [!solution]- Solution
>
> Let
>
> $$n_k=\sum_{i=1}^n\mathbf 1\{x_i=k\},\qquad \sum_{k=1}^K n_k=n.$$
>
> The log-likelihood is
>
> $$
> \ell(\theta)
> =
> \sum_{i=1}^n\log p_\theta(x_i)
> =
> \sum_{k=1}^K n_k\theta_k
> -
> n\log\sum_{j=1}^K e^{\theta_j}.
> $$
>
> Let $s_k(\theta)=p_\theta(x=k)$. Differentiating gives
>
> $$
> \frac{\partial\ell}{\partial\theta_k}
> =
> n_k-ns_k(\theta).
> $$
>
> Setting the gradient to zero gives
>
> $$
> s_k(\hat\theta)=\frac{n_k}{n}.
> $$
>
> Thus the MLE distribution is the empirical categorical distribution. If every $n_k>0$, one possible parameterization is
>
> $$
> \hat\theta_k=\log n_k+c,
> $$
>
> for any constant $c\in\mathbb R$. The parameter vector is not unique because softmax is unchanged when we add the same constant to all coordinates. The induced probability distribution is unique:
>
> $$
> p_{\hat\theta}(x=k)=\frac{n_k}{n}.
> $$
>
> If some $n_k=0$, no finite softmax parameter gives exactly zero probability. In that case the MLE over probabilities still has $\hat p_k=0$, but the corresponding softmax parameter is reached only in the limit $\theta_k\to-\infty$ for categories with zero count.
>
> For the numerical examples, direct `float64` evaluation roughly overflows for $\exp(x)$ when $x\gtrsim709$ and underflows to zero for very negative $x$.
>
> 1.  $\exp(10000)/(\exp(20000)+\exp(10000))$ is unstable if evaluated directly: both exponentials overflow to `Inf`, producing `Inf/Inf`, i.e. `NaN`. The mathematical value is close to $0$.
> 2.  $\exp(-20000)/(\exp(-10000)+\exp(-20000))$ is also unstable if evaluated directly: both numerator and denominator underflow to zero, giving `0/0`, i.e. `NaN`. The mathematical value is close to $0$.
> 3.  $\exp(-10000)/(\exp(-10000)+\exp(0))$ underflows in the numerator and returns $0/(0+1)=0$, which matches the limiting floating-point value for this tiny probability.
>
> The standard stable softmax subtracts the maximum logit. Let $m=\max_j\theta_j$. Then
>
> $$
> \operatorname{softmax}_k(\theta)
> =
> \frac{e^{\theta_k-m}}{\sum_{j=1}^K e^{\theta_j-m}},
> $$
>
> and
>
> $$
> \log\sum_{j=1}^K e^{\theta_j}
> =
> m+\log\sum_{j=1}^K e^{\theta_j-m}.
> $$
>
> Therefore a stable log-likelihood is
>
> $$
> \ell(\theta)
> =
> \sum_{k=1}^K n_k\theta_k
> -
> n\left(m+\log\sum_{j=1}^K e^{\theta_j-m}\right).
> $$
-->

## Problem 4

> [!problem|Energy-Based Models with Langevin Sampling]
> We want to fit a dataset $\mathcal{D} = \{x_i\}_{i=1}^n$ with an energy-based model on $\mathbb{R}^d$ of the form
>
> $$
> p_{\theta}(x)=\frac{\exp\big(f_\theta(x)\big)}{Z_\theta},
> \qquad
> Z_\theta=\int_{\mathbb{R}^d}\exp\big(f_\theta(x)\big)\,\mathrm d x.
> $$
>
> Here $f_\theta(x)$ is the (unnormalized) log-density (i.e., negative energy), and $Z_\theta$ is the partition function. To ensure integrability, we use
>
> $$f_\theta(x)=\mathrm{NN}_w(x)-\frac{\|x-\mu\|^2}{2\sigma^2},$$
>
> with $\mathrm{NN}_w$ a neural network (e.g. MLP) parameterized by $w$, $\mu\in\mathbb{R}^d$, and $\sigma>0$ a _scalar_ so the Gaussian term is isotropic ($\sigma^2 I_d$). We write $\theta=(w,\mu,\sigma)$; for simplicity, treat $(\mu,\sigma)$ as fixed hyperparameters (e.g., $\mu=\mathbf{0}$, $\sigma=0.1$) unless you wish to tune them manually.
>
> You will implement a toy MLE pipeline in $d=2$ and test it on the provided dataset in the starter Colab.
>
> <https://colab.research.google.com/drive/1aNetPvIM2LH2PinAKQxVs_Utwpyy4uYn?usp=sharing>
>
> 1.  **Sampling (Langevin vs. grid).** Exact sampling from $p_\theta$ is not available. Besides a provided brute-force grid sampler (on a bounded $2$D window with discretization), implement _Langevin Algorithm_:
>
>     $$
>     x^{t+1}=x^t+\frac{\epsilon}{2}\,\nabla_x \log p_\theta(x^t)+\sqrt{\epsilon}\,\xi^t,
>     \qquad
>     \xi^t\overset{\text{i.i.d.}}{\sim}\mathcal{N}(0,I_d),
>     $$
>
>     with stepsize $\epsilon>0$. It is expected that $x^t$ approximately follows $p_{\theta}$ when the number of steps $t$ is very large and the step size $\epsilon$ is very small. _Note:_ since $Z_\theta$ does not depend on $x$, $\nabla_x\log p_\theta(x)=\nabla_x f_\theta(x)$. Initialize, e.g., $x^0\sim\mathcal{N}(\mu,\sigma^2 I_d)$; run multiple times.
>
>     **Task:** Implement Langevin dynamics and qualitatively compare to the grid sampler.
>
> 2.  **MLE training.** Define the average log-likelihood
>
>     $$\ell(\theta)=\mathbb{E}_{x\sim \hat P_{\text{data}}}\big[f_\theta(x)\big]-\log Z_\theta,$$
>
>     where $\hat P_{\text{data}}$ is the empirical distribution of the dataset. The negative log-likelihood is therefore
>
>     $$\mathcal L(\theta)=-\ell(\theta)=-\mathbb{E}_{x\sim \hat P_{\text{data}}}\big[f_\theta(x)\big]+\log Z_\theta.$$
>
>     In the lecture we have shown that
>
>     $$\nabla_\theta \ell(\theta)=\mathbb{E}_{x\sim \hat P_{\text{data}}}\big[\nabla_\theta f_\theta(x)\big]\;-\;\mathbb{E}_{x\sim p_\theta}\big[\nabla_\theta f_\theta(x)\big].$$
>
>     **Task:** Implement gradient descent on the negative log-likelihood $\mathcal L(\theta)$, equivalently gradient ascent on $\ell(\theta)$:
>
>     $$\theta_{t+1}\gets \theta_t-\eta\,\widehat{\nabla_\theta \mathcal L(\theta_t)},$$
>
>     where the model expectation is approximated with samples from your Langevin sampler at current $\theta_t$. Train the model until it fits the toy data well (e.g., samples visually match data).

<!--
> [!solution]- Solution
>
> The key identity is
>
> $$
> \nabla_x\log p_\theta(x)=\nabla_x f_\theta(x),
> $$
>
> because $\log Z_\theta$ does not depend on $x$. One implementation is:
>
> ```python
> def langevin_sampler(f_theta, x0, num_steps=200, step_size=1e-3):
>     x = x0.detach()
>     for _ in range(num_steps):
>         x = x.detach().requires_grad_(True)
>         score = torch.autograd.grad(f_theta(x).sum(), x)[0]
>         noise = torch.randn_like(x)
>         x = x + 0.5 * step_size * score + step_size**0.5 * noise
>     return x.detach()
> ```
>
> Some references use the equivalent convention
>
> $$x_{t+1}=x_t+\delta\nabla_x f_\theta(x_t)+\sqrt{2\delta}\xi_t.$$
>
> This is the same update after setting $\epsilon=2\delta$, with the drift and noise scales changed together.
>
> In $2$D, the grid sampler evaluates $\exp(f_\theta(x))$ on a finite grid and normalizes the values. This gives a direct finite-grid approximation to $p_\theta$. Langevin sampling replaces the grid by a Markov chain approximation; too few steps, too large a step size, or poor initialization can produce biased samples.
>
> For training, the log-likelihood gradient is
>
> $$
> \nabla_\theta \ell(\theta)
> =
> \mathbb E_{\text{data}}[\nabla_\theta f_\theta(x)]
> -
> \mathbb E_{p_\theta}[\nabla_\theta f_\theta(x)].
> $$
>
> Therefore the negative log-likelihood gradient is
>
> $$
> \nabla_\theta \mathcal L(\theta)
> =
> -
> \mathbb E_{\text{data}}[\nabla_\theta f_\theta(x)]
> +
> \mathbb E_{p_\theta}[\nabla_\theta f_\theta(x)].
> $$
>
> In code, we can estimate the two expectations with a data batch and a batch of Langevin samples:
>
> ```python
> x_data = next_data_batch()
> x_model = langevin_sampler(f_theta, x_init, num_steps=K, step_size=eps)
>
> data_score = f_theta(x_data).mean()
> model_score = f_theta(x_model.detach()).mean()
>
> loss = -data_score + model_score
> optimizer.zero_grad()
> loss.backward()
> optimizer.step()
> ```
>
> The `detach()` is intentional: this update uses the samples to estimate the model expectation, but does not backpropagate through the whole sampling chain.
>
> After training:
>
> - samples from the trained model visually matching the data modes,
> - grid samples and Langevin samples concentrating on the same modes in $2$D,
> - training becoming unstable when Langevin step size is too large,
> - poor mixing when the number of Langevin steps is too small.
>
> With the grid estimate of $\log Z_\theta$, the $2$D loss is
>
> $$\mathcal L(\theta)\approx -\frac1B\sum_{i=1}^B f_\theta(x_i)+\log Z_\theta^{\text{grid}}.$$
>
> This finite-grid objective can be compared with the Langevin-based objective in the $2$D experiment.
-->

## Optional Problems

## Optional Problem 1

> [!problem|Laplace MLE]
> Consider a dataset $\{x_i\}_{i=1}^n$ generated from a distribution with density function:
>
> $$p_\theta(x) = \frac{\theta}{2} \exp(-\theta|x|), \quad x \in \mathbb{R}, \theta > 0.$$
>
> This is known as the Laplace (or double exponential) distribution.
>
> **Exercise:**
>
> 1.  Write down the log-likelihood function $\ell(\theta)$ for this distribution.
> 2.  Find the maximum likelihood estimator $\hat{\theta}_{\text{MLE}}$ by maximizing $\ell(\theta)$.
> 3.  Show that $\hat{\theta}_{\text{MLE}}$ can be written as a simple function of $\frac{1}{n}\sum_{i=1}^n |x_i|$.

<!--
> [!solution]- Solution
>
> The log-likelihood is
>
> $$
> \ell(\theta)
> =
> \sum_{i=1}^n\left(\log\frac{\theta}{2}-\theta |x_i|\right)
> =
> n\log\theta-n\log2-\theta\sum_{i=1}^n |x_i|.
> $$
>
> Differentiate:
>
> $$
> \frac{d\ell}{d\theta}
> =
> \frac{n}{\theta}-\sum_{i=1}^n |x_i|.
> $$
>
> Setting this to zero gives
>
> $$
> \hat\theta_{\text{MLE}}
> =
> \frac{n}{\sum_{i=1}^n |x_i|}
> =
> \frac{1}{\frac1n\sum_{i=1}^n |x_i|}.
> $$
>
> The second derivative is $-n/\theta^2<0$, so this critical point is a maximum.
-->

## Optional Problem 2

> [!problem|Exponential MLE]
> Consider a dataset $\{x_i\}_{i=1}^n$ of nonnegative numbers generated from an exponential distribution with density:
>
> $$p_\theta(x) = \theta \exp(-\theta x), \quad x \geq 0, \theta > 0.$$
>
> **Exercise:**
>
> 1.  Write down the log-likelihood function $\ell(\theta)$.
> 2.  Find the maximum likelihood estimator $\hat{\theta}_{\text{MLE}}$ by maximizing $\ell(\theta)$.
> 3.  Show that $\hat{\theta}_{\text{MLE}} = \frac{1}{\bar{x}}$, where $\bar{x}$ is the sample mean.

<!--
> [!solution]- Solution
>
> The log-likelihood is
>
> $$
> \ell(\theta)
> =
> \sum_{i=1}^n(\log\theta-\theta x_i)
> =
> n\log\theta-\theta\sum_{i=1}^n x_i.
> $$
>
> Differentiating,
>
> $$
> \frac{d\ell}{d\theta}
> =
> \frac{n}{\theta}-\sum_{i=1}^n x_i.
> $$
>
> Setting this to zero gives
>
> $$
> \hat\theta_{\text{MLE}}
> =
> \frac{n}{\sum_{i=1}^n x_i}
> =
> \frac1{\bar x}.
> $$
>
> Again, $d^2\ell/d\theta^2=-n/\theta^2<0$, so this is the maximum.
-->

## Optional Problem 3

> [!problem|Poisson MLE]
> Consider a dataset $\{x_i\}_{i=1}^n$ of non-negative integers following a Poisson distribution with probability mass function:
>
> $$p_\lambda(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x \in \{0,1,2,\ldots\}, \lambda > 0.$$
>
> **Exercise:**
>
> 1.  Write down the log-likelihood function $\ell(\lambda)$.
> 2.  Find the maximum likelihood estimator $\hat{\lambda}_{\text{MLE}}$.
> 3.  Prove that $\hat{\lambda}_{\text{MLE}}$ equals the sample mean of the observations.

<!--
> [!solution]- Solution
>
> The log-likelihood is
>
> $$
> \ell(\lambda)
> =
> \sum_{i=1}^n
> \left(x_i\log\lambda-\lambda-\log(x_i!)\right)
> =
> \left(\sum_{i=1}^n x_i\right)\log\lambda
> -
> n\lambda
> -
> \sum_{i=1}^n\log(x_i!).
> $$
>
> Differentiate:
>
> $$
> \frac{d\ell}{d\lambda}
> =
> \frac{\sum_i x_i}{\lambda}-n.
> $$
>
> Setting this to zero gives
>
> $$
> \hat\lambda_{\text{MLE}}
> =
> \frac1n\sum_{i=1}^n x_i
> =
> \bar x.
> $$
>
> The second derivative is
>
> $$
> \frac{d^2\ell}{d\lambda^2}
> =
> -\frac{\sum_i x_i}{\lambda^2}.
> $$
>
> If at least one observation is positive, this is strictly negative, so the critical point is the maximum. If all observations are zero, the likelihood is maximized at the boundary limit $\lambda\to0^+$.
-->
