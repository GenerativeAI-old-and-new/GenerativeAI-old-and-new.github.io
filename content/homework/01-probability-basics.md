---
title: "Homework 1: Probability and MLE"
description: "Homework problems for Module 1 covering Gaussian identities, KL divergence, categorical MLE, and energy-based models."
publish: true
---

[Back to Module 1 notes](/modules/01-probability-basics)

## Problem 1

> [!problem|Gaussian Density Operations]
> Let $p_i(x)$ be the density of $\mathcal N(\mu_i,\sigma_i^2)$ for $i=1,2$ (with $\sigma_i>0$), defined on $\mathbb R$. Answer the following:
>
> 1.  Let
>
>     $$p(x)=\frac{p_1(x)p_2(x)}{Z},\qquad Z=\int_{\mathbb R} p_1(x)p_2(x)\,\mathrm dx.$$
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
> 1.  **Product of two Gaussians (re-normalized) is Gaussian.** Since $p_1,p_2\ge0$ and
>
>     $$\int_{\mathbb R}\frac{p_1(x)p_2(x)}{Z}\,\mathrm dx=\frac1Z\int_{\mathbb R}p_1(x)p_2(x)\,\mathrm dx=\frac ZZ=1,$$
>
>     $p(x)$ is a valid density.
>
>     Moreover, $p_1(x)p_2(x)$ is proportional to a Gaussian:
>
>     $$
>     p_1(x)p_2(x)\ \propto\ \exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right)
>     =\exp\!\left(-\frac{(x-m)^2}{2v}\right),
>     $$
>
>     Expand the exponent and group terms in $x$: $$\begin{aligned}
>     &-\frac{(x-\mu_1)^2}{2\sigma_1^2}-\frac{(x-\mu_2)^2}{2\sigma_2^2}\\
>     &\quad= -\frac12\!\left(\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}\right)x^2
>       +\left(\frac{\mu_1}{\sigma_1^2}+\frac{\mu_2}{\sigma_2^2}\right)x
>       -\frac12\!\left(\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}\right).
>     \end{aligned}$$ Let $\tau=\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}$ and $\eta=\frac{\mu_1}{\sigma_1^2}+\frac{\mu_2}{\sigma_2^2}$. Then $$-\frac{(x-\mu_1)^2}{2\sigma_1^2}-\frac{(x-\mu_2)^2}{2\sigma_2^2}
>     = -\frac{\tau}{2}\!\left(x^2-2\frac{\eta}{\tau}x\right)
>       -\frac12\!\left(\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}\right)
>     = -\frac{\tau}{2}\!\left(x-\frac{\eta}{\tau}\right)^{\!2}
>       -\frac12\!\left(\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}-\frac{\eta^2}{\tau}\right).$$ The $x$-dependent part is $-\frac{(x-m)^2}{2v}$ with $$v=\frac{1}{\tau}=\left(\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}\right)^{-1}
>     =\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2},
>     \qquad
>     m=\frac{\eta}{\tau}
>     =\frac{\mu_1/\sigma_1^2+\mu_2/\sigma_2^2}{1/\sigma_1^2+1/\sigma_2^2}
>     =\frac{\mu_1\sigma_2^2+\mu_2\sigma_1^2}{\sigma_1^2+\sigma_2^2}.$$ The remaining constant term does not depend on $x$ and is absorbed into the normalizing constant $Z$.
>
>     We can rewrite $$p_1(x)p_2(x)=\frac{1}{2\pi\sigma_1\sigma_2}\,
>     \exp\!\left\{-\frac12\!\left(\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}-\frac{\eta^2}{\tau}\right)\right\}
>     \exp\!\left(-\frac{(x-m)^2}{2v}\right).$$ Therefore $$\begin{aligned}
>     Z
>     &=\int_{\mathbb R} p_1(x)p_2(x)\,\mathrm dx\\
>     &=\frac{1}{2\pi\sigma_1\sigma_2}\,
>     \exp\!\left\{-\frac12\!\left(\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}-\frac{\eta^2}{\tau}\right)\right\}
>     \int_{\mathbb R}\exp\!\left(-\frac{(x-m)^2}{2v}\right)\, \mathrm dx\\
>     &=\frac{1}{2\pi\sigma_1\sigma_2}\,
>     \exp\!\left\{-\frac12\!\left(\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}-\frac{\eta^2}{\tau}\right)\right\}
>     \sqrt{2\pi v}.
>     \end{aligned}$$ Since $\sqrt{2\pi v}/(2\pi\sigma_1\sigma_2)=1/\sqrt{2\pi(\sigma_1^2+\sigma_2^2)}$ and, by a short algebra check, $$\frac{\mu_1^2}{\sigma_1^2}+\frac{\mu_2^2}{\sigma_2^2}-\frac{\eta^2}{\tau}
>     =\frac{(\mu_1-\mu_2)^2}{\sigma_1^2+\sigma_2^2},$$ we obtain the closed form $$Z=\frac{1}{\sqrt{2\pi(\sigma_1^2+\sigma_2^2)}}
>     \exp\!\left(-\frac{(\mu_1-\mu_2)^2}{2(\sigma_1^2+\sigma_2^2)}\right)
>     =\mathcal N\!\big(\mu_1;\mu_2,\sigma_1^2+\sigma_2^2\big).$$ $\square$
>
> 2.  **Mixture of two Gaussians.** Clearly $p(x)=\tfrac12 p_1(x)+\tfrac12 p_2(x)\ge0$ and
>
>     $$\int_{\mathbb R} p(x)\,\mathrm dx=\frac12\int p_1+\frac12\int p_2= \frac12\cdot1+\frac12\cdot1=1,$$
>
>     so it is a valid density. It is a two-component Gaussian mixture (GMM) with weights $(\tfrac12,\tfrac12)$:
>
>     $$p(x)=\frac12\,\mathcal N(x;\mu_1,\sigma_1^2)+\frac12\,\mathcal N(x;\mu_2,\sigma_2^2),$$
>
>     which is _not_ a single Gaussian unless $\mu_1=\mu_2$ and $\sigma_1^2=\sigma_2^2$.
>
> 3.  **Sum of independent Gaussians is Gaussian.**
>
>     If $X_1\sim\mathcal N(\mu_1,\sigma_1^2)$ and $X_2\sim\mathcal N(\mu_2,\sigma_2^2)$ are independent, then $$X=X_1+X_2 \sim \mathcal N\!\left(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2\right).$$
>
>     _Proof (via MGF)._ By independence, $$M_X(t)\;=\;\mathbb E\!\left[e^{t(X_1+X_2)}\right]
>     =\mathbb E[e^{tX_1}]\,\mathbb E[e^{tX_2}]
>     =M_{X_1}(t)\,M_{X_2}(t).$$ For a normal $Y\sim\mathcal N(\mu,\sigma^2)$, $M_Y(t)=\exp\!\big(\mu t+\tfrac12\sigma^2 t^2\big)$. Hence $$M_X(t)=\exp\!\Big((\mu_1+\mu_2)t+\tfrac12(\sigma_1^2+\sigma_2^2)t^2\Big),$$ which is the MGF of $\mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$. Therefore $X\sim\mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$. $\square$
>
>     _(Optional) Proof (via convolution and completing the square)._ Let $f_{X_i}$ denote the densities. Then $$\begin{aligned}
>     f_X(x)
>     &=\int_{\mathbb R} f_{X_1}(x-y)f_{X_2}(y)\,dy \\
>     &=\frac{1}{2\pi\sigma_1\sigma_2}\int_{\mathbb R}
>     \exp\!\left[-\frac{(x-y-\mu_1)^2}{2\sigma_1^2}-\frac{(y-\mu_2)^2}{2\sigma_2^2}\right]dy.
>     \end{aligned}$$ Write $(x-y-\mu_1)^2=(y-(x-\mu_1))^2$ and complete the square in $y$: $$-\frac{(y-(x-\mu_1))^2}{2\sigma_1^2}-\frac{(y-\mu_2)^2}{2\sigma_2^2}
>     = -\frac{\tau}{2}\Big(y-\tfrac{\eta}{\tau}\Big)^{\!2}
>     -\frac{(x-(\mu_1+\mu_2))^2}{2(\sigma_1^2+\sigma_2^2)},$$ where $\tau=\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}$ and $\eta=\frac{x-\mu_1}{\sigma_1^2}+\frac{\mu_2}{\sigma_2^2}$. Integrating the Gaussian in $y$ gives $\sqrt{2\pi/\tau}$, and a short algebra check yields $$f_X(x)=\frac{1}{\sqrt{2\pi(\sigma_1^2+\sigma_2^2)}}
>     \exp\!\left(-\frac{(x-(\mu_1+\mu_2))^2}{2(\sigma_1^2+\sigma_2^2)}\right),$$ i.e., $X\sim\mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$. $\square$
>
> 4.  **Square of a standard normal.** Let $X=Z^2$ with $Z\sim\mathcal N(0,1)$. The map $x=z^2$ has two preimages $z=\pm\sqrt{x}$ for $x>0$. By change of variables, $$f_X(x)=\phi(\sqrt{x})\frac{1}{2\sqrt{x}}+\phi(-\sqrt{x})\frac{1}{2\sqrt{x}}
>     =\frac{1}{\sqrt{2\pi}}e^{-x/2}\cdot\frac{1}{\sqrt{x}}
>     =\frac{1}{\sqrt{2\pi x}}\,e^{-x/2},\qquad x>0,$$
>
>     and $f_X(x)=0$ for $x\le0$. Therefore $X\sim\chi^2(1)$ with support $[0,\infty)$. $\square$

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
>     \Big]\; \mathrm dx.
>     $$
>
>     Here $\alpha \in [0,1]$, and $p(x)$ and $q(x)$ are the densities of $P$ and $Q$ respectively. Answer the following questions:
>     1.  Is this divergence a valid notion of discrepancy? Explain your reasoning.
>     2.  Under what conditions does this divergence reduce to the KL divergence (either $\operatorname{KL}(P \,\|\, Q)$ or $\operatorname{KL}(Q \,\|\, P)$)?

<!--
> [!solution]- Solution
>
> 1.  **Two-outcome case.** Using $\operatorname{KL}(P\|Q)=\sum_i P(i)\log\frac{P(i)}{Q(i)}$ and the conventions $0\log 0=0$ and $a\log\frac{a}{0}=+\infty$ for $a>0$, we have $$\operatorname{KL}(P \| Q)=0.5\log\frac{0.5}{1.0}\;+\;0.5\log\frac{0.5}{0}
>     = -\tfrac{1}{2}\log 2\;+\;(+\infty) = +\infty,$$ since $Q(2)=0$ while $P(2)>0$.
>
>     For the reverse, $$\operatorname{KL}(Q\|P)=1.0\log\frac{1.0}{0.5}\;+\;0\cdot\log\frac{0}{0.5}
>     =\log 2.$$
>
> 2.  **Three-outcome case.** Similarly, $$\operatorname{KL}(P\|Q)
>     =0.5\log\frac{0.5}{1.0}\;+\;0.5\log\frac{0.5}{0}\;+\;0\cdot\log\frac{0}{0}
>     =-\tfrac{1}{2}\log 2\;+\;(+\infty)\;+\;0
>     =+\infty,$$ because $Q(2)=0$ while $P(2)>0$. For the reverse, $$\operatorname{KL}(Q\|P)
>     =1.0\log\frac{1.0}{0.5}\;+\;0\cdot\log\frac{0}{0.5}\;+\;0\cdot\log\frac{0}{0}
>     =\log 2.$$
> 3.  1.  **Yes.** If $\operatorname{KL}(Q\|P)<\infty$, then necessarily $\operatorname{supp}(Q)\subseteq\operatorname{supp}(P)$. Proof by contrapositive: if there exists $x^\star$ with $Q(x^\star)>0$ but $P(x^\star)=0$, then the KL term at $x^\star$ equals $$Q(x^\star)\log\!\frac{Q(x^\star)}{P(x^\star)}
>         = Q(x^\star)\log\!\frac{Q(x^\star)}{0}
>         = +\infty,$$ so $\operatorname{KL}(Q\|P)=+\infty$, a contradiction. Hence $\operatorname{supp}(Q)\subseteq\operatorname{supp}(P)$.
>     2.  **No.** Finiteness of $\operatorname{KL}(Q\|P)$ does _not_ imply $\operatorname{supp}(P)\subseteq\operatorname{supp}(Q)$. Counterexample on $\Omega=\{1,2\}$: $$P=(0.5,\,0.5),\qquad Q=(1.0,\,0.0).$$ Then $\operatorname{supp}(P)=\{1,2\}$ while $\operatorname{supp}(Q)=\{1\}$, so $\operatorname{supp}(P)\nsubseteq\operatorname{supp}(Q)$; yet $$\operatorname{KL}(Q\|P)=1\cdot\log\frac{1}{0.5}+0\cdot\log\frac{0}{0.5}=\log 2<\infty.$$
> 4.  1.  **Validity (nonnegativity and identity of indiscernibles).** Write $t(x):= \frac{p(x)}{q(x)}$ on $\{q>0\}$ and note that for $\alpha\in[0,1]$, $$\alpha p(x)+(1-\alpha)q(x)=q(x)\big(\alpha t(x)+(1-\alpha)\big)\ge 0,$$ with strict positivity whenever $q(x)>0$. Since the scalar function $g(t)\!=\!t-\log t-1$ satisfies $g(t)\ge 0$ for all $t>0$ and $g(t)=0$ iff $t=1$, the integrand $$\big(\alpha p+(1-\alpha)q\big)\,g\!\left(\tfrac{p}{q}\right)
>         = q\big(\alpha t+(1-\alpha)\big)\,g(t)$$ is pointwise $\ge 0$ wherever it is defined; if $q=0<p$ and $\alpha>0$, then $t=+\infty$ and the integrand is $+\infty$, still $\ge 0$ by convention. Hence $\mathrm D(P,Q)\ge 0$ (possibly $+\infty$).
>
>         Moreover, $\mathrm D(P,Q)=0$ implies the integrand is $0$ almost everywhere. On $\{q>0\}$ we have $\alpha t+(1-\alpha)>0$, so $g(t)=0$ and thus $t=1$, i.e. $p=q$ a.e. on $\{q>0\}$. If $\alpha>0$, then $\{q=0\}$ also forces $p=0$ a.e. there (otherwise the integrand is $+\infty$), so $p=q$ a.e. globally. If $\alpha=0$, then $$\mathrm D(P,Q)=\int q\,g\!\left(\tfrac{p}{q}\right)\,dx=\operatorname{KL}(Q\|P),$$ which equals $0$ iff $P=Q$ a.e. Therefore $\mathrm D$ is a (possibly improper) divergence: nonnegative and $=0$ iff $P=Q$ a.e. (it is not symmetric and need not satisfy the triangle inequality).
>
>     2.  **When does $\mathrm D$ reduce to a KL?** Using $t=\frac{p}{q}$ and $\int p=\int q=1$, $$\mathrm D(P,Q)
>         =\alpha\!\int p\Big(t-\log t-1\Big)\,dx+(1-\alpha)\!\int q\Big(t-\log t-1\Big)\,dx.$$ Compute the two pieces: $$\int q\big(t-\log t-1\big)\,dx
>         =1-\!\int q\log t\,dx-1
>         =\operatorname{KL}(Q\|P),$$ $$\int p\big(t-\log t-1\big)\,dx
>         =\int \frac{p^2}{q}\,dx-\operatorname{KL}(P\|Q)-1
>         =\chi^2(P\|Q)-\operatorname{KL}(P\|Q),$$ where $\chi^2(P\|Q)=\int \frac{p^2}{q}\,dx-1$ is the Pearson $\chi^2$-divergence (defined when $P\ll Q$). Thus $$\boxed{\;
>         \mathrm D(P,Q)=(1-\alpha)\,\operatorname{KL}(Q\|P)\;+\;\alpha\big(\chi^2(P\|Q)-\operatorname{KL}(P\|Q)\big).
>         \;}$$ Consequently:
>         - For $\alpha=0$, $\mathrm D(P,Q)=\operatorname{KL}(Q\|P)$ (with the usual $+\infty$ if $Q\not\ll P$).
>         - For $\alpha=1$, $\mathrm D(P,Q)=\chi^2(P\|Q)-\operatorname{KL}(P\|Q)$, which equals $\operatorname{KL}(P\|Q)$ only in the trivial case $P=Q$ (both sides $0$); in general it is _not_ a KL.
>         - For any $\alpha\in(0,1)$, $\mathrm D$ is a mixture of $\operatorname{KL}(Q\|P)$ and $\chi^2(P\|Q)-\operatorname{KL}(P\|Q)$ and does not equal $\operatorname{KL}(P\|Q)$ nor $\operatorname{KL}(Q\|P)$ except when $P=Q$.

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
> 1.  **Log-likelihood.** Let $n_k=\sum_{i=1}^n \mathbf 1\{x_i=k\}$ with $\sum_{k=1}^K n_k=n$. Using the softmax formula above, $$\ell(\theta)
>     =\sum_{i=1}^n \log p_\theta(x_i)
>     =\sum_{i=1}^n \Big(\theta_{x_i}-\log\!\sum_{j=1}^K e^{\theta_j}\Big)
>     =\sum_{k=1}^K n_k\,\theta_k\;-\;n\,\log\!\sum_{j=1}^K e^{\theta_j}.$$
> 2.  **Gradient, MLE, and uniqueness.** The softmax probabilities are $s_k(\theta):= p_\theta(x=k)=\frac{e^{\theta_k}}{\sum_j e^{\theta_j}}$. Differentiate: $$\frac{\partial \ell}{\partial \theta_k}
>     = n_k - n\, s_k(\theta).$$ Setting the gradient to zero gives $$s_k(\hat\theta)=\frac{n_k}{n} =: \hat p_k,\qquad k=1,\dots,K.$$ Thus the MLE over probabilities is the empirical frequency $\hat p_k=n_k/n$.
>
>     _Mapping to $\hat\theta$._ If all $n_k>0$, any vector of the form $$\hat\theta_k=\log n_k + c\quad\text{(equivalently, }\log\hat p_k + c\text{)},\qquad c\in\mathbb R,$$ satisfies $s_k(\hat\theta)=\hat p_k$, since $\frac{e^{\log n_k + c}}{\sum_j e^{\log n_j + c}}=\frac{n_k}{\sum_j n_j}=\frac{n_k}{n}$. Hence a finite maximizer exists but is _not unique_ : $\theta$ is identifiable only up to adding a constant multiple of $\mathbf 1$.
>
>     If some $n_k=0$, then the maximizer lies on the boundary with $\hat p_k=0$ for those $k$. No finite $\theta$ yields $s_k(\theta)=0$; rather, the supremum is attained in the limit $\theta_k\to -\infty$ for all $k$ with $n_k=0$, while for $n_k>0$ we can take $\theta_k=\log n_k + c$. In summary: $$\hat\theta_k=
>     \begin{cases}
>     \log n_k + c, & n_k>0,\\[2pt]
>     -\infty, & n_k=0,
>     \end{cases}
>     \qquad c\in\mathbb R.$$
>
>     _Uniqueness._ The **parameter** MLE $\hat\theta_{\mathrm{MLE}}$ is not unique due to the additive-constant invariance (and, with zeros, because coordinates at $-\infty$ admit any common finite shift on the finite entries). The **distribution** $p_{\hat\theta_{\mathrm{MLE}}}$ _is_ unique: it is $\hat p_k=n_k/n$ (including $\hat p_k=0$ when $n_k=0$).
>
> 3.  **Numerical stability of the three ratios.** Recall IEEE-754 `float64` roughly overflows for $\exp(x)$ when $x\gtrsim 709$ and underflows to $0$ when $x\lesssim -745$.
>     1.  $\displaystyle \frac{e^{10000}}{e^{20000}+e^{10000}}$ is **unstable**. Directly, $e^{10000}$ and $e^{20000}$ both overflow to `Inf`, giving `Inf`/(`Inf`+`Inf`) $=$ `NaN`. Mathematically one should factor $e^{10000}$: $$\frac{e^{10000}}{e^{20000}+e^{10000}}=\frac{1}{e^{10000}+1}\approx 0,$$ but this stable algebra is not what naive evaluation does.
>     2.  $\displaystyle \frac{e^{-20000}}{e^{-10000}+e^{-20000}}$ is **unstable**. Here $e^{-10000}$ and $e^{-20000}$ both underflow to $0$, so we get $0/(0+0)$ $=$ `NaN`. Mathematically factor $e^{-20000}$: $$\frac{e^{-20000}}{e^{-10000}+e^{-20000}}=\frac{1}{e^{10000}+1}\approx 0,$$ but naive evaluation fails.
>     3.  $\displaystyle \frac{e^{-10000}}{e^{-10000}+e^{0}}$ is **stable**. $e^{-10000}$ underflows to $0$, so the computation becomes $0/(0+1)=0$, which matches the true value extremely well since $e^{-10000}\approx 0$ in double precision. There is underflow, but no harmful cancellation and the returned value is accurate.
> 4.  **Stable softmax and log-likelihood.** For $\theta=(\theta_1,\ldots,\theta_K)$, let $m=\max_j \theta_j$. Then use the "subtract-the-max" trick: $$\mathrm{softmax}_k(\theta)
>     =\frac{e^{\theta_k-m}}{\sum_{j=1}^K e^{\theta_j-m}},
>     \qquad
>     \log\!\sum_{j=1}^K e^{\theta_j}
>     = m + \log\!\sum_{j=1}^K e^{\theta_j-m}.$$ This prevents overflow (large positives) and avoids catastrophic underflow when only differences matter.
>
>     For a dataset with counts $n_k=\sum_{i=1}^n\mathbf 1\{x_i=k\}$, the (stable) log-likelihood is $$\ell(\theta)
>     =\sum_{k=1}^K n_k\,\theta_k \;-\; n\;\underbrace{\Big(m+\log\!\sum_{j=1}^K e^{\theta_j-m}\Big)}_{\text{logsumexp}(\theta)}.$$ Equivalently, per-sample: $$\log p_\theta(x_i=k)=\theta_k-\big(m+\log\!\sum_{j=1}^K e^{\theta_j-m}\big).$$

-->

## Problem 4

> [!problem|Energy-Based Models with Langevin Sampling]
> We want to fit a dataset $\mathcal{D} = \{x_i\}_{i=1}^n$ with an energy-based model on $\mathbb{R}^d$ of the form
>
> $$
> p_{\theta}(x)=\frac{\exp\big(f_\theta(x)\big)}{Z_\theta},
> \qquad
> Z_\theta=\int_{\mathbb{R}^d}\exp\big(f_\theta(x)\big)\,dx.
> $$
>
> Here $f_\theta(x)$ is the (unnormalized) log-density (i.e., negative energy), and $Z_\theta$ is the partition function. To ensure integrability, we use
>
> $$f_\theta(x)=\mathrm{NN}_w(x)-\frac{\|x-\mu\|^2}{2\sigma^2},$$
>
> with $\mathrm{NN}_w$ a neural network (e.g. MLP) parameterized by $w$, $\mu\in\mathbb{R}^d$, and $\sigma>0$ a _scalar_ so the Gaussian term is isotropic ($\sigma^2 I_d$). We write $\theta=(w,\mu,\sigma)$; for simplicity, treat $(\mu,\sigma)$ as fixed hyperparameters (e.g. $\mu=\mathbf{0}$, $\sigma=0.1$) unless you wish to tune them manually.
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
> **Set-up.** We model $p_\theta(x)\propto \exp(f_\theta(x))$ with $$f_\theta(x)=\mathrm{NN}_w(x)-\frac{\|x-\mu\|^2}{2\sigma^2},\qquad
> \nabla_x \log p_\theta(x)=\nabla_x f_\theta(x).$$
>
> #### (1) Sampling: Langevin vs. grid
>
> We adopt the Langevin sampling form $$x_{t+1}=x_t+\epsilon\,\nabla_x \log p_\theta(x_t)+\sqrt{2\epsilon}\,\xi_t,
> \qquad \xi_t\stackrel{\text{i.i.d.}}{\sim}\mathcal N(0,I_d),$$
>
> ```python
> def langevin_sampler(model, x0, num_steps=100, step_size=0.01, record_traj=False):
>     x = x0.detach().requires_grad_(True)
>     traj = [x.detach()]
>     for _ in range(num_steps):
>         grad_logp = torch.autograd.grad(model(x).sum(), x)[0]
>         noise = torch.randn_like(x)
>         x = x + step_size * grad_logp + torch.sqrt(x.new_tensor(2.0 * step_size)) * noise
>         x = x.detach().requires_grad_(True)
>         if record_traj:
>             traj.append(x.detach())
>     if record_traj:
>         return x.detach(), torch.stack(traj)
>     else:
>         return x.detach()
> ```
>
> #### (2) MLE training
>
> The average log-likelihood and its gradient are $$\ell(\theta)=\mathbb E_{x\sim \hat P_{\text{data}}}[f_\theta(x)]-\log Z_\theta,\quad
> \nabla_\theta \ell(\theta)=\mathbb E_{\text{data}}[\nabla_\theta f_\theta(x)]-\mathbb E_{p_\theta}[\nabla_\theta f_\theta(x)].$$ We _minimize_ $\mathcal L(\theta)=-\ell(\theta)$; for 2D we estimate $\log Z_\theta$ via a grid.
>
> Both are okay if we use `Langevin` samples or `logZ_est`.
>
> ```python
> data_energy_mean = energy_model(data_batch).mean()
>
> model_samples = langevin_sampler(
>     energy_model,
>     torch.randn((batch_size, input_dim)),
>     num_steps=500,
>     step_size=1e-3
> )
> model_energy_mean = energy_model(model_samples).mean()
>
> # For verification: Use Grid to approx log Z
> # logZ_est = energy_model.logZ_2D_grid()
> # model_energy_mean = logZ_est
>
> loss = -data_energy_mean + model_energy_mean
> ```

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
> **1) Log-Likelihood Function:**
>
> $$
> \begin{aligned}
> \ell(\theta) &= \sum_{i=1}^n \log p_\theta(x_i) \\
> &= \sum_{i=1}^n \log \left(\frac{\theta}{2} \exp(-\theta|x_i|)\right) \\
> &= \sum_{i=1}^n \left[\log \frac{\theta}{2} - \theta|x_i|\right] \\
> &= n \log \frac{\theta}{2} - \theta \sum_{i=1}^n |x_i|
>
> \end{aligned}
> $$
>
> **2) Maximum Likelihood Estimator:** Setting the derivative to zero:
>
> $$
> \begin{aligned}
> \frac{d\ell}{d\theta} &= \frac{n}{\theta} - \sum_{i=1}^n |x_i| = 0 \\
> \frac{n}{\theta} &= \sum_{i=1}^n |x_i| \\
> \hat{\theta}_{\text{MLE}} &= \frac{n}{\sum_{i=1}^n |x_i|}
>
> \end{aligned}
> $$
>
> **3) Simple Function Form:** Let $\bar{|x|} = \frac{1}{n}\sum_{i=1}^n |x_i|$ be the sample mean of absolute values. Then:
>
> $$
> \begin{aligned}
> \hat{\theta}_{\text{MLE}} = \frac{1}{\bar{|x|}}
>
> \end{aligned}
> $$

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
> **1) Log-Likelihood Function:**
>
> $$
> \begin{aligned}
> \ell(\theta) &= \sum_{i=1}^n \log p_\theta(x_i) \\
> &= \sum_{i=1}^n \log \left(\theta \exp(-\theta x_i)\right) \\
> &= \sum_{i=1}^n \left[\log \theta - \theta x_i\right] \\
> &= n \log \theta - \theta \sum_{i=1}^n x_i
>
> \end{aligned}
> $$
>
> **2) Maximum Likelihood Estimator:** Setting the derivative to zero:
>
> $$
> \begin{aligned}
> \frac{d\ell}{d\theta} &= \frac{n}{\theta} - \sum_{i=1}^n x_i = 0 \\
> \frac{n}{\theta} &= \sum_{i=1}^n x_i \\
> \hat{\theta}_{\text{MLE}} &= \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}
>
> \end{aligned}
> $$
>
> **3) Verification:** We have shown that $\hat{\theta}_{\text{MLE}} = \frac{1}{\bar{x}}$ where $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is the sample mean.

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
> **1) Log-Likelihood Function:**
>
> $$
> \begin{aligned}
> \ell(\lambda) &= \sum_{i=1}^n \log p_\lambda(x_i) \\
> &= \sum_{i=1}^n \log \left(\frac{\lambda^{x_i} e^{-\lambda}}{x_i!}\right) \\
> &= \sum_{i=1}^n \left[x_i \log \lambda - \lambda - \log(x_i!)\right] \\
> &= \log \lambda \sum_{i=1}^n x_i - n\lambda - \sum_{i=1}^n \log(x_i!)
>
> \end{aligned}
> $$
>
> **2) Maximum Likelihood Estimator:** Setting the derivative to zero:
>
> $$
> \begin{aligned}
> \frac{d\ell}{d\lambda} &= \frac{1}{\lambda} \sum_{i=1}^n x_i - n = 0 \\
> \frac{1}{\lambda} \sum_{i=1}^n x_i &= n \\
> \hat{\lambda}_{\text{MLE}} &= \frac{1}{n} \sum_{i=1}^n x_i = \bar{x}
>
> \end{aligned}
> $$
>
> **3) Proof that MLE equals Sample Mean:** We have shown that $\hat{\lambda}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i = \bar{x}$.
>
> **Verification:** The second derivative is:
>
> $$
> \begin{aligned}
> \frac{d^2\ell}{d\lambda^2} = -\frac{1}{\lambda^2} \sum_{i=1}^n x_i < 0
>
> \end{aligned}$$ for all $\lambda > 0$, confirming that this is indeed a maximum.
> $$

-->
