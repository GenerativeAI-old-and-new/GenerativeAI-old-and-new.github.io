---
title: "Homework 4: Generative Adversarial Networks"
description: "Classical GANs, Wasserstein critics, gradient penalties, and WGAN-GP experiments."
publish: true
---

[Back to Module 4 notes](/modules/04-generative-adversarial-networks)

## Coding

> [!problem|GANs on 2D Toy Datasets]
> Complete the provided [Colab notebook](https://drive.google.com/file/d/1CGxibklD0-PjlJrNQsevGGIF5JMnra2c/view?usp=sharing) and answer the questions below. Submit the completed code, generated samples, and a short report comparing training behavior across settings.
>
> 1.  Train a **classic GAN** on the `rings` and `2spirals` toy datasets. Try at least three hyperparameter settings that vary learning rate, network width/depth, optimizer, or the number of discriminator updates per generator update. For each dataset, include sample plots from representative successful and failed runs, and identify signs of convergence or mode collapse.
> 2.  Implement a one-sided variance gradient penalty for WGAN:
>
>     $$
>     \operatorname{GP}(D)
>     =
>     \mathbb E_{\hat x}
>     \left[
>     \left(
>     \max\left(\|\nabla_{\hat x}D(\hat x)\|_2^2-1,\;0\right)
>     \right)^2
>     \right].
>     $$
>
>     This penalty should apply only when the squared gradient norm is larger than one. In your write-up, describe how you sample $\hat x$, how you compute $\nabla_{\hat x}D(\hat x)$ in code, and where this term enters the critic loss.
>
> 3.  Train WGAN with and without the gradient penalty. Compare the two versions on both toy datasets using at least two values of `lambda_GP` and at least two values of `n_critic`. Report the best configuration you found, show sample plots, and summarize how the penalty affected stability, loss curves, sample quality, and sample diversity.

> [!solution]- Solution
>
> Using binary cross-entropy with logits, the discriminator and generator losses are
>
> ```python
> real_logits = discriminator(x_real)
> fake = generator(z)
> fake_logits = discriminator(fake.detach())
>
> d_loss = (
>     F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
>     + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
> )
>
> g_logits = discriminator(fake)
> g_loss = F.binary_cross_entropy_with_logits(g_logits, torch.ones_like(g_logits))
> ```
>
> The `detach()` in the discriminator update prevents that loss from changing the generator. It must not be used in the generator update.
>
> The one-sided penalty in the question can be implemented as follows:
>
> ```python
> def gradient_penalty(critic, x_real, x_fake):
>     batch = x_real.size(0)
>     alpha = torch.rand(batch, 1, device=x_real.device)
>     x_hat = alpha * x_real + (1 - alpha) * x_fake.detach()
>     x_hat.requires_grad_(True)
>
>     score = critic(x_hat)
>     grad = torch.autograd.grad(
>         outputs=score.sum(), inputs=x_hat, create_graph=True
>     )[0]
>     grad_norm_sq = grad.flatten(1).square().sum(dim=1)
>     return F.relu(grad_norm_sq - 1).square().mean()
> ```
>
> With a critic that assigns larger values to real data, the WGAN losses are
>
> ```python
> critic_loss = critic(x_fake.detach()).mean() - critic(x_real).mean()
> critic_loss = critic_loss + lambda_GP * gradient_penalty(critic, x_real, x_fake)
> generator_loss = -critic(x_fake).mean()
> ```
>
> A successful run should place samples around the full ring or along both spiral arms. A dense cluster on only part of the target indicates mode collapse. In typical runs, the gradient penalty reduces sharp changes in the critic and makes training less sensitive, but a very large `lambda_GP` can make the critic too weak. The best setting should be chosen from the submitted plots rather than from the loss value alone.

> [!problem|WGAN-GP on MNIST and Beyond]
> Complete the provided [Colab notebook](https://drive.google.com/file/d/1yzTpfloEWz8uutfBHUAs97QYYkUTjERZ/view?usp=sharing). Submit the completed code, generated image grids, loss curves, and a short report.
>
> 1.  Train WGAN-GP on MNIST. Tune at least three hyperparameter settings involving learning rate, batch size, `n_critic`, `lambda_GP`, or network depth. For your best run, include generated digit samples and the critic/generator loss curves. Explain how the gradient penalty changes training compared with an otherwise similar run without the penalty.
> 2.  **Optional.** Repeat the experiment on another dataset, such as Fashion-MNIST, CIFAR-10, or a toy image dataset. Compare the samples and training curves against MNIST, and state which hyperparameter or architecture changes were needed.

> [!solution]- Solution
>
> The Wasserstein terms in the critic and generator losses are unchanged. For images, the interpolation coefficient must broadcast across channels and pixels:
>
> ```python
> batch = x_real.size(0)
> alpha = torch.rand(batch, 1, 1, 1, device=x_real.device)
> x_hat = alpha * x_real + (1 - alpha) * x_fake.detach()
> x_hat.requires_grad_(True)
> ```
>
> For the standard two-sided WGAN-GP penalty, flatten the gradient only when computing its norm:
>
> ```python
> grad = torch.autograd.grad(
>     critic(x_hat).sum(), x_hat, create_graph=True
> )[0]
> grad_norm = grad.flatten(1).norm(2, dim=1)
> gp = (grad_norm - 1).square().mean()
> ```
>
> One MNIST starting point is Adam with learning rate $10^{-4}$, `n_critic=5`, and `lambda_GP=10`. Without the penalty, the critic can develop very steep gradients and the generated digits may change abruptly or collapse to a few shapes. With a well-tuned penalty, training is usually smoother and the image grid contains a broader set of readable digits. Loss curves should be interpreted together with samples: GAN loss values do not directly measure image quality.

## Theory

Unless stated otherwise, let $P^\star$ denote the data distribution. Let $\xi\sim\pi_0$ be the input noise, let $T_\theta$ be the generator network, and let

$$
X_\theta = T_\theta(\xi), \qquad X_\theta\sim P_\theta
$$

denote a generated sample and its distribution. In the original GAN, a discriminator outputs a probability-like score. In WGAN, a critic $h_\beta$ outputs a real-valued score. We use $\mathcal F$ to denote a class of scoring functions $f:\mathcal X\to\mathbb R$ used to compare real and generated samples.

> [!problem|Likelihood-Free Training Signal]
> A GAN generator is usually implemented as code that maps noise to a sample. Explain why this code can easily generate fake samples but usually cannot return the density value $p_\theta(x)$ for a given image or data point $x$. Then explain what training signal a GAN uses instead of evaluating $p_\theta(x)$.

> [!solution]- Solution
>
> Sampling only requires drawing $\xi\sim\pi_0$ and running $x=T_\theta(\xi)$. Evaluating $p_\theta(x)$ is harder: a usual change-of-variables density requires an invertible map and a tractable Jacobian determinant. A GAN generator generally satisfies neither condition. Several latent codes may map to the same sample, and its outputs may lie on a lower-dimensional subset of image space.
>
> GAN training therefore compares samples. The discriminator or critic learns a score that separates real data from generated data, and the generator receives gradients through that score. No value of $p_\theta(x)$ is evaluated.

> [!problem|Critic Scores vs. Probabilities]
> In WGAN, the critic is trained to give higher scores to real samples and lower scores to generated samples, under a Lipschitz constraint. Explain why its output is a real-valued score rather than a sigmoid probability. Compare this training signal to the binary-classification loss used by the discriminator in the original GAN.

> [!solution]- Solution
>
> The WGAN objective uses a difference of average scores,
>
> $$
> \mathbb E_{P^\star}[h_\beta(X)]
> -
> \mathbb E_{P_\theta}[h_\beta(X)].
> $$
>
> The critic output is therefore not required to lie in $[0,1]$. The Lipschitz constraint controls how quickly the score can change with the input; no sigmoid is needed.
>
> The original GAN trains a binary classifier with cross-entropy: real samples have label $1$ and generated samples have label $0$. WGAN instead trains a constrained scoring function to increase the gap between the two average scores. The generator raises the critic score of its own samples.

> [!problem|Gradient Penalty on Interpolated Samples]
> In WGAN-GP, the gradient penalty is often evaluated on interpolated points
>
> $$
> \hat X = U X_{\text{data}} + (1-U)X_\theta,
> \qquad U\sim\operatorname{Uniform}[0,1].
> $$
>
> Explain why the penalty is evaluated on points between real and generated samples rather than only on real samples or only on generated samples. Use the figure below to describe what kind of critic behavior would produce a large penalty.
>
> <img src="/assets/homework/04-generative-adversarial-networks/gradient-penalty.png" alt="Gradient penalty illustration" width="520">

> [!solution]- Solution
>
> The critic supplies gradients in the region that separates generated samples from real samples. Interpolating between paired real and generated points places the penalty directly in this region, including locations where neither distribution currently has much mass.
>
> Penalizing only real points would leave the critic unrestricted immediately away from the data; penalizing only generated points has the same problem on the other side. The standard two-sided penalty is large when $\|\nabla_{\hat x}h_\beta(\hat x)\|_2$ is far from $1$. The one-sided version in the coding problem applies only when the norm is too large.

> [!problem|Mean Matching in 1D]
> Suppose $P^\star=\mathcal N(2,1)$ and $P_\theta=\mathcal N(\theta,1)$. Consider the loss
>
> $$
> L(\theta)=\left(\mathbb E_{P^\star}[X]
> -
> \mathbb E_{P_\theta}[X]\right)^2.
> $$
>
> Compute $L(\theta)$ explicitly and find the value of $\theta$ that minimizes it. Then run the gradient descent code below, plot or print the iterates, and check that $\theta$ moves toward the value you computed.
>
> ```python
> theta = 0.0
> lr = 0.1
>
> for t in range(50):
>     grad = 2 * (theta - 2)
>     theta -= lr * grad
> ```

> [!solution]- Solution
>
> The two means are $2$ and $\theta$, so
>
> $$
> L(\theta)=(2-\theta)^2.
> $$
>
> This is minimized at $\theta=2$. With learning rate $0.1$, the update is
>
> $$
> \theta_{t+1}
> =\theta_t-0.2(\theta_t-2)
> =0.8\theta_t+0.4.
> $$
>
> Starting from $\theta_0=0$, the iterates satisfy $\theta_t=2(1-0.8^t)$ and converge to $2$.

> [!problem|Discriminator Capacity]
> Think of each $f\in\mathcal F$ as a scoring rule that gives higher values to samples that look more real. We compare real and generated samples using
>
> $$
> \sup_{f\in\mathcal F}
> \left[
> \mathbb E_{X\sim P^\star} f(X)
> -
> \mathbb E_{X_\theta\sim P_\theta} f(X_\theta)
> \right].
> $$
>
> Explain why giving the discriminator a richer neural network can reveal differences that a very small model might miss. Give one concrete example, such as two datasets with the same mean but different shapes.

> [!solution]- Solution
>
> A small function class compares only the statistics it can represent. For example, a linear score $f(x)=a^\top x+b$ can detect a difference in means, but nothing more:
>
> $$
> \mathbb E_Pf(X)-\mathbb E_Qf(X)
> =a^\top(\mathbb E_PX-\mathbb E_QX).
> $$
>
> Consider $P=\mathcal N(0,1)$ and $Q$ that assigns probability $1/2$ to each of $-1$ and $1$. They have the same mean and variance but very different shapes. Linear scores cannot separate them by expectation, while a network with nonlinear features can respond differently to mass near zero and mass near $\pm1$.

> [!problem|Single-Neuron Discriminator]
> Let $\mathcal F=\{x\mapsto \sigma(a^\top x+b):\|a\|\le 1,\ |b|\le 1\}$, where $\sigma$ is a sigmoid activation. Describe the kind of boundary this one-neuron discriminator can draw. Then explain why a multi-layer discriminator can detect differences that this class cannot.

> [!solution]- Solution
>
> The score changes across the hyperplane
>
> $$
> a^\top x+b=0.
> $$
>
> The sigmoid makes this a soft linear boundary: points on one side receive lower scores and points on the other receive higher scores. Changing $a$ rotates the boundary and changing $b$ shifts it, but a single neuron still provides only one hyperplane.
>
> A multi-layer network can combine many such features to form curved or disconnected decision regions. It can therefore detect multimodal or local shape differences that cannot be separated by one linear boundary.

> [!problem|Moment Matching with Fixed Features]
> Suppose we define the loss
>
> $$
> L(\theta)=\left\|\mathbb E_{X\sim P^\star}[\phi(X)]
> -\mathbb E_{X_\theta\sim P_\theta}[\phi(X_\theta)]\right\|^2
> $$
>
> for some feature map $\phi$. State exactly which feature averages are being matched. If the loss is zero, does it mean the real and generated samples match in every possible way? Explain one concrete limitation of checking only a fixed set of features.

> [!solution]- Solution
>
> If $\phi(x)=(\phi_1(x),\ldots,\phi_m(x))$, then $L=0$ means
>
> $$
> \mathbb E_{P^\star}[\phi_j(X)]
> =
> \mathbb E_{P_\theta}[\phi_j(X)]
> \qquad\text{for every }j=1,\ldots,m.
> $$
>
> It does not generally imply that the distributions are equal. For example, with $\phi(x)=(x,x^2)$, a standard Gaussian and a random variable taking values $-1$ and $1$ equally often both give feature mean $(0,1)$. Their distributions are still different. Fixed features can only detect differences represented by those features.

> [!problem|When Generated Samples Match the Data]
> Suppose the generator distribution matches the data distribution exactly, so $P_\theta=P^\star$. In the original GAN, what discriminator output should we expect? In WGAN, what value should the critic objective have? Could many different critic networks tie for the best value? Explain why, at this point, the generator should no longer receive a useful direction for making samples closer to the data.

> [!solution]- Solution
>
> In the original GAN with equal real and fake sampling rates, the optimal discriminator is
>
> $$
> D^*(x)=\frac12.
> $$
>
> In WGAN, every admissible critic has
>
> $$
> \mathbb E_{P^\star}h(X)-\mathbb E_{P_\theta}h(X)=0
> $$
>
> when the two distributions are equal. The optimal critic value is therefore $0$, and many critic networks can tie. At this point there is no distribution mismatch left for the critic to expose, so the generator receives no useful direction for moving its distribution closer to the data.

> [!problem|Symmetric Scoring Rules]
> Give one example of a set of scoring functions where every score can be flipped from $f$ to $-f$, and one example where this is not true. Explain how this decides whether the absolute value in the IPM formula is needed.

> [!solution]- Solution
>
> The class of all $1$-Lipschitz real-valued functions is symmetric: if $f$ is $1$-Lipschitz, then $-f$ is also $1$-Lipschitz. A class of sigmoid outputs $f:\mathcal X\to[0,1]$ is not symmetric because $-f$ is usually not in the class.
>
> In a symmetric class, any negative expectation difference can be flipped into a positive one of the same magnitude, so the absolute value is unnecessary. Without symmetry, the flipped function may be unavailable, and the absolute value changes the quantity.

> [!problem|Removing the Absolute Value]
> The IPM compares two distributions by asking for the largest difference in average score:
>
> $$
> d_{\mathcal F}(P,Q)
> =
> \sup_{f\in\mathcal F}
> \left|\mathbb E_P f(X)-\mathbb E_Q f(X)\right|
> $$
>
> If $\mathcal F$ contains $f$ and also contains the flipped scoring rule $-f$, check algebraically that
>
> $$
> \sup_{f\in\mathcal F}
> \left[\mathbb E_P f-\mathbb E_Q f\right]
> =
> d_{\mathcal F}(P,Q).
> $$

> [!solution]- Solution
>
> Define
>
> $$
> \Delta(f)=\mathbb E_Pf-\mathbb E_Qf.
> $$
>
> Symmetry gives $-f\in\mathcal F$ and $\Delta(-f)=-\Delta(f)$. For any $f$, either $\Delta(f)\ge0$, in which case $|\Delta(f)|=\Delta(f)$, or $\Delta(f)<0$, in which case
>
> $$
> |\Delta(f)|=-\Delta(f)=\Delta(-f).
> $$
>
> Thus every absolute difference is attained as a signed difference by either $f$ or $-f$, so the left-hand supremum is no larger than the right-hand one. Conversely, $\Delta(f)\le |\Delta(f)|$ for every $f$. Therefore
>
> $$
> \sup_{f\in\mathcal F}|\Delta(f)|
> =
> \sup_{f\in\mathcal F}\Delta(f).
> $$
