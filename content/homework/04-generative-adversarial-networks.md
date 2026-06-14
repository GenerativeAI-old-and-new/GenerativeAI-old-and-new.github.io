---
title: "Homework 4: Generative Adversarial Networks"
description: "Homework problems for Module 4 covering GAN training signals, critics, gradient penalties, and WGAN-GP experiments."
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

> [!problem|WGAN-GP on MNIST and Beyond]
> Complete the provided [Colab notebook](https://drive.google.com/file/d/1yzTpfloEWz8uutfBHUAs97QYYkUTjERZ/view?usp=sharing). Submit the completed code, generated image grids, loss curves, and a short report.
>
> 1.  Train WGAN-GP on MNIST. Tune at least three hyperparameter settings involving learning rate, batch size, `n_critic`, `lambda_GP`, or network depth. For your best run, include generated digit samples and the critic/generator loss curves. Explain how the gradient penalty changes training compared with an otherwise similar run without the penalty.
> 2.  **Optional.** Repeat the experiment on another dataset, such as Fashion-MNIST, CIFAR-10, or a toy image dataset. Compare the samples and training curves against MNIST, and state which hyperparameter or architecture changes were needed.

## Theory

Unless stated otherwise, let $P^\star$ denote the data distribution. Let $\xi\sim\pi_0$ be the input noise, let $T_\theta$ be the generator network, and let

$$
X_\theta = T_\theta(\xi), \qquad X_\theta\sim P_\theta
$$

denote a generated sample and its distribution. In the original GAN, a discriminator outputs a probability-like score. In WGAN, a critic $h_\beta$ outputs a real-valued score. We use $\mathcal F$ to denote a class of scoring functions $f:\mathcal X\to\mathbb R$ used to compare real and generated samples.

> [!problem|Likelihood-Free Training Signal]
> A GAN generator is usually implemented as code that maps noise to a sample. Explain why this code can easily generate fake samples but usually cannot return the density value $p_\theta(x)$ for a given image or data point $x$. Then explain what training signal a GAN uses instead of evaluating $p_\theta(x)$.

> [!problem|Critic Scores vs. Probabilities]
> In WGAN, the critic is trained to give higher scores to real samples and lower scores to generated samples, under a Lipschitz constraint. Explain why its output is a real-valued score rather than a sigmoid probability. Compare this training signal to the binary-classification loss used by the discriminator in the original GAN.

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

> [!problem|Single-Neuron Discriminator]
> Let $\mathcal F=\{x\mapsto \sigma(a^\top x+b):\|a\|\le 1,\ |b|\le 1\}$, where $\sigma$ is a sigmoid activation. Describe the kind of boundary this one-neuron discriminator can draw. Then explain why a multi-layer discriminator can detect differences that this class cannot.

> [!problem|Moment Matching with Fixed Features]
> Suppose we define the loss
>
> $$
> L(\theta)=\left\|\mathbb E_{X\sim P^\star}[\phi(X)]
> -\mathbb E_{X_\theta\sim P_\theta}[\phi(X_\theta)]\right\|^2
> $$
>
> for some feature map $\phi$. State exactly which feature averages are being matched. If the loss is zero, does it mean the real and generated samples match in every possible way? Explain one concrete limitation of checking only a fixed set of features.

> [!problem|When Generated Samples Match the Data]
> Suppose the generator distribution matches the data distribution exactly, so $P_\theta=P^\star$. In the original GAN, what discriminator output should we expect? In WGAN, what value should the critic objective have? Could many different critic networks tie for the best value? Explain why, at this point, the generator should no longer receive a useful direction for making samples closer to the data.

> [!problem|Symmetric Scoring Rules]
> Give one example of a set of scoring functions where every score can be flipped from $f$ to $-f$, and one example where this is not true. Explain how this decides whether the absolute value in the IPM formula is needed.

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
