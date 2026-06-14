---
title: "Homework 5: Autoencoders and Variational Autoencoders"
description: "Homework problems covering autoencoders, VAE training, and beta-VAE experiments."
publish: true
---

[Back to Module 5 notes](/modules/05-autoencoder-models)

## Coding

> [!problem|VAE on MNIST]
> Complete the provided [Colab notebook](https://colab.research.google.com/drive/1gD3vBHzFvCNe6QWqjLgEdMEAJ2sH6JUr?usp=sharing). Your submission should include the completed loss implementation, training curves, reconstruction examples, prior samples, and short written answers to the prompts below.
>
> 1.  Implement the VAE loss
>
>     $$
>     \mathcal L_{\text{VAE}}(x)
>     =
>     \mathbb E_{q_\phi(z\mid x)}
>     \big[-\log p_\theta(x\mid z)\big]
>     +
>     \beta\,\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big).
>     $$
>
>     Fill in the reconstruction loss and the KL term in the notebook. Run a training job and report the total loss, reconstruction loss, and KL loss over training. Briefly comment on whether the total loss behaves like reconstruction loss plus $\beta$ times KL loss.
>
> 2.  Train the convolutional VAE on MNIST for several beta values using `train_conv_vae`. For each beta value, include a grid of reconstructions and a grid of samples generated from the prior. Compare reconstruction quality, sample quality, and the relative magnitudes of the reconstruction and KL losses. Conclude with the beta value you would choose and justify the choice.

## Theory

Unless stated otherwise, let $x\in\mathbb R^d$ be a data point and let $z\in\mathbb R^k$ be a latent code. We write $q^\phi(z\mid x)$ for the encoder distribution, $p^\theta(x\mid z)$ for the decoder likelihood, and $p(z)$ for the latent prior, usually $p(z)=\mathcal N(0,I_k)$. When the encoder is Gaussian, assume

$$
q^\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))),
$$

where $\mu_\phi(x)=(\mu_1,\ldots,\mu_k)\in\mathbb R^k$ and $\sigma_\phi(x)=(\sigma_1,\ldots,\sigma_k)\in\mathbb R_+^k$ are encoder outputs.

> [!problem|From Autoencoder to Generative Autoencoder]
> A deterministic autoencoder maps $x$ to $z=f_\phi(x)$ and reconstructs $x$ as $g_\theta(z)$. Explain why sampling $z\sim\mathcal N(0,I)$ and decoding $g_\theta(z)$ can produce unrealistic samples, even when training reconstruction error is small. Then explain one concrete way to enforce the encoder's latent codes to look more Gaussian, such as adding a penalty or adversarial loss that matches the aggregated latent codes to $\mathcal N(0,I)$.

> [!problem|Stochastic Encoder and Reparameterization]
> For the Gaussian encoder above, write the reparameterized sampling rule using $\epsilon\sim\mathcal N(0,I_k)$, where $\epsilon$ has the same shape as $z$. Explain why this version is friendly to automatic differentiation: the random draw is separated from the encoder network outputs, so gradients can flow through $\mu_\phi(x)$ and $\sigma_\phi(x)$.

> [!problem|Closed-Form Gaussian KL]
> For the Gaussian encoder above with prior $p(z)=\mathcal N(0,I_k)$, derive the closed-form expression for $\mathrm{KL}(q^\phi(z\mid x)\,\|\,p(z))$ as a sum over latent coordinates $i=1,\ldots,k$.

> [!problem|Effect of beta in beta-VAE]
> The beta-VAE objective can be written as
>
> $$
> \mathbb E_{q^\phi(z\mid x)}[-\log p^\theta(x\mid z)]
> +
> \beta\,\mathrm{KL}(q^\phi(z\mid x)\,\|\,p(z)).
> $$
>
> Explain the role of the reconstruction term and the KL term. As $\beta$ increases, describe what you expect to happen to reconstruction quality, how much information $z$ keeps about $x$, and sample quality when decoding $z\sim p(z)$.

> [!problem|Posterior Collapse Thought Experiment]
> Suppose the decoder is expressive enough to model the data distribution while using little information from $z$. Explain why the model might learn $q^\phi(z\mid x)\approx p(z)$ for most inputs. Describe how this failure mode would appear in the reconstruction loss, the KL loss, and samples decoded from the prior.

> [!problem|Matching Latent Codes to a Prior]
> In an adversarial autoencoder, run training examples through the encoder and collect the resulting latent codes. We want this batch of codes to look like samples from a prior $p(z)$, such as $\mathcal N(0,I)$. Write the two losses used in training: the reconstruction loss for the autoencoder and the adversarial loss that tries to make encoder outputs indistinguishable from prior samples. Specify which network parameters are updated by each loss.

> [!problem|Do the Latent Codes Match the Prior?]
> In a standard VAE, the KL penalty pushes each encoded distribution $q^\phi(z\mid x)$ toward $p(z)$. If we encode many training examples and pool all sampled latent codes together, does that pooled code distribution
>
> $$
> q^\phi(z)=\int q^\phi(z\mid x)p_{\text{data}}(x)\,\mathrm dx
> $$
>
> have to equal $p(z)$ exactly? Explain why or why not, using either a simple example or an intuitive argument. A formal proof is not required.

> [!problem|Different Latent Codes, Same Reconstruction]
> Suppose an autoencoder reconstructs the training data almost perfectly. Explain why this only tells us that the decoder works well on codes produced by the encoder; it does not mean those codes look like samples from $\mathcal N(0,I)$. Give a simple coordinate change, such as multiplying every latent code by a constant and compensating inside the decoder, that keeps all reconstructions the same but changes the scale or shape of the latent-code cloud.

> [!problem|Deriving the VAE Objective]
> Starting from
>
> $$
> \log p^\theta(x)
> =
> \log\int p^\theta(x,z)\,\mathrm dz,
> $$
>
> use $q^\phi(z\mid x)$ and Jensen's inequality to obtain the ELBO used for VAE training. In your final expression, label the reconstruction term and the KL penalty to the prior. Also state the extra KL term, $\mathrm{KL}(q^\phi(z\mid x)\,\|\,p^\theta(z\mid x))$, that explains when the ELBO is tight.

> [!problem|Linear Autoencoder as PCA]
> Assume the training data are centered. Consider a linear autoencoder with no bias terms, squared reconstruction loss, and bottleneck dimension $k<d$. Explain why the best $k$-dimensional reconstruction space is the span of the top-$k$ principal components of the data. Then give one concrete example where the encoder weights and decoder weights change, but every reconstructed output stays the same. For example, you may rotate or rescale the latent coordinates and undo that change in the decoder.
