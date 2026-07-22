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
>     \mathbb E_{q^\phi(z\mid x)}
>     \big[-\log p^\theta(x\mid z)\big]
>     +
>     \beta\,\mathrm{KL}\big(q^\phi(z\mid x)\,\|\,p(z)\big).
>     $$
>
>     Fill in the reconstruction loss and the KL term in the notebook. Run a training job and report the total loss, reconstruction loss, and KL loss over training. Briefly comment on whether the total loss behaves like reconstruction loss plus $\beta$ times KL loss.
>
> 2.  Train the convolutional VAE on MNIST for several beta values using `train_conv_vae`. For each beta value, include a grid of reconstructions and a grid of samples generated from the prior. Compare reconstruction quality, sample quality, and the relative magnitudes of the reconstruction and KL losses. Conclude with the beta value you would choose and justify the choice.

> [!solution]- Solution
>
> If the decoder returns image logits and the encoder returns `mu` and `logvar`, the loss can be implemented as
>
> ```python
> def vae_loss(x, recon_logits, mu, logvar, beta=1.0):
>     recon = F.binary_cross_entropy_with_logits(
>         recon_logits, x, reduction="sum"
>     )
>     kl = -0.5 * torch.sum(
>         1 + logvar - mu.square() - logvar.exp()
>     )
>     total = recon + beta * kl
>     return total, recon, kl
> ```
>
> Here `logvar` represents $\log\sigma^2$. If the training loop reports losses per example, divide all three returned values by the same batch size. The logged quantities should satisfy
>
> $$
> L_{\mathrm{total}}
> =L_{\mathrm{recon}}+\beta L_{\mathrm{KL}}
> $$
>
> up to the averaging convention used by the notebook.
>
> For a small $\beta$, the encoder has more freedom to store details about each digit. Reconstructions are usually strong, but samples from $z\sim\mathcal N(0,I)$ can be unreliable because the encoded latent distribution is weakly regularized. A moderate $\beta$ usually gives a better balance between reconstruction and prior sampling. If $\beta$ is too large, the KL term dominates, the code carries less information about the input, and reconstructions can lose digit-specific details. The preferred $\beta$ should be justified using both reconstruction and prior-sample grids; the smallest total loss across different $\beta$ values is not a fair comparison because the objectives have different weights.

## Theory

Unless stated otherwise, let $x\in\mathbb R^d$ be a data point and let $z\in\mathbb R^k$ be a latent code. We write $q^\phi(z\mid x)$ for the encoder distribution, $p^\theta(x\mid z)$ for the decoder likelihood, and $p(z)$ for the latent prior, usually $p(z)=\mathcal N(0,I_k)$. When the encoder is Gaussian, assume

$$
q^\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))),
$$

where $\mu_\phi(x)=(\mu_1,\ldots,\mu_k)\in\mathbb R^k$ and $\sigma_\phi(x)=(\sigma_1,\ldots,\sigma_k)\in\mathbb R_+^k$ are encoder outputs.

> [!problem|From Autoencoder to Generative Autoencoder]
> A deterministic autoencoder maps $x$ to $z=f_\phi(x)$ and reconstructs $x$ as $g_\theta(z)$. Explain why sampling $z\sim\mathcal N(0,I)$ and decoding $g_\theta(z)$ can produce unrealistic samples, even when training reconstruction error is small. Then explain one concrete way to enforce the encoder's latent codes to look more Gaussian, such as adding a penalty or adversarial loss that matches the aggregated latent codes to $\mathcal N(0,I)$.

> [!solution]- Solution
>
> Reconstruction training only evaluates the decoder on codes of the form $f_\phi(x)$. These codes may form separated clusters, have a different scale, or leave large unused regions in latent space. A random draw from $\mathcal N(0,I)$ can then land in a region the decoder never saw during training, even when $g_\theta(f_\phi(x))$ reconstructs every training input well.
>
> One fix is to match the distribution of encoded training points $f_\phi(x)$ to $\mathcal N(0,I)$. An adversarial autoencoder uses a latent-space discriminator for this: the discriminator separates encoded points from Gaussian samples, while the encoder learns to fool it. Prior draws then resemble the codes used to train the decoder.

> [!problem|Stochastic Encoder and Reparameterization]
> For the Gaussian encoder above, write the reparameterized sampling rule using $\epsilon\sim\mathcal N(0,I_k)$, where $\epsilon$ has the same shape as $z$. Explain why this version is friendly to automatic differentiation: the random draw is separated from the encoder network outputs, so gradients can flow through $\mu_\phi(x)$ and $\sigma_\phi(x)$.

> [!solution]- Solution
>
> Draw
>
> $$
> \epsilon\sim\mathcal N(0,I_k),
> \qquad
> z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
> $$
>
> This gives
>
> $$
> z\mid x
> \sim
> \mathcal N\!\left(
> \mu_\phi(x),
> \operatorname{diag}(\sigma_\phi^2(x))
> \right).
> $$
>
> The sampled noise $\epsilon$ does not depend on $\phi$. Once it is drawn, $z$ is a differentiable function of the encoder outputs, so ordinary backpropagation computes gradients through both $\mu_\phi(x)$ and $\sigma_\phi(x)$. In code, when the network outputs `logvar`, use
>
> ```python
> std = torch.exp(0.5 * logvar)
> z = mu + std * torch.randn_like(std)
> ```

> [!problem|Closed-Form Gaussian KL]
> For the Gaussian encoder above with prior $p(z)=\mathcal N(0,I_k)$, derive the closed-form expression for $\mathrm{KL}(q^\phi(z\mid x)\,\|\,p(z))$ as a sum over latent coordinates $i=1,\ldots,k$.

> [!solution]- Solution
>
> For Gaussians $\mathcal N(\mu,\Sigma)$ and $\mathcal N(0,I_k)$,
>
> $$
> \mathrm{KL}
> =
> \frac12\left(
> \operatorname{tr}(\Sigma)
> +\mu^\top\mu
> -k
> -\log\det\Sigma
> \right).
> $$
>
> Here $\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_k^2)$. Therefore
>
> $$
> \mathrm{KL}\big(q^\phi(z\mid x)\,\|\,p(z)\big)
> =
> \frac12\sum_{i=1}^k
> \left(
> \mu_i^2+\sigma_i^2-1-\log\sigma_i^2
> \right)
> $$
>
> Equivalently, because $\log\sigma_i^2=2\log\sigma_i$, the last term can be written as $-2\log\sigma_i$.

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

> [!solution]- Solution
>
> The reconstruction term rewards latent codes that let the decoder explain the input. The KL term keeps each encoder distribution close to the prior, making prior samples more similar to the latent values seen during training.
>
> Increasing $\beta$ makes information in $z$ more expensive. The encoder generally keeps less input-specific information, so reconstruction error rises. Moving from a very small $\beta$ to a moderate one can improve samples from $p(z)$ because the latent distribution better matches the prior. If $\beta$ becomes too large, the decoder may receive too little useful information from $z$, and both reconstructions and sample quality can deteriorate. Thus prior matching and reconstruction must be judged together.

> [!problem|Posterior Collapse Thought Experiment]
> Suppose the decoder is expressive enough to model the data distribution while using little information from $z$. Explain why the model might learn $q^\phi(z\mid x)\approx p(z)$ for most inputs. Describe how this failure mode would appear in the reconstruction loss, the KL loss, and samples decoded from the prior.

> [!solution]- Solution
>
> If the decoder can obtain a low reconstruction or likelihood loss without using $z$, then changing $q^\phi(z\mid x)$ has little effect on the first term. The encoder can reduce the remaining KL cost by setting
>
> $$
> q^\phi(z\mid x)\approx p(z)
> $$
>
> for every input. The latent code then carries little information about $x$.
>
> The main sign is a KL loss close to zero. The reconstruction loss may remain low if the decoder is strong enough, and prior samples may still look reasonable. The problem is that changing $z$ has little effect, so the latent code no longer represents useful information about the input.

> [!problem|Matching Latent Codes to a Prior]
> In an adversarial autoencoder, run training examples through the encoder and collect the resulting latent codes. We want this batch of codes to look like samples from a prior $p(z)$, such as $\mathcal N(0,I)$. Write the two losses used in training: the reconstruction loss for the autoencoder and the adversarial loss that tries to make encoder outputs indistinguishable from prior samples. Specify which network parameters are updated by each loss.

> [!solution]- Solution
>
> Let $E_\phi$ be the encoder, $G_\theta$ the decoder, and $D_\psi(z)$ a discriminator that predicts whether a latent code came from the prior. A squared-error reconstruction loss is
>
> $$
> L_{\mathrm{rec}}(\theta,\phi)
> =
> \mathbb E_{x\sim p_{\mathrm{data}}}
> \left[\|x-G_\theta(E_\phi(x))\|_2^2\right].
> $$
>
> The adversarial part alternates two updates. Train the latent discriminator with
>
> $$
> L_D(\psi)
> =
> -\mathbb E_{z\sim p(z)}\log D_\psi(z)
> -\mathbb E_x\log\big(1-D_\psi(E_\phi(x))\big).
> $$
>
> The encoder's adversarial loss is
>
> $$
> L_{\mathrm{adv}}(\phi)
> =
> -\mathbb E_x\log D_\psi(E_\phi(x)).
> $$
>
> Update $(\theta,\phi)$ with $L_{\mathrm{rec}}$, update $\psi$ with $L_D$ while treating encoded codes as detached samples, and update $\phi$ with $L_{\mathrm{adv}}$ while holding $\psi$ fixed. The decoder $G_\theta$ is not changed by the latent adversarial loss.

> [!problem|Do the Latent Codes Match the Prior?]
> In a standard VAE, the KL penalty pushes each encoded distribution $q^\phi(z\mid x)$ toward $p(z)$. If we encode many training examples and pool all sampled latent codes together, does that pooled code distribution
>
> $$
> q^\phi(z)=\int q^\phi(z\mid x)p_{\text{data}}(x)\,\mathrm d x
> $$
>
> have to equal $p(z)$ exactly? Explain why or why not, using either a simple example or an intuitive argument. A formal proof is not required.

> [!solution]- Solution
>
> No. The KL term is a soft penalty balanced against reconstruction, so the optimized conditional distributions are usually close to the prior rather than exactly equal to it. Pooling them produces a mixture,
>
> $$
> q^\phi(z)=\mathbb E_{x\sim p_{\mathrm{data}}}q^\phi(z\mid x),
> $$
>
> and a mixture of nearly Gaussian conditionals need not be a standard Gaussian. For example, if one group of inputs has posterior mean $+a$ and another has mean $-a$, the pooled distribution is a two-component Gaussian mixture. It may be symmetric with mean zero while still differing from $\mathcal N(0,I)$ in shape.
>
> Exact equality would hold in the stronger case $q^\phi(z\mid x)=p(z)$ for every $x$, but then $z$ would contain no information about the input.

> [!problem|Different Latent Codes, Same Reconstruction]
> Suppose an autoencoder reconstructs the training data almost perfectly. Explain why this only tells us that the decoder works well on codes produced by the encoder; it does not mean those codes look like samples from $\mathcal N(0,I)$. Give a simple coordinate change, such as multiplying every latent code by a constant and compensating inside the decoder, that keeps all reconstructions the same but changes the scale or shape of the latent-code cloud.

> [!solution]- Solution
>
> Let the original encoder and decoder be $f$ and $g$. Define
>
> $$
> \widetilde f(x)=2f(x),
> \qquad
> \widetilde g(z)=g(z/2).
> $$
>
> Then
>
> $$
> \widetilde g(\widetilde f(x))
> =g(f(x)),
> $$
>
> so every reconstruction is unchanged, while the variance of each latent coordinate is multiplied by $4$. Reconstruction alone therefore does not make the code distribution match $\mathcal N(0,I)$.

> [!problem|Deriving the VAE Objective]
> Starting from
>
> $$
> \log p^\theta(x)
> =
> \log\int p^\theta(x,z)\,\mathrm d z,
> $$
>
> use $q^\phi(z\mid x)$ and Jensen's inequality to obtain the ELBO used for VAE training. In your final expression, label the reconstruction term and the KL penalty to the prior. Also state the extra KL term, $\mathrm{KL}(q^\phi(z\mid x)\,\|\,p^\theta(z\mid x))$, that explains when the ELBO is tight.

> [!solution]- Solution
>
> Insert $q^\phi(z\mid x)$ into the marginal likelihood:
>
> $$
> \begin{aligned}
> \log p^\theta(x)
> &=
> \log\int
> q^\phi(z\mid x)
> \frac{p^\theta(x,z)}{q^\phi(z\mid x)}
> \,\mathrm d z\\
> &\ge
> \mathbb E_{q^\phi(z\mid x)}
> \left[
> \log\frac{p^\theta(x,z)}{q^\phi(z\mid x)}
> \right].
> \end{aligned}
> $$
>
> Using $p^\theta(x,z)=p^\theta(x\mid z)p(z)$ gives the ELBO
>
> $$
> \mathcal L_{\mathrm{ELBO}}(x)
> =
> \underbrace{
> \mathbb E_{q^\phi(z\mid x)}[\log p^\theta(x\mid z)]
> }_{\text{reconstruction term}}
> -
> \underbrace{
> \mathrm{KL}(q^\phi(z\mid x)\,\|\,p(z))
> }_{\text{prior penalty}}
> $$
>
> The exact gap is
>
> $$
> \log p^\theta(x)
> =
> \mathcal L_{\mathrm{ELBO}}(x)
> +
> \mathrm{KL}\big(q^\phi(z\mid x)\,\|\,p^\theta(z\mid x)\big).
> $$
>
> The bound is tight when the approximate posterior equals the model posterior, $q^\phi(z\mid x)=p^\theta(z\mid x)$.

> [!problem|Linear Autoencoder as PCA]
> Assume the training data are centered. Consider a linear autoencoder with no bias terms, squared reconstruction loss, and bottleneck dimension $k<d$. Explain why the best $k$-dimensional reconstruction space is the span of the top-$k$ principal components of the data. Then give one concrete example where the encoder weights and decoder weights change, but every reconstructed output stays the same. For example, you may rotate or rescale the latent coordinates and undo that change in the decoder.

> [!solution]- Solution
>
> Let the covariance be
>
> $$
> \Sigma=\mathbb E[XX^\top]
> =U\operatorname{diag}(\lambda_1,\ldots,\lambda_d)U^\top,
> \qquad
> \lambda_1\ge\cdots\ge\lambda_d.
> $$
>
> A linear autoencoder with a $k$-dimensional bottleneck reconstructs into at most a $k$-dimensional subspace. Once that subspace is fixed, the closest point to $X$ is its orthogonal projection. For an orthonormal basis $Q\in\mathbb R^{d\times k}$, this reconstruction is $QQ^\top X$, with error
>
> $$
> \mathbb E\|X-QQ^\top X\|_2^2
> =
> \operatorname{tr}(\Sigma)-\operatorname{tr}(Q^\top\Sigma Q).
> $$
>
> Minimizing the error is therefore equivalent to capturing as much variance as possible. The maximum is obtained by $Q=U_{1:k}$, the top-$k$ eigenvectors. One optimal encoder-decoder pair is
>
> $$
> W_e=U_{1:k}^\top,
> \qquad
> W_d=U_{1:k},
> $$
>
> with reconstruction $W_dW_eX=U_{1:k}U_{1:k}^\top X$. The minimum error is $\sum_{i=k+1}^d\lambda_i$.
>
> The factorization is not unique. For any invertible $R\in\mathbb R^{k\times k}$, set
>
> $$
> \widetilde W_e=R W_e,
> \qquad
> \widetilde W_d=W_dR^{-1}.
> $$
>
> Then $\widetilde W_d\widetilde W_e=W_dW_e$, so every reconstruction stays the same even though the latent coordinates have been rotated, rescaled, or mixed.
