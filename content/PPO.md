### Proximal Policy Optimization (PPO)

Applying the proximal framework to policy optimization with importance sampling:

[
 \theta^{k+1} =
 \arg\max_{\theta} \Big{
 \E_{\mathcal D}
 \Big[
 \frac{\pi^\theta(a|s)}{\pi^{\text{data}}(a|s)}, r(a,s)
 \Big]

- \beta D(\pi^\theta \Vert \pi^{\theta^k})
   \Big}.
   ]

The penalty term is usually the **KL divergence**:

[
 D(\pi^\theta \Vert \pi^{\theta^k})
 = \E_{s\sim p_0}
 \big[
 \mathrm{KL}(\pi^\theta(\cdot|s) \Vert \pi^{\theta^k}(\cdot|s))
 \big].
 ]