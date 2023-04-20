# Standard GAN Metrics Pytorch Implementation

Welcome to our GitHub repository that contains the implementation of several metrics for evaluating generative models! Our goal is to provide a user-friendly and efficient way for researchers and developers to evaluate the performance of their generative models. By using these metrics, you can get a better understanding of how well your generative model is performing and identify areas for improvement. We hope that our implementation will be a valuable resource for those looking to develop and evaluate generative models.

## Implemented metrics

The repository contains implementations of the following metrics:

- Inception Score
- Frechet Inception Distance
- Improved Precision and Recall for Distributions

### Inception Score

[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) (Section 4. Assessment of image quality)

The main component of IS is an Inception model, which is a pre-trained classifier. 

Inception Model returns conditional probability p(y|x) (y - class label, x - input image), this probability has **low entropy** if x is realistic. 
Besides, generative model should produce various images, so p(y) should have **high entropy**. It turns out that the distributions should be far from each other.

$$\text{IS}(G) = \exp(\mathbb{E}_{x\sim p_g}D_{KL}(p(y|x)\|p(y)))$$

Inception Score reflects KL-Divergence between p(y|x) and p(y), so it should be maximized.

### Frechet Inception Distance

[GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf) (Section Experiments.Performance Measure)

FID calculates Frechet Distance between distributions of real and generated features from Inception Model. 

$$\text{FID}(G) = \|\mu_g - \mu_r\|^2_2 + \text{tr}(\Sigma_g + \Sigma_r - 2 (\Sigma_g\Sigma_r)^{\frac12})$$

$\mu, \Sigma$ - mean vector and covariance matrix of generated or real features distribution.

You can also compute Clean-FID, proposed in [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://arxiv.org/pdf/2104.11222.pdf). It uses another image preprocessing pipeline and bicubic filter for image reconstruction.

### Imporoved Precision and Recall for Distributions

[Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/pdf/1904.06991.pdf)

The metric compares manifolds of real and generated Inception Model features.

Firstly, approximations of given manifolds are created. Each approximation is the union of hyperspheres with radius equal to the distance to object's k-th nearest neighbor.

![image](https://user-images.githubusercontent.com/75453192/231453057-9eef5c85-de12-42e6-b266-32005db634e8.png)

Secondly, using manifold's approximation we can determine if an object belongs to a manifold, so we can compute precision and recall as fraction of generated samples, which belogn to real manifold, and fraction of real objects, which belong to generated distribution, respectively.

$$f(\phi, \Phi) := \text{I}[\exists \phi{'}: \|\phi - \phi{'}\|_2 \leq \|\phi{'} - \text{NN}_k(\phi{'}, \Phi)\|_2]$$

$$\text{precision}(\Phi_r, \Phi_g) = \dfrac{1}{|\Phi_g|}\sum\limits_{\phi_g \in \Phi_g}f(\phi_g, \Phi_r) \space\space\space \text{recall}(\Phi_r, \Phi_g) = \dfrac{1}{|\Phi_r|}\sum\limits_{\phi_r \in \Phi_r}f(\phi_r, \Phi_g)$$

## Reproducing Results of pytorch-gan-metrics on CIFAR-10

[pytorch-gan-metrics](https://github.com/w86763777/pytorch-gan-metrics)

### Values

| | Train IS | Test IS  | FID | FID (precomputed train stats) | Clean FID |
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| pytorch-gan-metrics<br>`use_torch=False` | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1518$ | $$3.1518$$ | $$&mdash;$$ |
| pytorch-gan-metrics<br>`use_torch=True` | $11.26 \pm 0.14$ | $10.95 \pm 0.45$ | $3.1491$ | $$3.1518$$ | $$&mdash;$$ |
| gan-metrics-in-pytorch<br>`use_torch=False` | $$&mdash;$$ | $$&mdash;$$ | $3.1518$ | $$3.1488$$ | $$3.23$$ |
| gan-metrics-in-pytorch<br>`use_torch=True` | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1496$ | $$3.1528$$ | $$3.23$$ |

### Required Time in Seconds

| | Train IS | Test IS  | FID | FID (precomputed train stats) | Clean FID |
| ------------- | ------------- | -------------- | ------------- | ------------- | ------------- |
| pytorch-gan-metrics<br>`use_torch=False` | $60.89$ | $12.82$ | $65.70$ | $$3.52$$ | $$&mdash;$$ |
| pytorch-gan-metrics<br>`use_torch=True` | $62.02$ | $12.35$ | $59.62$ | $$10.27$$ | $$&mdash;$$ |
| gan-metrics-in-pytorch<br>`use_torch=False` | $$&mdash;$$ | $$&mdash;$$ | $82.98$ | $$24.64$$ | $$186.28$$ |
| gan-metrics-in-pytorch<br>`use_torch=True` | $62.58$ | $12.56$ | $59.17$ | $$0.66$$ | $$132.24$$ |

## Metric values for different GANs (CIFAR10)

Models were taken from [pytorch-gan-collections](https://github.com/w86763777/pytorch-gan-collections).

The FID is calculated by 50k generated images and CIFAR10.

| Model | IS | FID (w. r. t. train set) | Clean FID (w. r. t. train set) | FID (w. r. t. test set) | Clean FID (w. r. t. test set) | Precision (w. r. t. train set) | Recall (w. r. t. train set) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DCGAN | $$5.98 \pm 0.08$$ | $$47.33$$ | $$58.71$$ | $$49.29$$ | $$60.65$$ | $$0.53$$ | $$0.25$$ |
| WGAN(CNN) | $$6.46 \pm 0.06$$ | $$38.53$$ | $$45.80$$ | $$40.57$$ | $$47.89$$ | $$0.52$$ | $$0.15$$ |
| WGAN-GP(CNN) | $$7.72 \pm 0.08$$ | $$18.74$$ | $$24.75$$ | $$20.92$$ | $$26.97$$ | $$0.54$$ | $$0.51$$ |
| WGAN-GP(ResNet) | $$7.81 \pm 0.08$$ | $$16.75$$ | $$21.03$$ | $$18.87$$ | $$23.24$$ | $$0.55$$ | $$0.50$$ |
| SNGAN(CNN) | $$7.64 \pm 0.05$$ | $$18.30$$ | $$23.29$$ | $$20.47$$ | $$25.53$$ | $$0.55$$ | $$0.50$$ |
| SNGAN(ResNet) | $$8.20 \pm 0.12$$ | $$14.68$$ | $$18.76$$ | $$16.84$$ | $$20.97$$ | $$0.58$$ | $$0.51$$ |
| STYLEGAN2-ADA | $9.70 \pm 0.12$$ | $$2.94$$ | $$3.88$$ | $$5.07$$ | $$6.07$$ | $$0.64$$ | $$0.56$$ |
