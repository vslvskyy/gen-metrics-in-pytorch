# Standard Generative Models Metrics Pytorch Implementation

Welcome to our GitHub repository that contains the implementation of several metrics for evaluating generative models! Our goal is to provide a user-friendly and efficient way for researchers and developers to evaluate the performance of their generative models. By using these metrics, you can get a better understanding of how well your generative model is performing and identify areas for improvement. We hope that our implementation will be a valuable resource for those looking to develop and evaluate generative models.

## Implemented metrics

The repository contains implementations of the following metrics:

- Inception Score (is)
- Frechet Inception Distance (fid, clean_fid, fid_numpy) 
- Improved Precision and Recall for Distributions (precision_recall)

### Inception Score

[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) (Section 4. Assessment of image quality)

The main component of IS is an Inception model, which is a pre-trained classifier. 

Inception Model returns conditional probability p(y|x) (y - class label, x - input image), this probability has **low entropy** if x is realistic. 
Besides, generative model should produce various images, so p(y) should have **high entropy**. It turns out that the distributions should be far from each other.

$$\text{IS}(G) = \exp\big(\mathbb{E}_{x\sim p_g}\text{KL}[p(y|x)\|p(y)]\big)$$

Inception Score reflects KL-Divergence between p(y|x) and p(y), so it should be maximized.

### Frechet Inception Distance

[GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf) (Section Experiments.Performance Measure)

FID calculates Frechet Distance between distributions of real and generated features from Inception Model. 

$$\text{FID}(G) = \|\mu_g - \mu_r\|^2_2 + \text{tr}(\Sigma_g + \Sigma_r - 2 (\Sigma_g\Sigma_r)^{\frac12})$$

$\mu, \Sigma$ - mean vector and covariance matrix of generated or real features distribution.

Three configurations of this metric are implemented:
- FID: all calculations are performed on pytorch. Matrix root extracting from [matrix_sqrt](https://github.com/msubhransu/matrix-sqrt).
- FID-Numpy: most calculations on pytorch. Matrix root extracting via scipy.linalg.sqrtm for numerical stability.
- Clean-FID: proposed in [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://arxiv.org/pdf/2104.11222.pdf). It uses another image preprocessing pipeline and bicubic filter for image reconstruction.

### Imporoved Precision and Recall for Distributions

[Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/pdf/1904.06991.pdf)

The metric compares manifolds of real and generated Inception Model features.

Firstly, approximations of given manifolds are created. Each approximation is the union of hyperspheres with radius equal to the distance to object's k-th nearest neighbor.

![image](https://user-images.githubusercontent.com/75453192/231453057-9eef5c85-de12-42e6-b266-32005db634e8.png)

Secondly, using manifold's approximation we can determine if an object belongs to a manifold, so we can compute precision and recall as fraction of generated samples, which belogn to real manifold, and fraction of real objects, which belong to generated distribution, respectively.

$$f(\phi, \Phi) := \text{I}[\exists \phi{'}: \|\phi - \phi{'}\|_2 \leq \|\phi{'} - \text{NN}_k(\phi{'}, \Phi)\|_2]$$

$$\text{precision}(\Phi_r, \Phi_g) = \dfrac{1}{|\Phi_g|}\sum\limits_{\phi_g \in \Phi_g}f(\phi_g, \Phi_r) \space\space\space \text{recall}(\Phi_r, \Phi_g) = \dfrac{1}{|\Phi_r|}\sum\limits_{\phi_r \in \Phi_r}f(\phi_r, \Phi_g)$$

## Reproducing official implementations' results on CIFAR-10

Official results
- IS: [improved-gan](https://github.com/openai/improved-gan)
- FID (numpy): [TTUR](https://github.com/bioinf-jku/TTUR)
- FID (torch implementation): [pytorch-gan-metrics](https://github.com/w86763777/pytorch-gan-metrics)
- Clean FID: [clean-fid](https://github.com/GaParmar/clean-fid)
- Precision and Recall: [improved-precision-and-recall-metric](https://github.com/kynkaat/improved-precision-and-recall-metric)

### Values

Train/Test IS: Inception Score of cifar10 train(50k)/test(10k). FIDs and Precision and Recall were calculated beteween cifar10 train(50k) and test(10k) sets.

|| Train IS $\uparrow$ | Test IS $\uparrow$ | FID $\downarrow$ | FID Numpy $\downarrow$ | Clean FID $\downarrow$ | Precision $\uparrow$ | Recall $\uparrow$ |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| official | $$11.24 \pm 0.20$$ | $$10.98 \pm 0.22$$ | $$3.1509$$ | $$3.1508$$ | $$3.2351$$ | $$0.689$$ | $$0.695$$ |
| gen-metrics-in-pytorch | $11.26 \pm 0.12$ | $10.95 \pm 0.43$ | $3.1501$ | $$3.1492$$ | $$3.2312$$ | $$0.689$$ | $$0.695$$ |

### Required Time in Seconds

Calculations were made on one V100 GPU.

| Data type | Train IS | Test IS  | FID | FID Numpy | Clean FID | Precision and Recall |
| ------------- | ------------- | -------------- | ------------- | ------------- | ------------- | ------------- |
| folder | $$120$$ | $$27$$ | $$146$$ | $$174$$ | $$164$$ | $$294$$ |
| stats | $$3.24$$ | $$1.96$$ | $$3.92$$ | $$31$$ | $$3.77$$ | $$40$$ |

Metrics calculation with data type "folder" may last about 1000 seconds when you create Dataset with images for the first time (before python caching).

## Metric values for different models (CIFAR10)

Models were taken from
- GANs: [pytorch-gan-collections](https://github.com/w86763777/pytorch-gan-collections)
- StyleGAN2-ADA: [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- DDPM: [diffusion](https://github.com/hojonathanho/diffusion)
- DDPM-GAN: [denoising-diffusion-gan](https://github.com/NVlabs/denoising-diffusion-gan)

The metrics were calculated between 50k generated images and CIFAR10 train set (50k).

| Model | IS $\uparrow$ | FID $\downarrow$ | FID Numpy $\downarrow$ | Clean FID $\downarrow$  | Precision $\uparrow$ | Recall $\uparrow$ |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DCGAN | $$5.98 \pm 0.08$$ | $$47.33$$ | $$47.33$$ | $$58.71$$ | $$0.53$$ | $$0.25$$ |
| WGAN(CNN) | $$6.46 \pm 0.06$$ | $$38.53$$ | $$38.53$$ | $$45.80$$ | $$0.52$$ | $$0.15$$ |
| WGAN-GP(CNN) | $$7.72 \pm 0.08$$ | $$18.74$$ | $$18.74$$ | $$24.75$$ | $$0.54$$ | $$0.51$$ |
| WGAN-GP(ResNet) | $$7.81 \pm 0.08$$ | $$16.75$$ | $$16.75$$ | $$21.03$$ | $$0.55$$ | $$0.50$$ |
| SNGAN(CNN) | $$7.64 \pm 0.05$$ | $$18.30$$ | $$18.30$$ | $$23.29$$ | $$0.55$$ | $$0.50$$ |
| SNGAN(ResNet) | $$8.20 \pm 0.12$$ | $$14.68$$ | $$14.68$$ | $$18.76$$ | $$0.58$$ | $$0.51$$ |
| STYLEGAN2-ADA | $$9.70 \pm 0.12$$ | $$2.94$$ | $$2.94$$ | $$3.88$$ | $$0.64$$ | $$0.56$$ |
| DDPM | $$9.48 \pm 0.10$$ | $$3.22$$ | $$3.23$$ | $$4.01$$ | $$0.69$$ | $$0.58$$ |
| DDPM-GAN | $$9.32 \pm 0.10$$ | $$4.20$$ | $$4.20$$ | $$5.16$$ | $$0.66$$ | $$0.52$$ |


## Installation guide
Needed requirments:
- scipy==1.5.4
- torch>=1.8.1
- torchvision>=0.9.1
- numpy

Clone repository and install dependencies
```shell
git clone https://github.com/vslvskyy/gen-metrics-in-pytorch
cd gen_metrics_in_pytorch
pip install -r requirements.txt
```

## Metrics calculating

You can run `main.py` with appropriate arguments to calculate metrics. 
```shell
python main.py \
    --metric fid \
    --gen_data folder \path_to_folder_with\images \
    --real_data stats \path_to_file_with\stats.npz \
    --gen_save_path \path_to_file_to_save\stats
```

Or you can use metrics classes directly:
```python
from gen_metrics import Fid, InceptionScore, ImprovedPRD


metric_class = Fid()
metric_val = metric_class(
    gen_path, real_path,
    gen_type, real_type,
    gen_save_path, real_save_path
)
```

```python
from gen_metrics import ImprovedPRD


precision = ImprovedPRD.compute_coverage(
    real_knn_distances, real_ftrs, gen_ftrs
)
recall = ImprovedPRD.compute_coverage(
    gen_knn_distances, gen_ftrs, real_ftrs
)
```


