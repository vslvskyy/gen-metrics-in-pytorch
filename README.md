# Standard GAN Metrics in Pytorch


## Reproducing Results of Pytorch-gan-metrics on CIFAR-10

### Values

| | Train IS | Test IS  | FID | FID (precomputed train stats) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| pytorch-gan-metrics<br>`use_torch=False` | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1518$ | $$3.1518$$ |
| pytorch-gan-metrics<br>`use_torch=True` | $11.26 \pm 0.14$ | $10.95 \pm 0.45$ | $3.1491$ | $$3.1518$$ |
| gan-metrics-in-pytorch<br>`use_torch=False` | $$&mdash;$$ | $$&mdash;$$ | $3.1518$ | $$3.1488$$ |
| gan-metrics-in-pytorch<br>`use_torch=True` | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1496$ | $$3.1528$$ |

### Required Time in Seconds

| | Train IS | Test IS  | FID | FID (precomputed train stats) |
| ------------- | ------------- | -------------- | ------------- | ------------- |
| pytorch-gan-metrics<br>`use_torch=False` | $60.89$ | $12.82$ | $65.70$ | $$3.52$$ |
| pytorch-gan-metrics<br>`use_torch=True` | $62.02$ | $12.35$ | $59.62$ | $$10.27$$ |
| gan-metrics-in-pytorch<br>`use_torch=False` | $$&mdash;$$ | $$&mdash;$$ | $82.98$ | $$24.64$$ |
| gan-metrics-in-pytorch<br>`use_torch=True` | $62.58$ | $12.56$ | $59.17$ | $$0.66$$ |

## Metric values for different GAN's (CIFAR10)

Models were taken from https://github.com/w86763777/pytorch-gan-collections.

The FID is calculated by 50k generated images and CIFAR10 train set.

| Model | IS  | FID |
| ------------- | ------------- | ------------- |
| DCGAN | $5.98 \pm 0.08$ | $47.33$ |
| WGAN(CNN) | $6.46 \pm 0.06$ | $38.53$ |
| WGAN-GP(CNN) | $7.72 \pm 0.08$ | $18.74$ |
| WGAN-GP(ResNet) | $7.81 \pm 0.08$ | $16.75$ |
| SNGAN(CNN) | $7.64 \pm 0.05$ | $18.30$ |
| SNGAN(ResNet) | $8.20 \pm 0.12$ | $14.68$ |
