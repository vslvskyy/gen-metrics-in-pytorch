# Standard GAN Metrics in Pytorch


## Reproducing Results of Pytorch-gan-metrics on CIFAR-10

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

## Metric values for different GAN's (CIFAR10)

Models were taken from https://github.com/w86763777/pytorch-gan-collections.

The FID is calculated by 50k generated images and CIFAR10.

| Model | IS | FID (w. r. t. train set) | Clean FID (w. r. t. train set) | FID (w. r. t. test set) | Clean FID (w. r. t. test set) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DCGAN | $$5.98 \pm 0.08$$ | $$47.33$$ | $$58.71$$ | $$49.29$$ | $$60.65$$ |
| WGAN(CNN) | $$6.46 \pm 0.06$$ | $$38.53$$ | $$45.80$$ | $$40.57$$ | $$47.89$$ |
| WGAN-GP(CNN) | $$7.72 \pm 0.08$$ | $$18.74$$ | $$24.75$$ | $$20.92$$ | $$26.97$$ |
| WGAN-GP(ResNet) | $$7.81 \pm 0.08$$ | $$16.75$$ | $$21.03$$ | $$18.87$$ | $$23.24$$ |
| SNGAN(CNN) | $$7.64 \pm 0.05$$ | $$18.30$$ | $$23.29$$ | $$20.47$$ | $$25.53$$ |
| SNGAN(ResNet) | $$8.20 \pm 0.12$$ | $$14.68$$ | $$18.76$$ | $$16.84$$ | $$20.97$$ |
