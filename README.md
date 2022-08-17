# Standard GAN Metrics in Pytorch


## Reproducing Results of Pytorch-gan-metrics on CIFAR-10

### Values

| | Train IS | Test IS  | FID | FID (precomputed train stats) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| use_torch=False | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1518$ | $$3.1518$$ |
| use_torch=True | $11.26 \pm 0.14$ | $10.95 \pm 0.45$ | $3.1491$ | $$3.1518$$ |

### Required Time in seconds

| | Train IS | Test IS  | FID | FID (precomputed train stats) |
| ------------- | ------------- | -------------- | ------------- | ------------- |
| use_torch=False | $60.89$ | $12.82$ | $65.70$ | $$3.52$$ |
| use_torch=True | $62.02$ | $12.35$ | $59.62$ | $$10.27$$ |


## Our results on CIFAR-10

### Values

| | Train IS | Test IS  | FID |
| ------------- | ------------- | ------------- | ------------- |
| use_torch=False | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1518$ |
| use_torch=True | $11.26 \pm 0.13$ | $10.95 \pm 0.43$ | $3.1496$ |

### Required Time in seconds

| | Train IS | Test IS  | FID |
| ------------- | ------------- | ------------- | ------------- |
| use_torch=False | $61.62$ | $13.11$ | $80.61$ |
| use_torch=True | $62.58$ | $12.56$ | $72.16$ |
