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
