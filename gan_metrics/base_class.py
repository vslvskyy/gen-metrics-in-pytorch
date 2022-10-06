import torch

from typing import Union

class BaseGanMetric(object):
    """
    Base class for GAN's metric calculation
    """

    def calc_metric(self):
        """
        Computes metric value
        """
        raise NotImplementedError('calc_metric is not implemented')


class BaseGanMetricStats(BaseGanMetric):
    """
    Base class for GAN's metric with usage of
    real world data statistics
    """

    def __init__(self, test_data: Union[torch.utils.data.DataLoader, str]):
        if isinstance(test_data, torch.utils.data.DataLoader):
            self.test_loader = test_data
            self.test_stats_pth = None
        elif isinstance(test_data, str):
            self.test_loader = None
            self.test_stats_pth = test_data
        else:
            raise TypeError


    def get_stats(self):
        """
        Calculates distrubution's statistics
        """
        raise NotImplementedError('get_stats is not implemented')
