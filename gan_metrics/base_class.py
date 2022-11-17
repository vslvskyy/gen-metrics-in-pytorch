from typing import Union

from torch.utils.data import DataLoader


class BaseGanMetric(object):
    """
    Base class for GAN's metric calculation
    """

    def calc_metric(self, generated_data: Union[DataLoader, str]):
        """
        Computes metric value
        """
        raise NotImplementedError("calc_metric is not implemented")


class BaseGanMetricStats(BaseGanMetric):
    """
    Base class for GAN's metric with usage of
    real world data statistics
    """

    def __init__(self, real_data: Union[DataLoader, str]):
        if isinstance(real_data, DataLoader):
            self.real_data_loader = real_data
            self.real_data_stats_pth = None
        elif isinstance(real_data, str):
            self.real_data_loader = None
            self.real_data_stats_pth = real_data
        else:
            raise TypeError("real_data should be DataLoader or str")

    def calc_stats(self, loader: DataLoader):
        """
        Calculates distrubution's statistics
        """
        raise NotImplementedError("calc_stats is not implemented")
