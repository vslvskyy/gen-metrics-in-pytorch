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

    def __init__(self, test_loader):
        self.test_loader = test_loader


    def get_stats(self):
        """
        Calculates distrubution's statistics
        """
        raise NotImplementedError('get_stats is not implemented')
