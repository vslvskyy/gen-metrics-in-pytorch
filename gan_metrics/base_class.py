class BaseGanMetric(object):
    """
    Base class for GAN's metric calculation
    """

    def __init__(self, train_loader):
        self.train_loader = train_loader
        self.inception_features = None


    def calc_metric(self):
        """
        Computes metric value
        """
        raise NotImplementedError('calc_metric is not implemented')

class BaseGanMetricStats(BaseGanMetric):
    """
    Base class for GAN's metric with useage of
    real world data statistics calculation
    """

    def __init__(self, train_loader, test_loader):
        super().__init__(train_loader)
        self.test_loader = test_loader

    def get_stats(self):
        """
        Calculates distrubution's statistics
        """
        raise NotImplementedError('get_train_stats is not implemented')
