from .utils import Registry


metrics = Registry()


class BaseGanMetric(object):
    """
    Base class for GAN's metric calculation
    """

    def __call__(self, gen_path: str, gen_type: str = "folder"):
        raise NotImplementedError("__call__ is not implemented")


class BaseGanMetricStats(BaseGanMetric):
    """
    Base class for GAN's metric with usage of
    real world data statistics
    """

    def __call__(self, gen_path: str, real_path: str, gen_type: str = "folder", real_type: str = "folder"):
        raise NotImplementedError("__call__ is not implemented")
