from .model_zoo import PolarSpconv
from .model_zoo import PolarOffsetSpconv, PolarOffsetSpconvPytorchMeanshift
from .model_zoo import PolarOffsetSpconvPytorchMeanshiftTrackingMultiFrames

__all__ = {
    'PolarSpconv': PolarSpconv,
    'PolarOffsetSpconv': PolarOffsetSpconv,
    'PolarOffsetSpconvPytorchMeanshift': PolarOffsetSpconvPytorchMeanshift,
    'PolarOffsetSpconvPytorchMeanshiftTrackingMultiFrames': PolarOffsetSpconvPytorchMeanshiftTrackingMultiFrames,
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)
