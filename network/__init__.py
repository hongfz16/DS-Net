from .model_zoo import PolarOffsetSpconv, PolarOffsetSpconvPytorchMeanshift

__all__ = {
    'PolarOffsetSpconv': PolarOffsetSpconv,
    'PolarOffsetSpconvPytorchMeanshift': PolarOffsetSpconvPytorchMeanshift,
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)
