from .model_zoo import PolarOffsetSpconv, PolarOffsetSpconvPytorchMeanshift
from .model_zoo import PolarOffsetSpconvPytorchMeanshiftTracking
from .model_zoo import PolarOffsetSpconvPytorchMeanshiftTrackingSiamese
from .model_zoo import PolarOffsetSpconvPytorchMeanshiftTrackingSiameseWithGeoClue

__all__ = {
    'PolarOffsetSpconv': PolarOffsetSpconv,
    'PolarOffsetSpconvPytorchMeanshift': PolarOffsetSpconvPytorchMeanshift,
    'PolarOffsetSpconvPytorchMeanshiftTracking': PolarOffsetSpconvPytorchMeanshiftTracking,
    'PolarOffsetSpconvPytorchMeanshiftTrackingSiamese': PolarOffsetSpconvPytorchMeanshiftTrackingSiamese,
    'PolarOffsetSpconvPytorchMeanshiftTrackingSiameseWithGeoClue': PolarOffsetSpconvPytorchMeanshiftTrackingSiameseWithGeoClue,
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)
