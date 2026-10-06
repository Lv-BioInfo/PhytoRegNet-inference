"""PhytoRegNet's sequence-to-signal inference interface."""

__version__ = "0.1.0"

from .inference import Predictor, load_config

__all__ = ["Predictor", "load_config", "__version__"]
