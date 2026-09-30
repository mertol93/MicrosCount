"""MicrosCount: microscopy image analysis for non-programmers.

Modules
-------
translocation
    Nuclear/cytoplasmic intensity ratio (nuclear translocation), including an
    exact re-implementation of Noursadeghi et al. (2008) J Immunol Methods
    329:194-200 and a per-cell variant.
porosity
    Porosity and pore-size distribution of SEM images (Python port of
    A. Rabbani's SEM_Porosity MATLAB script, BSD-3-Clause).
"""

from ._version import __version__

APP_NAME = "MicrosCount"

__all__ = ["__version__", "APP_NAME"]
