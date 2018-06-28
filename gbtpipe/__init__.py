# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .Weather import *
    from .Calibration import *
    from .PipeLogging import *
    from .Integration import *
    from .ConvenientIntegration import ConvenientIntegration
    from .ObservationRows import *
    from .SdFitsIO import SdFits, SdFitsIndexRowReader
    from .smoothing import *
    from .commandline import CommandLine
    from .Gridding import griddata
    from .Baseline import *
    from .gbt_pipeline import *
    try:
        from .CyGridding import cygriddata
    except ImportError:
        pass

