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
    from Weather import Weather
    from Calibration import Calibration
    from PipeLogging import Logging
    from Integration import Integration
    from ConvenientIntegration import ConvenientIntegration
    from ObservationRows import ObservationRows
    from SdFitsIO import SdFits, SdFitsIndexRowReader
    import smoothing
    from commandline import CommandLine
    from Gridding import griddata
    import Baseline
    from gbt_pipeline import *
    try:
        from CyGridding import cygriddata
    except ImportError:
        pass

