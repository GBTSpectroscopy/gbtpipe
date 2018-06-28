The `gbtpipe` package is a package for reducing and mapping On-the-fly mapping data created at the GBT in spectral line observations.  The module consists of two main components:

* An installable version of the the GBTpipeline which is forked from NRAO's [gbt-pipeline](https://github.com/nrao/gbt-pipeline) package.  This is adapted for use in ARGUS operations.
* An On-the-fly gridding package that builds scan data into spectral line data cubes.

**pipeline** 

The pipeline component of `gbtpipe` is primarily used to calibrate VEGAS spectrometer data generated using the ARGUS W-band focal-plane array.  For other cases, such as the KFPA, the gbt-pipeline works well and does not require additional changes.  For ARGUS, the calibration process requires custom implementation of switching modes to use the vane calibration.  
