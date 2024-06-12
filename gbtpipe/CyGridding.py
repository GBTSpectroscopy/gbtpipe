#from sdpy import makecube
import numpy as np
import glob
from astropy.io import fits
import astropy.wcs as wcs
import itertools
from scipy.special import j1
import pdb
import numpy.fft as fft
import astropy.utils.console as console
import astropy.units as u
import astropy.constants as con
import numpy.polynomial.legendre as legendre
import warnings
import Baseline
import os

import cygrid

from spectral_cube import SpectralCube
from radio_beam import Beam
from astropy.coordinates import SkyCoord

import Gridding as gg
from . import __version__

def cygriddata(filelist, 
               cacheSpectra=False,
               pixPerBeam=3.5,
               templateHeader=None,
               gridFunction=gg.jincGrid,
               startChannel=1024, endChannel=3072,
               doBaseline=True,
               baselineRegion=None,
               blorder=1,
               rebase=None,
               rebaseorder=None,
               beamSize=None,
               OnlineDoppler=True,
               flagRMS=False,
               flagRipple=False,
               flagSpike=False,
               rmsThresh=1.25,
               spikeThresh=10,
               projection='TAN',
               outdir=None, 
               outname=None,
               dtype='float64',
               **kwargs):

    """Gridding code for GBT spectral scan data produced by pipeline using CyGrid
    
    Parameters
    ----------
    filelist : list
        List of FITS files to be gridded into an output

    Keywords
    --------
    pixPerBeam : float
        Number of pixels per beam FWHM

    templateHeader : `Header` object
        Template header used for spatial pixel grid.

    gridFunction : function 
        Gridding function to be used.  The default `jincGrid` is a
        tapered circular Bessel function.  The function has call
        signature of func(xPixelCentre, yPixelCenter, xData, yData,
        pixPerBeam)

    startChannel : int
        Starting channel for spectrum within the original spectral data.

    endChannel : int
        End channel for spectrum within the original spectral data

    doBaseline : bool
        Setting to True (default) performs per-scan baseline corrections.

    baselineRegion : `numpy.slice` or list of `numpy.slice`
        Regions in the original pixel data used for fitting the baseline.

    blorder : int
        Order of baseline.  Defaults to 1 (linear)

    rebase : bool
        Setting to True (default is False) performs per-pixel
        rebaselining of the resulting cube.

    beamSize : float
        Telescope beam size at this frequency measured in degrees.

    OnlineDopper : bool
        Setting to True (default) assumes that the Doppler corrections
        in the data are corrected during a telescope scan.  Setting to
        False assumes that the Doppler correction is updated at the
        end of a scan and linearly interpolates between scan ends.
        
    flagRMS : bool
        Setting to True (default = False) flags spectra with rms
        values >rmsThresh x higher than prediction from the radiometer
        formula.  This rms determination assumes that channels are not
        strongly correlated

    rmsThresh : float
        Threshold for scan flagging based on rms.  Default = 1.5

    flagRipple : bool
        Setting to True (default = False) flags spectra with structure
        in the line that is 2x higher than the rms prediction of the
        radiometer formula.  Note that these estimators are outlier
        robust.

    flagSpike : bool
        Setting to True (default = False) flags regions in spectra 
        that show jumps of > 5 times the typical pixel to pixel fluctuation.

    outdir : str
        Output directory name.  Defaults to current working directory.

    outname : str
        Output directory file name.  Defaults to object name in the
        original spectra.

    Returns
    -------
    None

    """

    if outdir is None:
        outdir = os.getcwd()

    if baselineRegion is None:
        baselineRegion = [slice(1024, 1536, 1), slice(2560, 3072, 1)]

    if len(filelist) == 0:
        warnings.warn('There are no FITS files to process ')
        return
    # check that every file in the filelist is valid
    # If not then remove it and send warning message
    for file_i in filelist:
        try:
            fits.open(file_i)
        except:
            warnings.warn('file {0} is corrupted'.format(file_i))
            filelist.remove(file_i)

    # pull a test structure
    hdulist = fits.open(filelist[0])
    s = hdulist[1].data
    
    # Constants block
    sqrt2 = np.sqrt(2)
    mad2rms = 1.4826
    prefac = mad2rms / sqrt2
    c = 299792458.

    nu0 = s[0]['RESTFREQ']
    Data_Unit = s[0]['TUNIT7']

    if outname is None:
        outname = s[0]['OBJECT']

    # New Beam size measurements use 1.18 vs. 1.22 based on GBT Memo 296.
    
    if beamSize is None:
        beamSize = 1.18 * (c / nu0 / 100.0) * 180 / np.pi  # in degrees
    naxis3 = len(s[0]['DATA'][startChannel:endChannel])

    # Default behavior is to park the object velocity at
    # the center channel in the VRAD-LSR frame

    crval3 = s[0]['RESTFREQ'] * (1 - s[0]['VELOCITY'] / c)
    crpix3 = s[0]['CRPIX1'] - startChannel
    ctype3 = s[0]['CTYPE1']
    cdelt3 = s[0]['CDELT1']

    w = wcs.WCS(naxis=3)

    w.wcs.restfrq = nu0
    # We are forcing this conversion to make nice cubes.
    w.wcs.specsys = 'LSRK'
    w.wcs.ssysobs = 'TOPOCENT'

    if templateHeader is None:
        wcsdict = gg.autoHeader(filelist, beamSize=beamSize,
                                pixPerBeam=pixPerBeam, projection=projection)
        w.wcs.crpix = [wcsdict['CRPIX1'], wcsdict['CRPIX2'], crpix3]
        w.wcs.cdelt = np.array([wcsdict['CDELT1'], wcsdict['CDELT2'], cdelt3])
        w.wcs.crval = [wcsdict['CRVAL1'], wcsdict['CRVAL2'], crval3]
        w.wcs.ctype = [wcsdict['CTYPE1'], wcsdict['CTYPE2'], ctype3]
        naxis2 = wcsdict['NAXIS2']
        naxis1 = wcsdict['NAXIS1']
        w.wcs.radesys = s[0]['RADESYS']
        w.wcs.equinox = s[0]['EQUINOX']

    else:
        w.wcs.crpix = [templateHeader['CRPIX1'],
                       templateHeader['CRPIX2'], crpix3]
        w.wcs.cdelt = np.array([templateHeader['CDELT1'],
                                templateHeader['CDELT2'], cdelt3])
        w.wcs.crval = [templateHeader['CRVAL1'],
                       templateHeader['CRVAL2'], crval3]
        w.wcs.ctype = [templateHeader['CTYPE1'],
                       templateHeader['CTYPE2'], ctype3]
        naxis2 = templateHeader['NAXIS2']
        naxis1 = templateHeader['NAXIS1']
        w.wcs.radesys = templateHeader['RADESYS']
        w.wcs.equinox = templateHeader['EQUINOX']
        pixPerBeam = np.abs(beamSize / w.pixel_scale_matrix[1,1])
        if pixPerBeam < 3.5:
            warnings.warn('Template header requests {0}'.format(pixPerBeam)+
                          ' pixels per beam.')
        if (((w.wcs.ctype[0]).split('-'))[0] !=
            ((s[0]['CTYPE1']).split('-'))[0]):
            warnings.warn('Spectral data not in same frame as template header')
            eulerFlag = True

#    outCube = np.zeros((int(naxis3), int(naxis2), int(naxis1)),dtype=dtype)
#   outWts = np.zeros((int(naxis2), int(naxis1)),dtype=dtype)

    xmat, ymat = np.meshgrid(np.arange(naxis1), np.arange(naxis2),
                             indexing='ij')
    xmat = xmat.reshape(xmat.size)
    ymat = ymat.reshape(ymat.size)
    xmat = xmat.astype(int)
    ymat = ymat.astype(int)

    ctr = 0

    speclist = []
    lonlist = []
    latlist = []
    for thisfile in filelist:
        ctr += 1
        cachefile = thisfile.replace('.fits', '_speccache.npz')
        if cacheSpectra and os.path.isfile(cachefile):
            npzfile = np.load(cachefile)
            speclist += npzfile['speclist'].tolist()
            lonlist += npzfile['lonlist'].tolist()
            latlist += npzfile['latlist'].tolist()
        else:
            s = fits.open(thisfile)
            print("Now processing {0}".format(thisfile))
            print("This is file {0} of {1}".format(ctr, len(filelist)))

            nuindex = np.arange(len(s[1].data['DATA'][0]))

            if not OnlineDoppler:
                vframe = gg.VframeInterpolator(s[1].data)
            else:
                vframe = s[1].data['VFRAME']
            flagct = 0
            if eulerFlag:
                if 'GLON' in s[1].data['CTYPE2'][0]:
                    inframe = 'galactic'
                elif 'RA' in s[1].data['CTYPE2'][0]:
                    inframe = 'fk5'
                else:
                    raise NotImplementedError
                if 'GLON' in w.wcs.ctype[0]:
                    outframe = 'galactic'
                elif 'RA' in w.wcs.ctype[0]:
                    outframe = 'fk5'
                else:
                    raise NotImplementedError

                coords = SkyCoord(s[1].data['CRVAL2'],
                                  s[1].data['CRVAL3'],
                                  unit = (u.deg, u.deg),
                                  frame=inframe)
                coords_xform = coords.transform_to(outframe)
                if outframe == 'fk5':
                    longCoord = coords_xform.ra.deg
                    latCoord = coords_xform.dec.deg
                elif outframe == 'galactic':
                    longCoord = coords_xform.l.deg
                    latCoord = coords_xform.b.deg
            else:
                longCoord = s[1].data['CRVAL2'],
                latCoord = s[1].data['CRVAL3']
            lonlist += [longCoord]
            latlist += [latCoord]

            for idx, spectrum in enumerate(console.ProgressBar((s[1].data))):
                # Generate Baseline regions
                baselineIndex = np.concatenate([nuindex[ss]
                                                for ss in baselineRegion])

                specData = spectrum['DATA']

                # baseline fit
                if doBaseline & np.all(np.isfinite(specData)):
                    specData = gg.baselineSpectrum(specData, order=blorder,
                                                   baselineIndex=baselineIndex)

                # This part takes the TOPOCENTRIC frequency that is at
                # CRPIX1 (i.e., CRVAL1) and calculates the what frequency
                # that would have in the LSRK frame with freqShiftValue.
                # This then compares to the desired frequency CRVAL3.

                DeltaNu = gg.freqShiftValue(spectrum['CRVAL1'],
                                            -vframe[idx]) - crval3
                DeltaChan = DeltaNu / cdelt3
                specData = gg.channelShift(specData, -DeltaChan)
                outslice = (specData)[startChannel:endChannel]
                spectrum_wt = np.isfinite(outslice).astype(float)
                outslice = np.nan_to_num(outslice)
                # xpoints, ypoints, zpoints = w.wcs_world2pix(longCoord[idx],
                #                                             latCoord[idx],
                #                                             spectrum['CRVAL1'], 0)
                tsys = spectrum['TSYS']

                if flagSpike:
                    jumps = (outslice - np.roll(outslice, -1))
                    noise = gg.mad1d(jumps) * 2**(-0.5)
                    spikemask = (np.abs(jumps) < spikeThresh * noise)
                    spikemask = spikemask * np.roll(spikemask, 1)
                    spectrum_wt *= spikemask
                if flagRMS:
                    radiometer_rms = tsys / np.sqrt(np.abs(spectrum['CDELT1']) *
                                                    spectrum['EXPOSURE'])
                    scan_rms = prefac * np.median(np.abs(outslice[0:-2] -
                                                            outslice[2:]))

                    if scan_rms > rmsThresh * radiometer_rms:
                        tsys = 0 # Blank spectrum

                if flagRipple:
                    scan_rms = prefac * np.median(np.abs(outslice[0:-2] -
                                                         outslice[2:]))
                    ripple = prefac * sqrt2 * np.median(np.abs(outslice))

                    if ripple > 2 * scan_rms:
                        tsys = 0 # Blank spectrum
                if tsys == 0:
                    flagct +=1
                speclist += [outslice]
                # if (tsys > 10) and (xpoints > 0) and (xpoints < naxis1) \
                #         and (ypoints > 0) and (ypoints < naxis2):
                #     pixelWeight, Index = gridFunction(xmat, ymat,
                #                                       xpoints, ypoints,
                #                                       pixPerBeam)
                #     vector = np.outer(outslice * spectrum_wt,
                #                       pixelWeight / tsys**2)
                #     wts = pixelWeight / tsys**2
                #     outCube[:, ymat[Index], xmat[Index]] += vector
                #     outWts[ymat[Index], xmat[Index]] += wts
            print ("Percentage of flagged scans: {0:4.2f}".format(
                    100*flagct/float(idx)))
            # Temporarily do a file write for every batch of scans.
            if cacheSpectra:
                np.savez(thisfile.replace('.fits','_speccache.npz'),
                         lonlist=lonlist, latlist=latlist,
                         speclist=speclist)
    try:
        lonlist = np.squeeze(np.stack(lonlist))
        latlist = np.squeeze(np.stack(latlist))
        speclist = np.squeeze(np.stack(speclist))
    except ValueError:
        lonlist = np.squeeze(np.hstack(lonlist))
        latlist = np.squeeze(np.hstack(latlist))
        speclist = np.squeeze(np.stack(speclist))
    # DO THE GRIDDING
    hdr = w.to_header()
    hdr['NAXIS1'] = naxis1
    hdr['NAXIS2'] = naxis2
    hdr['NAXIS3'] = 10
    
    gridder = cygrid.WcsGrid(hdr)
    gridder.set_kernel('gauss1d',(beamSize/2.355/2), 2*(beamSize/2.355),
                       beamSize/2/2.355)
    gridder.grid(lonlist, latlist, speclist,
                 dtype=dtype)
    hdr = fits.Header(w.to_header())
    s = fits.open(thisfile)
    hdr = gg.addHeader_nonStd(hdr, beamSize, s[1].data[0])
    hdr.add_history('Using GBTPIPE gridder version {0}'.format(__version__))

    hdu = fits.PrimaryHDU(gridder.get_datacube(),header=hdr)
    hdu.writeto(outdir + '/' + outname + '.fits', clobber=True)

    # outWts.shape = (1,) + outWts.shape
    # outCube /= outWts

    # hdu = fits.PrimaryHDU(outCube, header=hdr)

    # w2 = w.dropaxis(2)
    # hdr2 = fits.Header(w2.to_header())
    # hdu2 = fits.PrimaryHDU(outWts, header=hdr2)
    # hdu2.writeto(outdir + '/' + outname + '_wts.fits', clobber=True)

    if rebase:
        if rebaseorder is None:
            rebaseorder = blorder
        if 'NH3_11' in outname:
            Baseline.rebaseline(outdir + '/' + outname + '.fits',
                                windowFunction=Baseline.ammoniaWindow,
                                line='oneone', blorder=rebaseorder,
                                **kwargs)

        elif 'NH3_22' in outname:
            Baseline.rebaseline(outdir + '/' + outname + '.fits',
                                windowFunction=Baseline.ammoniaWindow,
                                line='twotwo', blorder=rebaseorder,
                                **kwargs)

        elif 'NH3_33' in outname:
            Baseline.rebaseline(outdir + '/' + outname + '.fits',
                                winfunc = Baseline.ammoniaWindow,
                                blorder=rebaseorder,
                                line='threethree', **kwargs)
        else:
            Baseline.rebaseline(outdir + '/' + outname + '.fits',
                                blorder=rebaseorder,
                                windowFunction=Baseline.tightWindow, 
                                **kwargs)
