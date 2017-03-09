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

from . import __version__


def baselineSpectrum(spectrum, order=1, baselineIndex=()):
    x = np.linspace(-1, 1, len(spectrum))
    coeffs = legendre.legfit(x[baselineIndex], spectrum[baselineIndex], order)
    spectrum -= legendre.legval(x, coeffs)
    return(spectrum)


def freqShiftValue(freqIn, vshift, convention='RADIO'):
    cms = 299792458.
    if convention.upper() in 'OPTICAL':
        return freqIn / (1.0 + vshift / cms)
    if convention.upper() in 'TRUE':
        return freqIn * ((cms + vshift) / (cms - vshift))**0.5
    if convention.upper() in 'RADIO':
        return freqIn * (1.0 - vshift / cms)


def channelShift(x, ChanShift):
    # Shift a spectrum by a set number of channels.
    ftx = np.fft.fft(x)
    m = np.fft.fftfreq(len(x))
    phase = np.exp(2 * np.pi * m * 1j * ChanShift)
    x2 = np.real(np.fft.ifft(ftx * phase))
    return(x2)


def jincGrid(xpix, ypix, xdata, ydata, pixPerBeam):
    a = 1.55 / (3.0 / pixPerBeam)
    b = 2.52 / (3.0 / pixPerBeam)

    Rsup = 1.09 * pixPerBeam  # Support radius is ~1 FWHM (Leroy likes 1.09)
    dmin = 1e-4
    dx = (xdata - xpix)
    dy = (ydata - ypix)

    pia = np.pi / a
    b2 = 1. / (b**2)
    distance = np.sqrt(dx**2 + dy**2)

    ind  = (np.where(distance <= Rsup))
    d = distance[ind]
    wt = j1(d * pia) / \
        (d * pia) * \
        np.exp(-d**2 * b2)
    wt[(d < dmin)] = 0.5  # Peak of the jinc function is 0.5 not 1.0

    return(wt, ind)


def VframeInterpolator(scan):
    # Find cases where the scan number is 
    startidx = scan['PROCSEQN']!=np.roll(scan['PROCSEQN'],1)
    scanstarts = scan[startidx]
    indices = np.arange(len(scan))
    startindices = indices[startidx]
    scannum = scanstarts['PROCSEQN']
    vfs = scanstarts['VFRAME']

    odds = (scannum % 2) == 1
    evens = (scannum % 2) == 0
    
    coeff_odds,_,_,_ = np.linalg.lstsq(\
        np.c_[startindices[odds]*1.0,
              np.ones_like(startindices[odds])],
        vfs[odds])

    coeff_evens,_,_,_ = np.linalg.lstsq(\
        np.c_[startindices[evens]*1.0,
              np.ones_like(startindices[evens])],
        vfs[evens])

    vfit = np.zeros(len(scan))+np.nan

    for thisone, singlescan in enumerate(scan):
        startv = vfs[scannum == singlescan['PROCSEQN']]
        startt = startindices[scannum == singlescan['PROCSEQN']]
        if singlescan['PROCSEQN'] % 2 == 0:
            endv = coeff_odds[1] + coeff_odds[0] * (startt+94)
        if singlescan['PROCSEQN'] % 2 == 1:
            endv = coeff_evens[1] + coeff_evens[0] * (startt+94)

        endt = startt+94
        try:
            vfit[thisone] = (thisone - startt) * \
                (endv - startv)/94 + startv
        except:
            pass
    return(vfit)
    
def autoHeader(filelist, beamSize=0.0087, pixPerBeam=3.0):
    RAlist = []
    DEClist = []
    for thisfile in filelist:
        s = fits.getdata(thisfile)
        try:
            RAlist = RAlist + [s['CRVAL2']]
            DEClist = DEClist + [s['CRVAL3']]
        except:
            pdb.set_trace()

    longitude = np.array(list(itertools.chain(*RAlist)))
    latitude = np.array(list(itertools.chain(*DEClist)))
    longitude = longitude[longitude != 0]
    latitude = latitude[latitude != 0]
    minLon = np.nanmin(longitude)
    maxLon = np.nanmax(longitude)
    minLat = np.nanmin(latitude)
    maxLat = np.nanmax(latitude)

    naxis2 = np.ceil((maxLat - minLat) /
                     (beamSize / pixPerBeam) + 2 * pixPerBeam)
    crpix2 = naxis2 / 2
    cdelt2 = beamSize / pixPerBeam
    crval2 = (maxLat + minLat) / 2
    ctype2 = s[0]['CTYPE3'] + '--TAN'
    # Negative to go in the usual direction on sky:
    cdelt1 = -beamSize / pixPerBeam

    naxis1 = np.ceil((maxLon - minLon) /
                     (beamSize / pixPerBeam) *
                     np.cos(crval2 / 180 * np.pi) + 2 * pixPerBeam)
    crpix1 = naxis1 / 2
    crval1 = (minLon + maxLon) / 2
    ctype1 = s[0]['CTYPE2'] + '---TAN'
    outdict = {'CRVAL1': crval1, 'CRPIX1': crpix1,
               'CDELT1': cdelt1, 'NAXIS1': naxis1,
               'CTYPE1': ctype1, 'CRVAL2': crval2,
               'CRPIX2': crpix2, 'CDELT2': cdelt2,
               'NAXIS2': naxis2, 'CTYPE2': ctype2}

    return(outdict)


def addHeader_nonStd(hdr, beamSize, sample):

    bunit_dict = {'Tmb':'K',
                  'Ta*':'K'}
    inst_dict = {'RcvrArray75_115':'ARGUS',
                 'RcvrArray18_26':'KFPA',
                 'Rcvr1_2':'L-BAND'}

    hdr.set('BUNIT', value= bunit_dict[sample['TUNIT7']], 
            comment=sample['TUNIT7'])
    hdr['INSTRUME'] = inst_dict[sample['FRONTEND']]
    hdr['BMAJ'] = beamSize
    hdr['BMIN'] = beamSize
    hdr['BPA'] = 0.0
    hdr['TELESCOP'] = 'GBT'

    return(hdr)

def griddata(filelist, 
             pixPerBeam=3.5,
             templateHeader=None,
             gridFunction=jincGrid,
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
             outdir=None, 
             outname=None,
             **kwargs):

    """Gridding code for GBT spectral scan data produced by pipeline.
    
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
        values >1.25x higher than prediction from the radiometer
        formula.  This rms determination assumes that channels are not
        strongly correlated.

    flagRipple : bool
        Setting to True (default = False) flags spectra with structure
        in the line that is 2x higher than the rms prediction of the
        radiometer formula.  Note that these estimators are outlier
        robust.

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

    if beamSize is None:
        beamSize = 1.22 * (c / nu0 / 100.0) * 180 / np.pi  # in degrees
    naxis3 = len(s[0]['DATA'][startChannel:endChannel])

    # Default behavior is to park the object velocity at
    # the center channel in the VRAD-LSR frame

    crval3 = s[0]['RESTFREQ'] * (1 - s[0]['VELOCITY'] / c)
    crpix3 = s[0]['CRPIX1'] - startChannel
    ctype3 = s[0]['CTYPE1']
    cdelt3 = s[0]['CDELT1']

    w = wcs.WCS(naxis=3)

    w.wcs.restfrq = nu0
    w.wcs.radesys = s[0]['RADESYS']
    w.wcs.equinox = s[0]['EQUINOX']
    # We are forcing this conversion to make nice cubes.
    w.wcs.specsys = 'LSRK'
    w.wcs.ssysobs = 'TOPOCENT'

    if templateHeader is None:
        wcsdict = autoHeader(filelist, beamSize=beamSize,
                             pixPerBeam=pixPerBeam)
        w.wcs.crpix = [wcsdict['CRPIX1'], wcsdict['CRPIX2'], crpix3]
        w.wcs.cdelt = np.array([wcsdict['CDELT1'], wcsdict['CDELT2'], cdelt3])
        w.wcs.crval = [wcsdict['CRVAL1'], wcsdict['CRVAL2'], crval3]
        w.wcs.ctype = [wcsdict['CTYPE1'], wcsdict['CTYPE2'], ctype3]
        naxis2 = wcsdict['NAXIS2']
        naxis1 = wcsdict['NAXIS1']
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
    outCube = np.zeros((int(naxis3), int(naxis2), int(naxis1)))
    outWts = np.zeros((int(naxis2), int(naxis1)))

    xmat, ymat = np.meshgrid(np.arange(naxis1), np.arange(naxis2),
                             indexing='ij')
    xmat = xmat.reshape(xmat.size)
    ymat = ymat.reshape(ymat.size)
    xmat = xmat.astype(np.int)
    ymat = ymat.astype(np.int)

    ctr = 0


    for thisfile in filelist:
        ctr += 1
        s = fits.open(thisfile)
        print("Now processing {0}".format(thisfile))
        print("This is file {0} of {1}".format(ctr, len(filelist)))

        nuindex = np.arange(len(s[1].data['DATA'][0]))

        if not OnlineDoppler:
            vframe = VframeInterpolator(s[1].data)
        else:
            vframe = s[1].data['VFRAME']

        for idx, spectrum in enumerate(console.ProgressBar((s[1].data))):
            # Generate Baseline regions
            baselineIndex = np.concatenate([nuindex[ss]
                                            for ss in baselineRegion])

            specData = spectrum['DATA']
            # baseline fit
            if doBaseline & np.all(np.isfinite(specData)):
                specData = baselineSpectrum(specData, order=blorder,
                                            baselineIndex=baselineIndex)

            # This part takes the TOPOCENTRIC frequency that is at
            # CRPIX1 (i.e., CRVAL1) and calculates the what frequency
            # that would have in the LSRK frame with freqShiftValue.
            # This then compares to the desired frequency CRVAL3.
                
            DeltaNu = freqShiftValue(spectrum['CRVAL1'],
                                     -vframe[idx]) - crval3
            DeltaChan = DeltaNu / cdelt3
            specData = channelShift(specData, -DeltaChan)
            outslice = (specData)[startChannel:endChannel]
            spectrum_wt = np.isfinite(outslice).astype(np.float)
            outslice = np.nan_to_num(outslice)
            xpoints, ypoints, zpoints = w.wcs_world2pix(spectrum['CRVAL2'],
                                                        spectrum['CRVAL3'],
                                                        spectrum['CRVAL1'], 0)
            tsys = spectrum['TSYS']
            if flagRMS:
                radiometer_rms = tsys / np.sqrt(np.abs(spectrum['CDELT1']) *
                                                spectrum['EXPOSURE'])
                scan_rms = prefac * np.median(np.abs(outslice[0:-2] -
                                                        outslice[2:]))

                if scan_rms > 1.25 * radiometer_rms:
                    tsys = 0 # Blank spectrum
            if flagRipple:
                scan_rms = prefac * np.median(np.abs(outslice[0:-2] -
                                                     outslice[2:]))
                ripple = prefac * sqrt2 * np.median(np.abs(outslice))

                if ripple > 2 * scan_rms:
                    tsys = 0 # Blank spectrum
                
            if (tsys > 10) and (xpoints > 0) and (xpoints < naxis1) \
                    and (ypoints > 0) and (ypoints < naxis2):
                pixelWeight, Index = gridFunction(xmat, ymat,
                                                  xpoints, ypoints,
                                                  pixPerBeam)
                vector = np.outer(outslice * spectrum_wt,
                                  pixelWeight / tsys**2)
                wts = pixelWeight / tsys**2
                outCube[:, ymat[Index], xmat[Index]] += vector
                outWts[ymat[Index], xmat[Index]] += wts
        # Temporarily do a file write for every batch of scans.
        outWtsTemp = np.copy(outWts)
        outWtsTemp.shape = (1,) + outWtsTemp.shape
        outCubeTemp = np.copy(outCube)
        outCubeTemp /= outWtsTemp

        hdr = fits.Header(w.to_header())
        hdr = addHeader_nonStd(hdr, beamSize, s[1].data[0])
        #
        hdu = fits.PrimaryHDU(outCubeTemp, header=hdr)
        hdu.writeto(outdir + '/' + outname + '.fits', clobber=True)

    outWts.shape = (1,) + outWts.shape
    outCube /= outWts



    # Create basic fits header from WCS structure
    hdr = fits.Header(w.to_header())
    # Add non standard fits keyword
    hdr = addHeader_nonStd(hdr, beamSize, s[1].data[0])
    # Adds history message
    # try:
    #    hdr.add_history(history_message)
    # except UnboundLocalError:
    #    pass
    hdr.add_history('Using GBTPIPE gridder version {0}'.format(__version__))
    hdu = fits.PrimaryHDU(outCube, header=hdr)
    hdu.writeto(outdir + '/' + outname + '.fits', clobber=True)

    w2 = w.dropaxis(2)
    hdr2 = fits.Header(w2.to_header())
    hdu2 = fits.PrimaryHDU(outWts, header=hdr2)
    hdu2.writeto(outdir + '/' + outname + '_wts.fits', clobber=True)

    if rebase:
        if rebaseorder is None:
            rebaseorder = blorder
        if 'NH3_11' in outname:
            Baseline.rebaseline(outdir + '/' + outname + '.fits',
                                windowFunction=Baseline.ammoniaWindow,
                                line='oneone', blorder=rebaseorder,
                                **kwargs)

        elif 'NH3_22' in outname:
            winfunc = baseline.ammoniaWindow
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
